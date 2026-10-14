# Copyright (c) IDEA, Inc. and its affiliates.
"""SimCLRv2 ResNet SSL backbone -- a real multi-level encoder for Detectron2.

SimCLR contributes only the ResNet image encoder; its projection head never
enters the downstream model. Because the SimCLRv2 network (ResNet-D stem,
selective-kernel bottlenecks, average-pool shortcut projections) is *not*
layer-wise identical to Detectron2's ResNet, we keep the native SimCLRv2 encoder
and expose all four residual stages explicitly::

    stem + stage1 -> res2 (stride  4, 256 ch for width=1)
           stage2 -> res3 (stride  8, 512 ch)
           stage3 -> res4 (stride 16, 1024 ch)
           stage4 -> res5 (stride 32, 2048 ch)

``res5`` is **never** upsampled to fake the lower levels -- that would erase the
low-level local information SimCLR provides, which small lesions, boundary masks
and small boxes depend on. Channel counts are read from the built encoder so a
``WIDTH_MULTIPLIER != 1`` variant cannot silently fall back to ResNet-50 sizes.

See ``Docs/ssl_backbone_adapter_implementation_plan.md`` §7.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from .simclrv2_resnet import SimCLRv2Backbone
from .ssl_common import (
    FEATURE_NAMES,
    SSL_LOG,
    FeaturePyramidBackbone,
    align_fpn_size_divisibility,
    checkpoint_digest,
    drop_train_only_keys,
    extract_state_dict,
    find_detector_backbone,
    load_checkpoint,
    load_state_flexible,
    resolve_ssl_settings,
    strip_common_prefixes,
    unwrap_subtree,
    validate_feature_contract,
)

__all__ = [
    "ResNetSSLBackbone",
    "ResNetAdapter",
    "build_resnet_ssl_backbone",
    "build_resnet_ssl_fpn_backbone",
    "load_resnet_ssl_pretrained_backbone",
    "resolve_resnet_arch",
]

#: Dotted paths to the SimCLRv2 encoder, probed in order when
#: ``SSL_CHECKPOINT_KEY`` is blank. ``model.backbone.net`` first: it is the
#: layout of the checkpoints in ``.checkpoints/pretrain/simclr``.
_SIMCLR_KEY_CANDIDATES = (
    "model.backbone.net",
    "backbone.net",
    "encoder.net",
    "resnet",
    "net",
    "model",
)

#: Wrapper prefixes stripped to reach the ``net`` sequential, longest last.
_SIMCLR_UNWRAP_ROOTS = ("net.", "backbone.net.", "encoder.net.", "module.net.")


def resolve_resnet_arch(cfg) -> dict:
    """Resolve the SimCLRv2 ResNet architecture from ``MODEL.RESNET_SSL``.

    ``MODEL.RESNET_SSL`` is used only when it was configured explicitly
    (``DEPTH > 0``); otherwise ``MODEL.RESNETS`` supplies ``DEPTH`` and
    ``NUM_GROUPS * WIDTH_PER_GROUP`` style defaults are ignored because the
    SimCLRv2 encoder has its own width convention.
    """
    node = getattr(cfg.MODEL, "RESNET_SSL", None)
    depth = int(getattr(node, "DEPTH", 0) or 0) if node is not None else 0
    if depth > 0:
        width = int(getattr(node, "WIDTH_MULTIPLIER", 1) or 1)
        sk_ratio = float(getattr(node, "SK_RATIO", 0.0625))
        stem_type = str(getattr(node, "STEM_TYPE", "simclrv2") or "simclrv2").lower()
        source = "MODEL.RESNET_SSL"
    else:
        depth = int(getattr(cfg.MODEL.RESNETS, "DEPTH", 50))
        width = 1
        sk_ratio = 0.0625
        stem_type = "simclrv2"
        source = "MODEL.RESNETS"

    if stem_type != "simclrv2":
        raise ValueError(
            f"ResNetSSLBackbone only supports STEM_TYPE='simclrv2', got {stem_type!r}."
        )
    if sk_ratio < 0:
        raise ValueError(f"MODEL.RESNET_SSL.SK_RATIO must be >= 0, got {sk_ratio}.")
    return {
        "depth": depth,
        "width_multiplier": width,
        "sk_ratio": sk_ratio,
        "stem_type": stem_type,
        "source": source,
        "freeze_at": int(getattr(cfg.MODEL.BACKBONE, "FREEZE_AT", 0)),
    }


class ResNetAdapter(nn.Module):
    """Standardise the native SimCLRv2 stage outputs into the ``res2``-``res5`` contract.

    The SimCLRv2 stages already emit NCHW tensors, so the adapter only fixes the
    feature *names* and the channel/stride metadata; it deliberately contains no
    parameters and never fabricates a resolution it did not receive.
    """

    def __init__(self, stage_names, channels, strides):
        super().__init__()
        self.stage_names = list(stage_names)
        self.channels = dict(channels)
        self.strides = dict(strides)

    def forward(self, stage_features) -> dict:
        return {name: stage_features[name] for name in self.stage_names}


@BACKBONE_REGISTRY.register()
class ResNetSSLBackbone(FeaturePyramidBackbone):
    """SimCLRv2 ResNet encoder exposing ``res2``-``res5``."""

    #: Index of the first ``Blocks`` group inside ``encoder.net`` (0 is the stem).
    _FIRST_STAGE_INDEX = 1

    def __init__(self, cfg, input_shape):
        del input_shape
        arch = resolve_resnet_arch(cfg)
        out_features = list(
            getattr(cfg.MODEL.BACKBONE, "FEATURE_NAMES", FEATURE_NAMES) or FEATURE_NAMES
        )
        super().__init__(out_features)

        self.arch = arch
        self.encoder = SimCLRv2Backbone(
            depth=arch["depth"],
            width_multiplier=arch["width_multiplier"],
            sk_ratio=arch["sk_ratio"],
        )

        stages = list(self.encoder.net)  # [Stem, Blocks0, Blocks1, Blocks2, Blocks3]
        if len(stages) != 5:
            raise RuntimeError(
                f"Unexpected SimCLRv2 encoder layout: {len(stages)} top-level stages "
                "(expected 5: stem + 4 residual stages)."
            )
        self._stage_count = len(stages) - self._FIRST_STAGE_INDEX
        if self._stage_count != len(FEATURE_NAMES):
            raise RuntimeError(
                f"SimCLRv2 encoder produced {self._stage_count} stages; "
                f"expected {len(FEATURE_NAMES)}."
            )

        # Derive channels from the built encoder -- never hard-code ResNet-50 sizes.
        channels = [
            int(stages[self._FIRST_STAGE_INDEX + offset].channels_out)
            for offset in range(self._stage_count)
        ]
        strides = [4, 8, 16, 32]
        self.stage_channels = dict(zip(FEATURE_NAMES, channels))
        self._native_strides = dict(zip(FEATURE_NAMES, strides))
        self.adapter = ResNetAdapter(FEATURE_NAMES, self.stage_channels, self._native_strides)

        self._apply_freeze_at(arch["freeze_at"])

    # -- freezing ---------------------------------------------------------- #
    def _apply_freeze_at(self, freeze_at: int) -> None:
        if freeze_at <= 0:
            return
        if freeze_at >= 1:
            for parameter in self.encoder.net[0].parameters():
                parameter.requires_grad = False
        for index in range(min(freeze_at - 1, self._stage_count)):
            stage = self.encoder.net[self._FIRST_STAGE_INDEX + index]
            for parameter in stage.parameters():
                parameter.requires_grad = False

    # -- forward ----------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"ResNetSSLBackbone expects NCHW input, got {tuple(x.shape)}.")
        stage_features = {}
        # ``encoder.net[0]`` is the stem; run it before the four residual stages.
        h = self.encoder.net[0](x)
        for offset, name in enumerate(FEATURE_NAMES):
            h = self.encoder.net[self._FIRST_STAGE_INDEX + offset](h)
            stage_features[name] = h
        outputs = self.adapter(stage_features)
        outputs = {name: outputs[name] for name in self._out_features}
        validate_feature_contract(outputs, self._out_features, name="ResNetSSLBackbone")
        return outputs

    @property
    def size_divisibility(self) -> int:
        return 32


@BACKBONE_REGISTRY.register()
def build_resnet_ssl_backbone(cfg, input_shape):
    """MaskDINO path: raw ``res2``-``res5`` without an FPN."""
    return ResNetSSLBackbone(cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_resnet_ssl_fpn_backbone(cfg, input_shape):
    """Mask R-CNN path: SimCLRv2 ResNet wrapped in a Detectron2 FPN (P2-P6)."""
    bottom_up = ResNetSSLBackbone(cfg, input_shape)
    fpn = FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=getattr(cfg.MODEL.FPN, "FUSE_TYPE", "sum"),
    )
    align_fpn_size_divisibility(fpn, bottom_up)
    return fpn


def _looks_like_resnet_net(state) -> bool:
    """True when ``state`` is keyed like the SimCLRv2 ``net`` sequential.

    The stem lives at ``0.*`` and the four residual stages at ``1.*``-``4.*``,
    so requiring both ``0.`` and ``1.blocks.`` catches every plausible nesting
    level without depending on tensor shapes.
    """
    keys = list(state)
    return any(key.startswith("0.") for key in keys) and any(
        key.startswith("1.blocks.") for key in keys
    )


def _unwrap_resnet_net(state):
    """Strip wrapper prefixes until the keys reach the ``net`` sequential."""
    dropped: list[str] = []
    for root in _SIMCLR_UNWRAP_ROOTS:
        state, removed = unwrap_subtree(state, root)
        dropped.extend(removed)
        if _looks_like_resnet_net(state):
            break
    return state, dropped


def _extract_simclr_state(checkpoint, dotted_key: str):
    """Resolve the SimCLRv2 encoder tensors and normalise their key prefixes.

    The checkpoints produced by different code bases nest the encoder
    differently, all of which are handled here:

    ``model.backbone.net.*``
        DentMaG multi-task checkpoints (``backbone.net`` plus ``heads.*``); the
        ``heads.*`` task heads are dropped as train-only tensors.
    ``resnet.net.*`` / ``model.net.*``
        lightly-ai HuggingFace conversions that also carry a 1000-class ``fc``.

    An explicitly configured ``SSL_CHECKPOINT_KEY`` is honoured, but if it is
    absent from the checkpoint the auto-detection runs anyway (with a warning)
    so a config stays usable across checkpoint flavours.
    """
    if dotted_key:
        try:
            state, selected = extract_state_dict(checkpoint, dotted_key)
        except (KeyError, TypeError) as error:
            SSL_LOG.warning(
                "SSL_CHECKPOINT_KEY=%r not found in the SimCLR checkpoint (%s); "
                "falling back to auto-detection.",
                dotted_key,
                error,
            )
        else:
            state, stripped = strip_common_prefixes(state)
            state, dropped = drop_train_only_keys(state)
            state, unwrapped = _unwrap_resnet_net(state)
            if _looks_like_resnet_net(state):
                return state, selected, stripped, dropped, unwrapped
            SSL_LOG.warning(
                "SSL_CHECKPOINT_KEY=%r resolved to %d tensors that are not keyed like "
                "the SimCLRv2 `net` sequential; falling back to auto-detection.",
                dotted_key,
                len(state),
            )

    for candidate in _SIMCLR_KEY_CANDIDATES:
        try:
            node, _ = extract_state_dict(checkpoint, candidate)
        except (KeyError, TypeError):
            continue
        state, stripped = strip_common_prefixes(node)
        state, dropped = drop_train_only_keys(state)
        state, unwrapped = _unwrap_resnet_net(state)
        if _looks_like_resnet_net(state):
            return state, candidate, stripped, dropped, unwrapped

    available = (
        sorted(checkpoint[ "model"].keys()) if "model" in checkpoint else list(checkpoint)
    ) if isinstance(checkpoint, dict) else []
    raise KeyError(
        "Unable to locate the SimCLRv2 encoder (`net.0.*` stem + `net.1..4.*` stages) "
        "in the checkpoint. Set MODEL.BACKBONE.SSL_CHECKPOINT_KEY explicitly, e.g. "
        "'model.backbone.net'. Top-level entries: "
        f"{sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__} "
        f"(keys of 'model': {available})"
    )


def load_resnet_ssl_pretrained_backbone(model, cfg, settings=None) -> dict:
    """Initialise a :class:`ResNetSSLBackbone` from a SimCLRv2 checkpoint."""
    settings = settings or resolve_ssl_settings(cfg)
    if not settings.is_ssl:
        return {}
    if settings.method != "simclr":
        raise ValueError(
            f"load_resnet_ssl_pretrained_backbone cannot handle SSL_METHOD={settings.method!r}."
        )

    checkpoint_path = Path(settings.weights).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "MODEL.BACKBONE.SSL_WEIGHTS is set to a SimCLR checkpoint that does not exist: "
            f"{checkpoint_path}"
        )

    _, _, bottom_up = find_detector_backbone(model)
    if not isinstance(bottom_up, ResNetSSLBackbone):
        raise TypeError(
            "MODEL.BACKBONE.NAME must build a ResNetSSLBackbone to load SimCLR weights; "
            f"got {type(bottom_up).__name__}."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    state, selected_key, stripped, dropped, unwrapped = _extract_simclr_state(
        checkpoint, settings.checkpoint_key
    )

    SSL_LOG.info(
        "SimCLR SSL init: file=%s sha1=%s key=%r stripped=%s dropped=%d unwrapped=%s "
        "depth=%d width=%d sk_ratio=%s channels=%s",
        checkpoint_path,
        checkpoint_digest(checkpoint_path),
        selected_key,
        stripped or "-",
        len(dropped),
        unwrapped or "-",
        bottom_up.arch["depth"],
        bottom_up.arch["width_multiplier"],
        bottom_up.arch["sk_ratio"],
        list(bottom_up.stage_channels.values()),
    )

    return load_state_flexible(
        bottom_up.encoder.net,
        state,
        strict=settings.strict,
        required_roots=("0.", "1.", "2.", "3.", "4."),
        name="ResNetSSLBackbone/simclr",
        log_fn=SSL_LOG.info,
    )
