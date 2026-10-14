# Copyright (c) IDEA, Inc. and its affiliates.
"""Swin SSL backbone -- one Detectron2 encoder for Swin-MAE and DentMag-Swin.

The pretrained Swin encoders we ship (``swinmae`` and ``dentmag_swin``) share the
exact architecture (Swin-Small: ``EMBED_DIM=96``, ``DEPTHS=[2, 2, 18, 2]``,
``NUM_HEADS=[3, 6, 12, 24]``, ``PATCH_SIZE=4``, ``WINDOW_SIZE=7``), hence a single
:class:`SwinSSLBackbone` serves both. The only behavioural difference is
positional encoding:

* ``swinmae``      -- ``POSITION_ENCODING=native_fixed``: the learned absolute
  table the Swin-MAE encoder adds right after the patch embedding. It is loaded
  at the checkpoint's native grid and bicubic-resampled to the runtime grid.
* ``dentmag_swin`` -- ``POSITION_ENCODING=dentmag_sincos``: the DentMag encoder
  adds a *dynamic* 2-D sin-cos positional embedding right after the patch
  embedding. This is preserved so a DentMag checkpoint is not merely loadable but
  actually runs in its original inference distribution.

Neither may be replaced by the other, and neither may be silently dropped for
convenience (doc §6). ``POSITION_ENCODING=none`` exists only for controls.

Both variants expose ``res2``-``res5`` captured **before** every ``PatchMerging``
downsample (strides 4/8/16/32), which is what the MaskDINO pixel decoder and the
Mask R-CNN FPN expect. See ``Docs/ssl_backbone_adapter_implementation_plan.md``
§6.
"""

from __future__ import annotations

import functools
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from .ssl_common import (
    FEATURE_NAMES,
    SSL_LOG,
    FeaturePyramidBackbone,
    align_fpn_size_divisibility,
    extract_state_dict,
    drop_train_only_keys,
    find_detector_backbone,
    load_checkpoint,
    load_state_flexible,
    resolve_ssl_settings,
    strip_common_prefixes,
    validate_feature_contract,
)
from .swin_unet import BasicBlock, PatchEmbedding

__all__ = [
    "SwinSSLBackbone",
    "SwinAdapter",
    "build_swin_ssl_backbone",
    "build_swin_ssl_fpn_backbone",
    "load_swin_ssl_pretrained_backbone",
    "resolve_swin_arch",
    "get_2d_sincos_pos_embed",
]

#: Stage channel multipliers of every Swin variant, relative to ``EMBED_DIM``.
_SWIN_CHANNEL_MULTIPLIER = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Dynamic 2-D sin-cos positional embedding (DentMag-Swin)
# --------------------------------------------------------------------------- #
def get_2d_sincos_pos_embed(embed_dim: int, grid_hw) -> torch.Tensor:
    """Return ``[1, grid_h*grid_w, embed_dim]`` 2-D sin-cos embeddings.

    Reimplemented locally (rather than imported from the pretraining project) so
    the backbone has no runtime dependency on it. ``grid_hw`` is an int or a
    ``(grid_h, grid_w)`` pair, matching DentMag's rectangular-grid support.
    """
    if isinstance(grid_hw, (int, np.integer)):
        grid_h = grid_w = int(grid_hw)
    else:
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sin-cos embedding, got {embed_dim}.")

    grid_w_idx, grid_h_idx = np.meshgrid(
        np.arange(grid_w, dtype=np.float32), np.arange(grid_h, dtype=np.float32)
    )
    grids = np.stack([grid_w_idx, grid_h_idx], axis=0).reshape(2, 1, grid_h, grid_w)

    half = embed_dim // 2
    # ``half // 2`` frequencies produce ``half`` channels after the sin/cos
    # concatenation; ``emb_h`` and ``emb_w`` then give the full ``embed_dim``.
    omega = np.arange(half // 2, dtype=np.float32) / (half / 2.0)
    omega = 1.0 / np.power(10000.0, omega)

    def _sincos(pos):
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = _sincos(grids[1])
    emb_w = _sincos(grids[0])
    pos = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(pos).unsqueeze(0).float()


# --------------------------------------------------------------------------- #
# Configuration resolution
# --------------------------------------------------------------------------- #
def resolve_swin_arch(cfg) -> dict:
    """Return the Swin architecture dictated by the config.

    ``MODEL.SWIN_SSL`` is honoured only when it was configured explicitly
    (detected through a non-empty ``DEPTHS``); otherwise ``MODEL.SWIN`` -- the
    historical location -- stays authoritative. This prevents a populated
    integer default from silently overriding an existing ``MODEL.SWIN`` setting.
    """
    ssl_node = getattr(cfg.MODEL, "SWIN_SSL", None)
    ssl_depths = list(getattr(ssl_node, "DEPTHS", []) or []) if ssl_node is not None else []
    source = "MODEL.SWIN_SSL" if ssl_depths else "MODEL.SWIN"
    node = ssl_node if ssl_depths else cfg.MODEL.SWIN

    settings = resolve_ssl_settings(cfg)
    position_encoding = str(getattr(node, "POSITION_ENCODING", "") or "").strip().lower()
    if position_encoding in ("", "auto"):
        # Derive the native behaviour of the method: Swin-MAE adds a learned
        # absolute table after the patch embedding, DentMag adds a dynamic
        # 2-D sin-cos embedding. Both are part of the pretrained encoder's
        # inference distribution, so neither may be silently dropped.
        position_encoding = {
            "dentmag_swin": "dentmag_sincos",
            "swinmae": "native_fixed",
        }.get(settings.method, "none")
    if position_encoding in ("learned", "absolute", "native"):
        position_encoding = "native_fixed"
    if position_encoding not in ("none", "native_fixed", "dentmag_sincos"):
        raise ValueError(
            f"Unsupported MODEL.SWIN_SSL.POSITION_ENCODING={position_encoding!r}; "
            "expected one of none | native_fixed | dentmag_sincos."
        )

    arch = {
        "patch_size": int(node.PATCH_SIZE),
        "embed_dim": int(node.EMBED_DIM),
        "depths": list(ssl_depths or list(cfg.MODEL.SWIN.DEPTHS)),
        "num_heads": list(node.NUM_HEADS),
        "window_size": int(node.WINDOW_SIZE),
        "mlp_ratio": float(node.MLP_RATIO),
        "qkv_bias": bool(node.QKV_BIAS),
        "drop_rate": float(node.DROP_RATE),
        "attn_drop_rate": float(node.ATTN_DROP_RATE),
        "drop_path_rate": float(node.DROP_PATH_RATE),
        "patch_norm": bool(node.PATCH_NORM),
        "position_encoding": position_encoding,
        "img_size": int(getattr(node, "IMG_SIZE", 0) or getattr(cfg.MODEL.SWIN, "IMG_SIZE", 0) or 224),
        "freeze_at": int(getattr(cfg.MODEL.BACKBONE, "FREEZE_AT", 0)),
        "source": source,
    }
    if len(arch["depths"]) != 4:
        raise ValueError(
            f"Swin backbone expects 4 stage depths, got {arch['depths']} "
            f"(from {source})."
        )
    return arch


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class SwinAdapter(nn.Module):
    """Normalise Swin stage features into the Detectron2 ``res2``-``res5`` contract.

    Responsibilities (and nothing else -- no checkpoint parsing, no task heads):

    * permute the encoder's ``BHWC`` stage features to ``NCHW``;
    * apply the per-stage ``LayerNorm`` over the channel dimension;
    * emit exactly the configured feature names.
    """

    def __init__(self, stage_names: Sequence[str], norms: nn.ModuleList):
        super().__init__()
        self.stage_names = list(stage_names)
        self.norms = norms

    def forward(self, stage_features: Mapping[str, torch.Tensor]) -> dict:
        outputs = {}
        for index, name in enumerate(self.stage_names):
            feature = stage_features[name]
            feature = self.norms[index](feature)
            outputs[name] = feature.permute(0, 3, 1, 2).contiguous()
        return outputs


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
class SwinSSLBackbone(FeaturePyramidBackbone):
    """Swin encoder exposing ``res2``-``res5`` for Mask R-CNN and MaskDINO."""

    def __init__(self, cfg, input_shape):
        del input_shape
        arch = resolve_swin_arch(cfg)
        out_features = list(
            getattr(cfg.MODEL.BACKBONE, "FEATURE_NAMES", FEATURE_NAMES) or FEATURE_NAMES
        )
        super().__init__(out_features)

        self.arch = arch
        self.patch_size = arch["patch_size"]
        self.embed_dim = arch["embed_dim"]
        self.depths = arch["depths"]
        self.num_heads = arch["num_heads"]
        self.window_size = arch["window_size"]
        self.num_layers = len(self.depths)
        self.position_encoding = arch["position_encoding"]

        norm_layer = functools.partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_c=3,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if arch["patch_norm"] else None,
        )
        self.layers = nn.ModuleList(
            [
                BasicBlock(
                    index=index,
                    depths=self.depths,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=arch["drop_path_rate"],
                    window_size=self.window_size,
                    mlp_ratio=arch["mlp_ratio"],
                    qkv_bias=arch["qkv_bias"],
                    drop_rate=arch["drop_rate"],
                    attn_drop_rate=arch["attn_drop_rate"],
                    norm_layer=norm_layer,
                    patch_merging=index != self.num_layers - 1,
                )
                for index in range(self.num_layers)
            ]
        )

        self.num_features = [self.embed_dim * 2**i for i in range(self.num_layers)]
        self.stage_names = [f"res{i + 2}" for i in range(self.num_layers)]
        self.stage_channels = {
            name: self.num_features[i] for i, name in enumerate(self.stage_names)
        }
        self._native_strides = {
            name: self.patch_size * 2**i for i, name in enumerate(self.stage_names)
        }
        norms = nn.ModuleList([norm_layer(channels) for channels in self.num_features])
        self.adapter = SwinAdapter(self.stage_names, norms)

        # ``native_fixed``: the learned absolute table that Swin-MAE adds right
        # after the patch embedding. It is *not* dropped -- the pretrained table
        # is part of the encoder's inference distribution. The parameter is
        # created at the fallback grid size and replaced at load time by
        # ``install_pos_embed`` with the checkpoint's table at its native grid,
        # so no information is lost to a premature interpolation.
        self._pos_embed_cache: dict = {}
        if self.position_encoding == "native_fixed":
            base = max(int(arch.get("img_size", 0) or 224), self.patch_size)
            base_tokens = (base // self.patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, base_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self._apply_freeze_at(arch["freeze_at"])

    # -- learned absolute positional table (Swin-MAE native behaviour) ------ #
    def install_pos_embed(self, tensor: torch.Tensor) -> tuple[int, int]:
        """Adopt a pretrained absolute positional table at its own grid size.

        Returns the ``(grid_h, grid_w)`` the table was trained on. Accepts both
        ``[1, L, C]`` (SimMIM / Swin-MAE) and ``[1, C, H, W]`` (DentMaG-MAE)
        layouts.
        """
        if self.position_encoding != "native_fixed":
            raise ValueError(
                "install_pos_embed requires POSITION_ENCODING='native_fixed', got "
                f"{self.position_encoding!r}."
            )
        if tensor.dim() == 4:
            tensor = tensor.flatten(2).transpose(1, 2).contiguous()
        tensor = tensor.reshape(1, -1, self.embed_dim).detach().clone()
        tokens = int(tensor.shape[1])
        grid = math.isqrt(tokens)
        if grid * grid != tokens:
            raise ValueError(
                f"native_fixed positional table must come from a square patch grid; "
                f"got {tokens} tokens which is not a perfect square."
            )
        self.pos_embed = nn.Parameter(tensor)
        self._pos_embed_cache = {}
        return grid, grid

    def _resample_pos_embed(self, height: int, width: int) -> torch.Tensor:
        """Return ``[H, W, C]`` for the runtime patch grid, bicubic-resampled."""
        cached = self._pos_embed_cache.get((height, width))
        if cached is not None:
            return cached

        table = self.pos_embed
        if table is None or table.numel() == 0:
            raise RuntimeError(
                "SwinSSLBackbone was configured with POSITION_ENCODING='native_fixed' "
                "but no positional table is available."
            )
        tokens = int(table.shape[1])
        base = math.isqrt(tokens)
        if base * base != tokens:
            raise ValueError(
                f"native_fixed positional table has {tokens} tokens (not a square grid)."
            )
        if (base, base) == (height, width):
            pos = table.reshape(height, width, self.embed_dim)
        else:
            pos = table.transpose(1, 2).reshape(1, self.embed_dim, base, base)
            pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
            pos = pos.reshape(1, self.embed_dim, height, width).permute(0, 2, 3, 1)
            pos = pos.reshape(height, width, self.embed_dim)
        self._pos_embed_cache[(height, width)] = pos
        return pos

    # -- Detectron2 ``MODEL.WEIGHTS`` bridging ------------------------------ #
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Accept a raw Swin-MAE / DentMag checkpoint through ``MODEL.WEIGHTS`` too.

        Adds remapped ``patch_embed.*`` / ``layers.*`` entries so Detectron2's
        standard loading path also works when a bare encoder checkpoint (rather
        than a complete detector) is pointed at by ``MODEL.WEIGHTS``.
        """
        self._add_encoder_key_aliases(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _add_encoder_key_aliases(self, state_dict, prefix):
        if any(key.startswith(prefix + "patch_embed.") for key in state_dict):
            return
        source_prefixes = (
            "module.encoder.",
            "model.encoder.",
            "encoder.",
            "encoder_teacher.",
            "teacher.",
            "module.",
            "model.",
            "",
        )
        target_roots = ("patch_embed.", "layers.")
        for key, value in list(state_dict.items()):
            for source_prefix in source_prefixes:
                if not key.startswith(source_prefix):
                    continue
                stripped = key[len(source_prefix) :]
                if stripped.startswith(target_roots):
                    state_dict.setdefault(prefix + stripped, value)
                break

    # -- freezing ---------------------------------------------------------- #
    def _apply_freeze_at(self, freeze_at: int) -> None:
        """Freeze the first ``freeze_at`` encoder blocks (0 = stem, 1..4 = stages)."""
        if freeze_at <= 0:
            return
        if freeze_at >= 1:
            for parameter in self.patch_embed.parameters():
                parameter.requires_grad = False
        for index in range(min(freeze_at - 1, self.num_layers)):
            for parameter in self.layers[index].parameters():
                parameter.requires_grad = False

    # -- forward ----------------------------------------------------------- #
    def _add_position_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.position_encoding == "none":
            return x
        _, height, width, channels = x.shape
        if self.position_encoding == "dentmag_sincos":
            pos = get_2d_sincos_pos_embed(channels, (height, width)).to(
                device=x.device, dtype=x.dtype
            )
            return x + pos.view(1, height, width, channels)
        pos = self._resample_pos_embed(height, width).to(device=x.device, dtype=x.dtype)
        return x + pos.view(1, height, width, channels)

    def _pad_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the BHWC patch grid so window attention stays exact.

        ``swin_unet``'s window attention requires the grid to be divisible by the
        window size at every stage, i.e. by ``window_size * 2**(num_layers-1)``.
        ``size_divisibility`` already requests that from the input pipeline, so
        this is a safety net for callers that bypass it; it pads each axis
        independently so the aspect ratio is preserved.
        """
        _, height, width, _ = x.shape
        divisor = self.window_size * 2 ** (self.num_layers - 1)
        pad_h = (-height) % divisor
        pad_w = (-width) % divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        return x

    def _run_stage(self, layer, x):
        for block in layer.blocks:
            x = block(x)
        stage_out = x
        if layer.downsample is not None:
            x = layer.downsample(x)
        return stage_out, x

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"SwinSSLBackbone expects NCHW input, got {tuple(x.shape)}.")
        x = self.patch_embed(x)
        x = self._add_position_encoding(x)
        x = self._pad_patch_grid(x)

        stage_features = {}
        for index, layer in enumerate(self.layers):
            stage_out, x = self._run_stage(layer, x)
            stage_features[self.stage_names[index]] = stage_out

        outputs = self.adapter(stage_features)
        outputs = {name: outputs[name] for name in self._out_features}
        validate_feature_contract(outputs, self._out_features, name="SwinSSLBackbone")
        return outputs

    @property
    def size_divisibility(self) -> int:
        return self.patch_size * 2 ** (self.num_layers - 1) * self.window_size


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
def build_swin_ssl_backbone(cfg, input_shape):
    """MaskDINO path: raw ``res2``-``res5`` without an FPN."""
    return SwinSSLBackbone(cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_swin_ssl_fpn_backbone(cfg, input_shape):
    """Mask R-CNN path: Swin SSL encoder wrapped in a Detectron2 FPN (P2-P6)."""
    bottom_up = SwinSSLBackbone(cfg, input_shape)
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


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def _extract_swin_state(checkpoint, method: str, configured_key: str):
    """Extract the encoder state dict, honouring the method's key priority."""
    if configured_key:
        return extract_state_dict(checkpoint, configured_key)

    if method == "dentmag_swin":
        for candidate in ("encoder_teacher", "encoder"):
            try:
                return extract_state_dict(checkpoint, candidate)
            except (KeyError, TypeError):
                continue
        return extract_state_dict(checkpoint, "")  # auto-detect fallback

    # Swin-MAE: ``model`` holds the full MAE (encoder + decoder); the decoder
    # tensors are removed right after.
    for candidate in ("model", "state_dict"):
        try:
            return extract_state_dict(checkpoint, candidate)
        except (KeyError, TypeError):
            continue
    return extract_state_dict(checkpoint, "")


def load_swin_ssl_pretrained_backbone(model, cfg, settings=None) -> dict:
    """Initialise a :class:`SwinSSLBackbone` from a Swin-MAE / DentMag-Swin checkpoint."""
    settings = settings or resolve_ssl_settings(cfg)
    if not settings.is_ssl:
        return {}
    if settings.method not in ("swinmae", "dentmag_swin"):
        raise ValueError(
            f"load_swin_ssl_pretrained_backbone cannot handle SSL_METHOD={settings.method!r}."
        )

    from .ssl_common import checkpoint_digest

    checkpoint_path = Path(settings.weights).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "MODEL.BACKBONE.SSL_WEIGHTS/USE_SSL_PRETRAINED is set but the file is missing: "
            f"{checkpoint_path}"
        )

    _, _, bottom_up = find_detector_backbone(model)
    if not isinstance(bottom_up, SwinSSLBackbone):
        raise TypeError(
            "Configured MODEL.BACKBONE.NAME must build a SwinSSLBackbone to load "
            f"Swin SSL weights; got {type(bottom_up).__name__}."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    state, selected_key = _extract_swin_state(
        checkpoint, settings.method, settings.checkpoint_key
    )
    state, stripped = strip_common_prefixes(state)
    state, dropped = drop_train_only_keys(state)

    # The learned absolute table is a plain tensor in the checkpoints; hand it
    # to the backbone separately so it is adopted at its *native* grid instead
    # of being rejected as a shape mismatch against the fallback grid.
    pos_note = "-"
    if bottom_up.position_encoding == "native_fixed":
        table = state.pop("pos_embed", None)
        if table is None:
            raise KeyError(
                "POSITION_ENCODING='native_fixed' (Swin-MAE) requires a 'pos_embed' "
                f"tensor, but none was found in {checkpoint_path} under key "
                f"{selected_key!r}. Use POSITION_ENCODING='none' only if the encoder "
                "was genuinely trained without a positional table."
            )
        grid = bottom_up.install_pos_embed(table)
        pos_note = f"{table.shape} -> grid {grid[0]}x{grid[1]}"
        # ``install_pos_embed`` owns the parameter now; put it back into the
        # state dict so the audit below reports it as loaded from the checkpoint
        # instead of listing it as randomly initialised.
        state["pos_embed"] = bottom_up.pos_embed.detach()
    else:
        state.pop("pos_embed", None)

    SSL_LOG.info(
        "Swin SSL init: method=%s file=%s sha1=%s key=%r stripped=%s dropped_train_only=%d "
        "arch=%s depths=%s embed=%d pos_enc=%s pos_embed=%s",
        settings.method,
        checkpoint_path,
        checkpoint_digest(checkpoint_path),
        selected_key,
        stripped or "-",
        len(dropped),
        bottom_up.arch["source"],
        bottom_up.depths,
        bottom_up.embed_dim,
        bottom_up.position_encoding,
        pos_note,
    )

    return load_state_flexible(
        bottom_up,
        state,
        strict=settings.strict,
        required_roots=("patch_embed.", "layers."),
        name=f"SwinSSLBackbone/{settings.method}",
        log_fn=SSL_LOG.info,
    )
