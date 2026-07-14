# Copyright (c) IDEA, Inc. and its affiliates.
from functools import partial
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from swin_unet import BasicBlock, PatchEmbedding


def _extract_checkpoint_state_dict(checkpoint, checkpoint_key):
    state_dict = checkpoint
    if checkpoint_key:
        for key in checkpoint_key.split("."):
            if not isinstance(state_dict, Mapping) or key not in state_dict:
                raise KeyError(
                    f"Swin-MAE checkpoint does not contain configured key "
                    f"{checkpoint_key!r}."
                )
            state_dict = state_dict[key]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "The configured Swin-MAE checkpoint entry must be a state dict, "
            f"got {type(state_dict).__name__}."
        )
    return state_dict


def _load_checkpoint(path):
    """Load a trusted local training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # ``weights_only`` was added in newer PyTorch versions.
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as full_load_error:
            raise RuntimeError(f"Unable to load Swin-MAE checkpoint: {path}") from full_load_error


def load_swin_mae_pretrained_backbone(model, cfg):
    """Initialize only ``D2SwinMAEBackbone`` from a Swin-MAE checkpoint.

    This is intentionally separate from Detectron2's ``MODEL.WEIGHTS`` path:
    the latter restores a complete detector, while this function only maps
    patch embedding and hierarchical Swin encoder stages.
    """
    if not cfg.MODEL.BACKBONE.USE_PRETRAINED:
        return
    if cfg.MODEL.BACKBONE.PRETRAINED_FORMAT != "swin_mae":
        raise ValueError(
            "MODEL.BACKBONE.PRETRAINED_FORMAT must be 'swin_mae' for "
            "D2SwinMAEBackbone."
        )
    if cfg.MODEL.BACKBONE.NAME not in {
        "D2SwinMAEBackbone",
        "build_swin_mae_fpn_backbone",
    }:
        raise ValueError(
            "MODEL.BACKBONE.USE_PRETRAINED requires "
            "a D2SwinMAEBackbone-based backbone."
        )

    checkpoint_path = Path(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS).expanduser()
    if not str(checkpoint_path) or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "MODEL.BACKBONE.USE_PRETRAINED is true, but "
            "MODEL.BACKBONE.PRETRAINED_WEIGHTS is not a valid file: "
            f"{checkpoint_path}"
        )

    detector_backbone = model.module.backbone if hasattr(model, "module") else model.backbone
    backbone = getattr(detector_backbone, "bottom_up", detector_backbone)
    if not isinstance(backbone, D2SwinMAEBackbone):
        raise TypeError(
            "Configured D2SwinMAEBackbone was not built; cannot load "
            "Swin-MAE encoder weights."
        )

    checkpoint = _load_checkpoint(checkpoint_path)
    source_state = _extract_checkpoint_state_dict(
        checkpoint, cfg.MODEL.BACKBONE.PRETRAINED_CHECKPOINT_KEY
    )
    remapped = backbone.remap_swin_mae_encoder_state_dict(source_state)
    target_state = backbone.state_dict()
    compatible = {}
    shape_mismatches = []
    for key, value in remapped.items():
        if key not in target_state:
            continue
        if target_state[key].shape != value.shape:
            shape_mismatches.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        compatible[key] = value

    if shape_mismatches:
        details = ", ".join(
            f"{key}: checkpoint{source_shape} != model{target_shape}"
            for key, source_shape, target_shape in shape_mismatches[:5]
        )
        raise ValueError(
            "Swin-MAE encoder architecture does not match MODEL.SWIN. " + details
        )

    required_roots = ("patch_embed.",) + tuple(
        f"layers.{index}." for index in range(backbone.num_layers)
    )
    missing_roots = [
        root for root in required_roots if not any(key.startswith(root) for key in compatible)
    ]
    if missing_roots:
        raise ValueError(
            "Swin-MAE checkpoint did not provide all encoder stages: "
            + ", ".join(missing_roots)
        )

    incompatible = backbone.load_state_dict(compatible, strict=False)
    ignored_count = len(source_state) - len(remapped)
    logging.getLogger("detectron2").info(
        "Initialized Swin-MAE image encoder from %s: loaded %d tensors; "
        "ignored %d MAE-only tensors; %d backbone tensors remain randomly initialized "
        "(e.g. detection output norms).",
        checkpoint_path,
        len(compatible),
        ignored_count,
        len(incompatible.missing_keys),
    )


@BACKBONE_REGISTRY.register()
class D2SwinMAEBackbone(Backbone):
    """Use the Swin-MAE encoder as a Detectron2 multi-scale backbone.

    The original SwinMAE class keeps its decoder/reconstruction path unchanged.
    This wrapper only rebuilds the encoder-side modules and exposes stage
    features in the format expected by MaskDINO.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        del input_shape
        self.patch_size = cfg.MODEL.SWIN.PATCH_SIZE
        self.embed_dim = cfg.MODEL.SWIN.EMBED_DIM
        self.depths = cfg.MODEL.SWIN.DEPTHS
        self.num_heads = cfg.MODEL.SWIN.NUM_HEADS
        self.window_size = cfg.MODEL.SWIN.WINDOW_SIZE
        self.num_layers = len(self.depths)
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_c=3,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if cfg.MODEL.SWIN.PATCH_NORM else None,
        )
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                BasicBlock(
                    index=i,
                    depths=self.depths,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=cfg.MODEL.SWIN.DROP_PATH_RATE,
                    window_size=self.window_size,
                    mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                    qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
                    drop_rate=cfg.MODEL.SWIN.DROP_RATE,
                    attn_drop_rate=cfg.MODEL.SWIN.ATTN_DROP_RATE,
                    norm_layer=norm_layer,
                    patch_merging=False if i == self.num_layers - 1 else True,
                )
            )

        self.num_features = [self.embed_dim * 2**i for i in range(self.num_layers)]
        self.stage_names = [f"res{i + 2}" for i in range(self.num_layers)]
        self.stage_strides = {
            name: self.patch_size * 2**i for i, name in enumerate(self.stage_names)
        }
        self.stage_channels = {
            name: self.num_features[i] for i, name in enumerate(self.stage_names)
        }
        for i, channels in enumerate(self.num_features):
            setattr(self, f"norm{i}", norm_layer(channels))

    def _pad_patch_grid(self, x):
        """Pad BHWC patch features for Swin-MAE's window attention implementation."""
        _, height, width, _ = x.shape
        divisor = self.window_size * 2 ** (self.num_layers - 1)
        target = max(height, width)
        target = ((target + divisor - 1) // divisor) * divisor
        pad_h = target - height
        pad_w = target - width
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

    def forward(self, x):
        assert x.dim() == 4, f"D2SwinMAEBackbone expects NCHW input. Got {x.shape}."
        x = self.patch_embed(x)
        x = self._pad_patch_grid(x)

        outputs = {}
        for i, layer in enumerate(self.layers):
            stage_out, x = self._run_stage(layer, x)
            name = self.stage_names[i]
            if name in self._out_features:
                stage_out = getattr(self, f"norm{i}")(stage_out)
                outputs[name] = stage_out.permute(0, 3, 1, 2).contiguous()
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.stage_channels[name],
                stride=self.stage_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self.patch_size * 2 ** (self.num_layers - 1) * self.window_size

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
        self._add_swin_mae_encoder_keys(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _add_swin_mae_encoder_keys(self, state_dict, prefix):
        if any(k.startswith(prefix + "patch_embed.") for k in state_dict):
            return

        source_prefixes = ("module.encoder.", "model.encoder.", "encoder.", "module.", "model.", "")
        target_roots = ("patch_embed.", "layers.")
        norm_roots = tuple(f"norm{i}." for i in range(self.num_layers))
        for key, value in list(state_dict.items()):
            for source_prefix in source_prefixes:
                if not key.startswith(source_prefix):
                    continue
                stripped = key[len(source_prefix) :]
                if stripped.startswith(target_roots) or stripped.startswith(norm_roots):
                    mapped_key = prefix + stripped
                    state_dict.setdefault(mapped_key, value)
                    break

    def remap_swin_mae_encoder_state_dict(self, state_dict):
        """Return only MAE encoder tensors, with checkpoint prefixes removed."""
        source_prefixes = ("module.encoder.", "model.encoder.", "encoder.", "module.", "model.", "")
        target_roots = ("patch_embed.", "layers.")
        remapped = {}
        for key, value in state_dict.items():
            for source_prefix in source_prefixes:
                if not key.startswith(source_prefix):
                    continue
                stripped = key[len(source_prefix):]
                if stripped.startswith(target_roots):
                    remapped.setdefault(stripped, value)
                break
        return remapped


@BACKBONE_REGISTRY.register()
def build_swin_mae_fpn_backbone(cfg, input_shape):
    """Build a Mask R-CNN-ready FPN over all four Swin-MAE encoder stages."""
    bottom_up = D2SwinMAEBackbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    return FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=getattr(cfg.MODEL.FPN, "FUSE_TYPE", "sum"),
    )
