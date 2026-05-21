# Copyright (c) IDEA, Inc. and its affiliates.
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from swin_unet import BasicBlock, PatchEmbedding


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
