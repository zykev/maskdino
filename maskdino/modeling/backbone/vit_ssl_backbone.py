# Copyright (c) IDEA, Inc. and its affiliates.
"""ViT SSL backbone (DINOv2 / ViT-MAE / DentMag-ViT) with a shared pyramid adapter.

A ViT natively emits a single-scale token grid, which neither the Mask R-CNN FPN
nor the MaskDINO pixel decoder can consume. Instead of the heavy DINO
``ViTAdapter`` (spatial prior + deformable attention + interaction blocks, all
randomly initialised), this module uses one *lightweight* adapter that is
parameter-identical across every ViT method::

    block 3  tokens -> 1x1 conv -> LN -> resize -> res2  (stride  4)
    block 6  tokens -> 1x1 conv -> LN -> resize -> res3  (stride  8)
    block 9  tokens -> 1x1 conv -> LN -> resize -> res4  (stride 16)
    block 12 tokens -> 1x1 conv -> LN -> resize -> res5  (stride 32)

The ViT encoder itself is architecture-compatible with DINOv2 (identical
sub-module names), so a DINOv2 ``teacher`` state dict loads with
``load_state_dict`` and no extra conversion.

Token maps are reshaped with the *true* patch grid ``(H/patch, W/patch)`` -- never
``sqrt(num_tokens)`` -- so non-square inputs work. Padding is
``lcm(patch_size, 32)`` (DINOv2 p14 -> 224, ViT-MAE p16 -> 32).

See ``Docs/ssl_backbone_adapter_implementation_plan.md`` §8.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from .ssl_common import (
    FEATURE_NAMES,
    FEATURE_STRIDES,
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
    resample_vit_pos_embed,
    strip_common_prefixes,
    validate_feature_contract,
)

__all__ = [
    "ViTSSLBackbone",
    "ViTAdapter",
    "build_vit_ssl_backbone",
    "build_vit_ssl_fpn_backbone",
    "load_vit_ssl_pretrained_backbone",
    "resolve_vit_arch",
    "DINOv2ViT",
]

#: Architecture presets keyed by ``MODEL.VIT_SSL.ARCH``.
_VIT_PRESETS = {
    "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 14},
    "vit_small_p16": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 16},
    "vit_base_p16": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},
}


# --------------------------------------------------------------------------- #
# ViT building blocks (sub-module names match DINOv2 exactly)
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """Conv wrapped in ``.proj`` so keys match DINOv2's ``patch_embed.proj.*``."""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"ViT patch_size={self.patch_size} requires an input divisible by it, "
                f"got {height}x{width}. Set INPUT.SIZE_DIVISIBILITY to "
                f"lcm(patch_size, 32)."
            )
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]


class LayerScale(nn.Module):
    """Per-channel residual scale (loads DINOv2 ``ls1.gamma`` / ``ls2.gamma``)."""

    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, length, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1.0, norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DINOv2ViT(nn.Module):
    """Plain ViT encoder whose sub-module names match the DINOv2 checkpoints."""

    def __init__(
        self,
        patch_size=14,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_register_tokens=0,
        init_values=1.0,
        img_size=448,
        norm_eps=1e-6,
        out_blocks=(3, 6, 9, 12),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.out_blocks = list(out_blocks)

        base_grid = int(img_size) // int(patch_size)
        if base_grid * patch_size != int(img_size):
            base_grid = int(math.ceil(int(img_size) / int(patch_size)))
        self.base_grid = base_grid

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + base_grid * base_grid, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, init_values, norm_eps)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    # -- helpers ----------------------------------------------------------- #
    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.num_register_tokens

    def pos_encoding_for(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Return ``[1, num_prefix + h*w, C]`` position encoding for the current grid."""
        stored_grid = int(round(math.sqrt(self.pos_embed.shape[1] - 1)))
        if grid_h == stored_grid and grid_w == stored_grid:
            return self.pos_embed
        SSL_LOG.debug(
            "ViT pos_embed resampled from %dx%d to %dx%d",
            stored_grid,
            stored_grid,
            grid_h,
            grid_w,
        )
        return resample_vit_pos_embed(
            self.pos_embed,
            (grid_h, grid_w),
            num_prefix_tokens=1,
        )

    def _strip_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, self.num_prefix_tokens :]

    # -- forward ----------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """Run the encoder and return ``(selected_token_maps, grid_hw)``.

        ``selected_token_maps`` is a list ordered like :attr:`out_blocks`, each
        entry ``[B, h*w, C]`` with prefix tokens already removed.
        """
        batch, _, height, width = x.shape
        tokens = self.patch_embed(x)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(batch, -1, -1)
            tokens = torch.cat([tokens[:, :1], registers, tokens[:, 1:]], dim=1)
        tokens = tokens + self.pos_encoding_for(grid_h, grid_w).to(tokens.dtype)

        wanted = set(self.out_blocks)
        collected: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if index in wanted:
                collected[index] = self.norm(self._strip_prefix_tokens(tokens))

        missing = [index for index in self.out_blocks if index not in collected]
        if missing:
            raise ValueError(
                f"OUT_BLOCKS {missing} exceed the ViT depth {len(self.blocks)}."
            )
        return [collected[index] for index in self.out_blocks], (grid_h, grid_w)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class ViTAdapter(nn.Module):
    """Token-to-feature-pyramid adapter shared by every ViT family.

    Each selected token map is reshaped to its true grid, projected to
    ``pyramid_channels`` with a 1x1 conv, normalised, optionally smoothed by a
    3x3 conv, then bilinearly resized to the target stride. The 3x3 smoothing is
    off by default and, when enabled, is structurally identical for all ViT
    methods (see the plan §8.3).
    """

    def __init__(
        self,
        embed_dim: int,
        pyramid_channels: int = 256,
        stride_per_level=(4, 8, 16, 32),
        use_smoothing: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.pyramid_channels = pyramid_channels
        self.strides = tuple(stride_per_level)
        self.projections = nn.ModuleList(
            [nn.Conv2d(embed_dim, pyramid_channels, kernel_size=1) for _ in self.strides]
        )
        self.norms = nn.ModuleList(
            [LayerNorm2d(pyramid_channels, norm_eps) for _ in self.strides]
        )
        self.smoothers = (
            nn.ModuleList(
                [
                    nn.Conv2d(
                        pyramid_channels,
                        pyramid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=1,
                    )
                    for _ in self.strides
                ]
            )
            if use_smoothing
            else None
        )

    def forward(self, token_maps, token_grid, image_hw):
        grid_h, grid_w = token_grid
        image_h, image_w = image_hw
        outputs = {}
        for level, name in enumerate(FEATURE_NAMES):
            tokens = token_maps[level]
            batch, num_tokens, channels = tokens.shape
            if num_tokens != grid_h * grid_w:
                raise ValueError(
                    f"Token map {level} has {num_tokens} tokens but the patch grid is "
                    f"{grid_h}x{grid_w}={grid_h * grid_w}."
                )
            feature = tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
            feature = self.projections[level](feature)
            feature = self.norms[level](feature)
            if self.smoothers is not None:
                feature = self.smoothers[level](feature)

            stride = self.strides[level]
            target_h = int(math.ceil(image_h / stride))
            target_w = int(math.ceil(image_w / stride))
            if feature.shape[-2:] != (target_h, target_w):
                feature = F.interpolate(
                    feature,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs[name] = feature.contiguous()
        return outputs


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def resolve_vit_arch(cfg) -> dict:
    node = getattr(cfg.MODEL, "VIT_SSL", None)
    if node is None:
        raise ValueError(
            "MODEL.VIT_SSL is required for a ViT SSL backbone; call "
            "add_ssl_backbone_config(cfg) when building the config."
        )
    arch_name = str(getattr(node, "ARCH", "") or "").lower()
    preset = dict(_VIT_PRESETS.get(arch_name, {}))

    patch_size = int(getattr(node, "PATCH_SIZE", 0) or preset.get("patch_size", 14))
    embed_dim = int(getattr(node, "EMBED_DIM", 0) or preset.get("embed_dim", 384))
    depth = int(getattr(node, "DEPTH", 0) or preset.get("depth", 12))
    num_heads = int(getattr(node, "NUM_HEADS", 0) or preset.get("num_heads", 6))
    out_blocks = list(getattr(node, "OUT_BLOCKS", []) or [3, 6, 9, 12])
    pyramid_channels = int(getattr(node, "PYRAMID_CHANNELS", 256) or 256)
    num_register_tokens = int(getattr(node, "NUM_REGISTER_TOKENS", 0) or 0)
    img_size = int(getattr(node, "IMG_SIZE", 448) or 448)
    use_smoothing = bool(getattr(node, "PYRAMID_SMOOTHING", False))

    if len(out_blocks) != len(FEATURE_NAMES):
        raise ValueError(
            f"MODEL.VIT_SSL.OUT_BLOCKS must contain {len(FEATURE_NAMES)} block indices, "
            f"got {out_blocks}."
        )
    if sorted(out_blocks) != list(out_blocks):
        raise ValueError(f"MODEL.VIT_SSL.OUT_BLOCKS must be ascending, got {out_blocks}.")
    if out_blocks[-1] > depth:
        raise ValueError(
            f"MODEL.VIT_SSL.OUT_BLOCKS {out_blocks} exceed the ViT depth {depth}."
        )

    return {
        "arch": arch_name or "custom",
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "out_blocks": out_blocks,
        "pyramid_channels": pyramid_channels,
        "num_register_tokens": num_register_tokens,
        "img_size": img_size,
        "use_smoothing": use_smoothing,
        "freeze_at": int(getattr(cfg.MODEL.BACKBONE, "FREEZE_AT", 0)),
    }


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
@BACKBONE_REGISTRY.register()
class ViTSSLBackbone(FeaturePyramidBackbone):
    """ViT encoder + shared token-to-feature-pyramid adapter."""

    def __init__(self, cfg, input_shape):
        del input_shape
        arch = resolve_vit_arch(cfg)
        out_features = list(
            getattr(cfg.MODEL.BACKBONE, "FEATURE_NAMES", FEATURE_NAMES) or FEATURE_NAMES
        )
        super().__init__(out_features)

        self.arch = arch
        self.patch_size = arch["patch_size"]
        self.encoder = DINOv2ViT(
            patch_size=arch["patch_size"],
            embed_dim=arch["embed_dim"],
            depth=arch["depth"],
            num_heads=arch["num_heads"],
            num_register_tokens=arch["num_register_tokens"],
            img_size=arch["img_size"],
            out_blocks=arch["out_blocks"],
        )
        self.adapter = ViTAdapter(
            embed_dim=arch["embed_dim"],
            pyramid_channels=arch["pyramid_channels"],
            stride_per_level=[FEATURE_STRIDES[name] for name in FEATURE_NAMES],
            use_smoothing=arch["use_smoothing"],
        )
        self.stage_channels = {
            name: arch["pyramid_channels"] for name in FEATURE_NAMES
        }
        self._native_strides = {name: FEATURE_STRIDES[name] for name in FEATURE_NAMES}

        self._apply_freeze_at(arch["freeze_at"])

    def _apply_freeze_at(self, freeze_at: int) -> None:
        if freeze_at <= 0:
            return
        for parameter in self.encoder.patch_embed.parameters():
            parameter.requires_grad = False
        for index in range(min(freeze_at - 1, len(self.encoder.blocks))):
            for parameter in self.encoder.blocks[index].parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"ViTSSLBackbone expects NCHW input, got {tuple(x.shape)}.")
        token_maps, token_grid = self.encoder(x)
        outputs = self.adapter(token_maps, token_grid, x.shape[-2:])
        outputs = {name: outputs[name] for name in self._out_features}
        validate_feature_contract(outputs, self._out_features, name="ViTSSLBackbone")
        return outputs

    @property
    def size_divisibility(self) -> int:
        return math.lcm(self.patch_size, 32)


@BACKBONE_REGISTRY.register()
def build_vit_ssl_backbone(cfg, input_shape):
    """MaskDINO path: raw ``res2``-``res5`` without an FPN."""
    return ViTSSLBackbone(cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_vit_ssl_fpn_backbone(cfg, input_shape):
    """Mask R-CNN path: ViT SSL encoder wrapped in a Detectron2 FPN (P2-P6)."""
    bottom_up = ViTSSLBackbone(cfg, input_shape)
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


def load_vit_ssl_pretrained_backbone(model, cfg, settings=None) -> dict:
    """Initialise a :class:`ViTSSLBackbone` from a DINOv2 / ViT-MAE checkpoint."""
    settings = settings or resolve_ssl_settings(cfg)
    if not settings.is_ssl:
        return {}
    if settings.method == "imagenet":
        SSL_LOG.info("SSL_METHOD='imagenet' for a ViT backbone: nothing to load here.")
        return {}
    if settings.method not in ("dinov2", "vitmae", "dentmag_vit"):
        raise ValueError(
            f"load_vit_ssl_pretrained_backbone cannot handle SSL_METHOD={settings.method!r}."
        )

    checkpoint_path = Path(settings.weights).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "MODEL.BACKBONE.SSL_WEIGHTS is set to a ViT checkpoint that does not exist: "
            f"{checkpoint_path}"
        )

    _, _, bottom_up = find_detector_backbone(model)
    if not isinstance(bottom_up, ViTSSLBackbone):
        raise TypeError(
            "MODEL.BACKBONE.NAME must build a ViTSSLBackbone to load ViT weights; "
            f"got {type(bottom_up).__name__}."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    key = settings.checkpoint_key
    if not key:
        key = {
            "dinov2": "teacher",
            "vitmae": "model",
            "dentmag_vit": "encoder_teacher",
        }.get(settings.method, "")
    state, selected_key = extract_state_dict(checkpoint, key)
    # Order matters: the train-only tensors (DINOv2's ``dino_head.*``, ViT-MAE's
    # ``decoder.*``) must be removed *before* the common-prefix strip, otherwise
    # ``backbone.`` is not shared by every key and would survive, leaving the
    # encoder keys prefixed and unmatchable.
    state, dropped = drop_train_only_keys(state)
    state, stripped = strip_common_prefixes(state)

    encoder = bottom_up.encoder
    state, pos_note = _adapt_pos_embed(state, encoder)

    SSL_LOG.info(
        "ViT SSL init: method=%s file=%s sha1=%s key=%r dropped_train_only=%d "
        "stripped=%s %s arch=%s patch=%d dim=%d depth=%d out_blocks=%s "
        "pyramid_channels=%d",
        settings.method,
        checkpoint_path,
        checkpoint_digest(checkpoint_path),
        selected_key,
        len(dropped),
        stripped or "-",
        pos_note,
        bottom_up.arch["arch"],
        bottom_up.arch["patch_size"],
        bottom_up.arch["embed_dim"],
        bottom_up.arch["depth"],
        bottom_up.arch["out_blocks"],
        bottom_up.arch["pyramid_channels"],
    )

    # Load only the ViT encoder; the adapter keeps its fresh initialisation
    # (the same for every method, which is what makes the comparison fair).
    return load_state_flexible(
        encoder,
        state,
        strict=settings.strict,
        required_roots=("patch_embed.", "blocks."),
        name=f"ViTSSLBackbone/{settings.method}",
        log_fn=SSL_LOG.info,
    )


def _infer_vit_pos_grid(total_tokens: int):
    """Split a ``pos_embed`` token count into ``(prefix_tokens, grid)``.

    The smallest prefix that leaves a perfect square wins, so a patch-only table
    is not mistaken for a CLS-prefixed one. Returns ``(None, None)`` when no
    square grid exists.
    """
    for prefix in range(0, 9):
        tokens = total_tokens - prefix
        if tokens <= 0:
            break
        grid = math.isqrt(tokens)
        if grid > 1 and grid * grid == tokens:
            return prefix, grid
    return None, None


def _adapt_pos_embed(state: dict, encoder: DINOv2ViT) -> tuple[dict, str]:
    """Align the checkpoint's ``pos_embed`` with the built encoder's grid.

    Bicubic resampling is applied only to a *square* patch grid and only when the
    token count actually differs (doc §5); a non-square or otherwise unexpected
    table is passed through so ``load_state_flexible`` reports it as a shape
    mismatch instead of silently accepting a corrupted positional table.
    """
    if "pos_embed" not in state:
        return state, "no pos_embed in checkpoint"

    source = state["pos_embed"]
    target = encoder.pos_embed
    source = source if source.dim() == 3 else source.unsqueeze(0)
    if tuple(source.shape) == tuple(target.shape):
        return state, "pos_embed shape matches"

    prefix, grid = _infer_vit_pos_grid(int(source.shape[1]))
    if prefix is None:
        return state, (
            f"pos_embed has {source.shape[1]} tokens with no square patch grid; "
            "left as-is and will be reported as a shape mismatch"
        )
    if prefix < encoder.num_prefix_tokens:
        return state, (
            f"pos_embed carries {prefix} prefix token(s) but the encoder expects "
            f"{encoder.num_prefix_tokens}; left as-is for a shape mismatch report"
        )

    head = source[:, : encoder.num_prefix_tokens]
    patch = resample_vit_pos_embed(
        source[:, prefix:],
        (encoder.base_grid, encoder.base_grid),
        num_prefix_tokens=0,
    )
    state = dict(state)
    state["pos_embed"] = torch.cat([head, patch], dim=1)
    return state, (
        f"pos_embed resampled {grid}x{grid}->{encoder.base_grid}x{encoder.base_grid}"
    )
