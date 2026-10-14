# Copyright (c) IDEA, Inc. and its affiliates.
"""Family-agnostic helpers shared by every SSL backbone adapter.

This module is the single place that knows how to

1. safely load a *trusted local* training checkpoint (``torch.load``);
2. locate the encoder state dict inside that checkpoint (``encoder_teacher``,
   ``encoder``, ``teacher``, ``model``, ``resnet`` ...);
3. strip the wrapper prefixes that the pretraining code added
   (``module.`` / ``model.`` / ``backbone.`` / ``net.``);
4. drop optimiser / decoder / projection-head tensors that never belong to a
   downstream image encoder;
5. compare the checkpoint tensors against the newly built module *before*
   loading, and **fail loudly** when nothing usable was found or when the
   architecture mismatch ratio exceeds a threshold (``strict=False`` must never
   silently swallow a broken checkpoint);
6. optionally bicubic-resample a ViT ``pos_embed`` when the downstream input
   grid differs from the pretraining grid.

The loaders intentionally do **not** import anything from the ``dentmag``
project at runtime; the reference implementation there only informed the
contract implemented here.

See ``Docs/ssl_backbone_adapter_implementation_plan.md`` §4-§5 for the design
rationale and the unified config schema (``MODEL.BACKBONE.SSL_*``).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_STRIDES",
    "SSL_METHODS",
    "SSLSettings",
    "resolve_ssl_settings",
    "load_checkpoint",
    "extract_state_dict",
    "strip_common_prefixes",
    "drop_train_only_keys",
    "unwrap_subtree",
    "load_state_flexible",
    "resample_vit_pos_embed",
    "register_safe_globals",
    "checkpoint_digest",
    "find_detector_backbone",
    "load_ssl_pretrained_backbone",
    "SSL_BACKBONE_BUILDERS",
    "align_fpn_size_divisibility",
    "SSL_LOG",
    "TRAIN_ONLY_PATTERNS",
    "CANDIDATE_STATE_KEYS",
]

SSL_LOG = logging.getLogger("detectron2.maskdino.ssl")

#: The four feature levels every SSL backbone must expose, in Detectron2 naming.
FEATURE_NAMES = ("res2", "res3", "res4", "res5")

#: Stride of every feature level, keyed by Detectron2 feature name.
FEATURE_STRIDES = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

#: Supported values of ``MODEL.BACKBONE.SSL_METHOD``.
SSL_METHODS = (
    "none",
    "imagenet",
    "swinmae",
    "dentmag_swin",
    "simclr",
    "dinov2",
    "vitmae",
    "dentmag_vit",
)

#: Canonical dotted key of the encoder state dict, in priority order. The first
#: mapping whose values are tensors wins when the config leaves the key blank.
CANDIDATE_STATE_KEYS = (
    "encoder_teacher",
    "encoder",
    "teacher",
    "model",
    "state_dict",
    "resnet",
    "net",
)

#: Wrapper prefixes that pretraining code prepends; stripped when *all* keys share one.
COMMON_PREFIXES = ("module.", "model.", "backbone.", "ema.", "net.", "_orig_mod.")

#: Name fragments identifying train-only tensors (decoders, heads, optimiser, EMA).
#:
#: Verified against the actual checkpoints under ``.checkpoints/pretrain``:
#:   * Swin-MAE / DentMaG-Swin keep ``layers_up.*``, ``norm_up.*``,
#:     ``decoder_pred.*``, ``first_patch_expanding.*`` and ``mask_token``.
#:   * DINOv2 keeps ``dino_head.*`` (and ``last_layer.*`` in some releases).
#:   * The SimCLR ResNet checkpoint keeps DentMaG's ``heads.*`` task heads.
TRAIN_ONLY_PATTERNS = (
    "layers_up.",
    "norm_up.",
    "decoder_pred.",
    "decoder.",
    "_decoder.",
    "first_patch_expanding.",
    "mask_token",
    "reconstruction",
    "projection_head.",
    "contrastive_head.",
    "projection_mlp.",
    "logit_scale",
    "optimizer",
    "ema_",
    "teacher_head",
    "student_head",
    "dino_head.",
    "last_layer.",
    # DentMaG downstream task heads (fdi / sextant / tooth_class / view).
    "heads.",
    "criterion.",
    "loss.",
    # SimCLRv2's 1000-class classifier; note the trailing dot so MLP ``fc1``/``fc2``
    # of transformer encoders are untouched.
    "fc.",
)

_LEGACY_FORMAT_TO_METHOD = {
    "swin_mae": "swinmae",
    "swinmae": "swinmae",
    "swin": "swinmae",
    "simclr": "simclr",
    "dinov2": "dinov2",
}

#: Fail the load when more than this fraction of tensors have a wrong shape.
DEFAULT_MAX_MISMATCH_RATIO = 0.25

#: ``MODEL.BACKBONE.NAME`` values served by the unified SSL loader. Anything
#: else (e.g. the native ``D2SwinMAEBackbone``) keeps its own legacy path.
SSL_BACKBONE_BUILDERS = (
    "build_swin_ssl_backbone",
    "build_swin_ssl_fpn_backbone",
    "build_resnet_ssl_backbone",
    "build_resnet_ssl_fpn_backbone",
    "build_vit_ssl_backbone",
    "build_vit_ssl_fpn_backbone",
)


# --------------------------------------------------------------------------- #
# Safe checkpoint loading
# --------------------------------------------------------------------------- #
def register_safe_globals() -> None:
    """Allow the plain python objects our own training scripts store in checkpoints.

    The pretraining code saves ``argparse.Namespace`` (hyper-parameters) next to
    the tensors, which ``weights_only=True`` rejects. Only our own trusted local
    files are loaded, so opting these in is safe.
    """
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return
    try:
        add_safe_globals([argparse.Namespace])
    except Exception:  # pragma: no cover - depends on the torch build
        pass


def load_checkpoint(path: str | Path):
    """Load a trusted local checkpoint, preferring the safe (weights-only) path."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SSL checkpoint does not exist: {checkpoint_path}")

    register_safe_globals()
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # Checkpoints that embed an argparse.Namespace / numpy scalars need the
        # full unpickler. These are trusted, locally produced files.
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unable to load SSL checkpoint: {checkpoint_path}") from exc


def checkpoint_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Short SHA-1 digest used for the auditable load log."""
    digest = hashlib.sha1()
    with open(Path(path).expanduser(), "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# State-dict extraction / normalisation
# --------------------------------------------------------------------------- #
def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, Mapping) or not len(obj):
        return False
    tensors = 0
    for value in obj.values():
        if torch.is_tensor(value):
            tensors += 1
            if tensors >= 3:
                return True
    return tensors >= max(1, len(obj) // 2)


def _descend(checkpoint, dotted_key: str):
    """Resolve ``dotted_key`` against either nesting style found in practice.

    State dicts are stored in two shapes:

    * nested mappings -- ``{"model": {"backbone": {"net": {...}}}}``
    * one flat mapping whose keys are already dotted paths --
      ``{"model": {"backbone.net.0.0.weight": tensor(...)}}``

    A plain ``node[part]`` walk only understands the first shape, so a perfectly
    unambiguous path such as ``model.backbone.net`` was rejected for the second
    one (the flat mapping has no literal ``"backbone"`` entry). The loop below
    consumes as many segments as the mapping actually nests, then treats what is
    left as a prefix filter when the mapping is flat.
    """
    parts = dotted_key.split(".")
    node = checkpoint
    for index, part in enumerate(parts):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
            continue
        remainder = ".".join(parts[index:])
        if isinstance(node, Mapping) and any(
            key == remainder or key.startswith(f"{remainder}.") for key in node
        ):
            return {
                key: value
                for key, value in node.items()
                if key == remainder or key.startswith(f"{remainder}.")
            }
        raise KeyError(
            f"SSL checkpoint does not contain the configured key {dotted_key!r} "
            f"(failed at segment {part!r})."
        )
    return node


def extract_state_dict(
    checkpoint,
    dotted_key: str = "",
    candidates: Sequence[str] = CANDIDATE_STATE_KEYS,
) -> tuple[Mapping, str]:
    """Return ``(state_dict, selected_key)`` for the encoder tensors.

    When ``dotted_key`` is provided it is used verbatim. Otherwise the
    candidate keys are probed in order, then any top-level mapping that looks
    like a state dict (largest first). Raises when nothing usable is found -- a
    silently empty state dict is exactly the failure mode this loader exists to
    prevent.
    """
    if dotted_key:
        node = _descend(checkpoint, dotted_key)
        if not isinstance(node, Mapping):
            raise TypeError(
                f"SSL checkpoint key {dotted_key!r} does not resolve to a state dict "
                f"(got {type(node).__name__})."
            )
        return node, dotted_key

    if _looks_like_state_dict(checkpoint):
        return checkpoint, ""

    if isinstance(checkpoint, Mapping):
        for candidate in candidates:
            node = checkpoint.get(candidate)
            if _looks_like_state_dict(node):
                return node, candidate

        # Fall back to the largest state-dict-looking entry.
        plausible = [
            (len(node), key)
            for key, node in checkpoint.items()
            if _looks_like_state_dict(node)
        ]
        if plausible:
            plausible.sort(reverse=True)
            _, key = plausible[0]
            return checkpoint[key], key

    available = sorted(checkpoint.keys()) if isinstance(checkpoint, Mapping) else type(checkpoint).__name__
    raise KeyError(
        "Unable to locate an encoder state dict in the SSL checkpoint. "
        f"Configure MODEL.BACKBONE.SSL_CHECKPOINT_KEY explicitly. Top-level entries: {available}"
    )


def strip_common_prefixes(
    state_dict: Mapping,
    prefixes: Sequence[str] = COMMON_PREFIXES,
) -> tuple[dict, list[str]]:
    """Strip a leading wrapper prefix when *every* key shares it.

    Applied repeatedly so ``backbone.net.0.0.weight`` collapses to
    ``0.0.weight`` without needing a hand-written rule per checkpoint.
    """
    current = dict(state_dict)
    stripped_log: list[str] = []
    while True:
        for prefix in prefixes:
            if not prefix.endswith("."):
                continue
            if current and all(key.startswith(prefix) for key in current):
                cut = len(prefix)
                current = {key[cut:]: value for key, value in current.items()}
                stripped_log.append(prefix)
                break
        else:
            break
    return current, stripped_log


def drop_train_only_keys(
    state_dict: Mapping,
    patterns: Sequence[str] = TRAIN_ONLY_PATTERNS,
) -> tuple[dict, list[str]]:
    """Remove decoder / head / optimiser tensors that must never reach the encoder."""
    if not patterns:
        return dict(state_dict), []
    kept: dict = {}
    dropped: list[str] = []
    for key, value in state_dict.items():
        if any(pattern in key for pattern in patterns):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped


def unwrap_subtree(
    state_dict: Mapping,
    root: str,
    *,
    min_ratio: float = 0.5,
) -> tuple[dict, list[str]]:
    """Keep only the keys under ``root`` and strip it, when enough keys live there.

    Used to reach a nested encoder (e.g. the ``net.`` sequential of SimCLRv2)
    without hand-writing a rule per checkpoint. Returns ``(state_dict, dropped)``
    where ``dropped`` lists the keys that were outside ``root``.
    """
    if not state_dict or not root.endswith("."):
        return dict(state_dict), []
    under = [key for key in state_dict if key.startswith(root)]
    threshold = max(1, int(min_ratio * len(state_dict)))
    if len(under) < threshold:
        return dict(state_dict), []
    kept = {key[len(root) :]: value for key, value in state_dict.items() if key.startswith(root)}
    dropped = [key for key in state_dict if not key.startswith(root)]
    return kept, dropped


# --------------------------------------------------------------------------- #
# Fail-loud loading
# --------------------------------------------------------------------------- #
def load_state_flexible(
    module: nn.Module,
    state_dict: Mapping,
    *,
    strict: bool = True,
    required_roots: Sequence[str] = (),
    max_shape_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
    name: str = "encoder",
    log_fn=SSL_LOG.info,
) -> dict:
    """Load ``state_dict`` into ``module`` and report exactly what happened.

    Fails loudly when no tensor could be loaded or when too many tensors have a
    mismatching shape. Always returns the audit summary so callers can forward
    it to their own logger.

    Parameters
    ----------
    strict:
        When true, *any* shape mismatch and *any* missing ``required_roots``
        entry aborts the load. Non-strict tolerates up to
        ``max_shape_mismatch_ratio`` mismatching tensors but still requires the
        configured roots.
    required_roots:
        Key prefixes that must receive at least one tensor (e.g.
        ``("patch_embed.", "layers.")``); guards against loading only a bias.
    """
    target = module.state_dict()
    compatible: dict = {}
    shape_mismatches: list[tuple[str, tuple, tuple]] = []
    unexpected: list[str] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            unexpected.append(key)
            continue
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        compatible[key] = value

    if not compatible:
        raise RuntimeError(
            f"[{name}] loaded 0 tensors; the checkpoint does not match the built module. "
            f"First mismatches: "
            + ", ".join(
                f"{key}: ckpt{src} != model{dst}"
                for key, src, dst in shape_mismatches[:5]
            )
        )

    total_target = max(1, len(target))
    mismatch_ratio = len(shape_mismatches) / total_target
    limit = 0.0 if strict else max_shape_mismatch_ratio
    if shape_mismatches and mismatch_ratio > limit:
        details = ", ".join(
            f"{key}: ckpt{src} != model{dst}"
            for key, src, dst in shape_mismatches[:8]
        )
        raise ValueError(
            f"[{name}] {len(shape_mismatches)} shape mismatches "
            f"({mismatch_ratio:.1%} of {total_target} tensors) exceed the "
            f"allowed ratio {limit:.0%}; the architecture does not match the checkpoint. "
            f"Examples: {details}"
        )

    missing_roots = [
        root
        for root in required_roots
        if not any(key.startswith(root) for key in compatible)
    ]
    if missing_roots:
        raise ValueError(
            f"[{name}] checkpoint provided none of the required encoder sections: "
            + ", ".join(missing_roots)
        )

    incompatible = module.load_state_dict(compatible, strict=False)
    still_missing = [
        key for key in incompatible.missing_keys if key not in compatible
    ]

    summary = {
        "loaded": len(compatible),
        "target": total_target,
        "load_ratio": len(compatible) / total_target,
        "shape_mismatches": len(shape_mismatches),
        "unexpected": len(unexpected),
        "missing_after_load": len(still_missing),
        "randomly_initialised": sorted(still_missing),
    }
    log_fn(
        "[%s] loaded %d/%d tensors (%.1f%%); skipped %d mismatched shapes, "
        "%d unexpected tensors; %d tensors keep their random init.",
        name,
        summary["loaded"],
        summary["target"],
        100.0 * summary["load_ratio"],
        summary["shape_mismatches"],
        summary["unexpected"],
        summary["missing_after_load"],
    )
    if summary["randomly_initialised"]:
        preview = ", ".join(summary["randomly_initialised"][:8])
        more = "" if len(summary["randomly_initialised"]) <= 8 else " ..."
        log_fn("[%s] random init kept for: %s%s", name, preview, more)
    return summary


# --------------------------------------------------------------------------- #
# ViT positional-embedding resampling
# --------------------------------------------------------------------------- #
def resample_vit_pos_embed(
    pos_embed: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    num_prefix_tokens: int = 1,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Bicubic-resample a square ViT ``pos_embed`` to a different patch grid.

    ``pos_embed`` has shape ``[1, P + g*g, C]`` where ``P`` is the number of
    prefix tokens (the CLS token, plus optional register tokens) and ``g`` the
    pretraining grid size. Returns ``[1, P + h*w, C]`` for ``target_hw=(h, w)``.
    A no-op when the grid already matches.
    """
    tensor = pos_embed if pos_embed.dim() == 3 else pos_embed.unsqueeze(0)
    if tensor.shape[0] != 1:
        raise ValueError(
            f"Expected a single-row pos_embed, got shape {tuple(tensor.shape)}."
        )
    total, dim = tensor.shape[1], tensor.shape[2]
    patch_tokens = total - num_prefix_tokens
    grid = int(round(math.sqrt(patch_tokens)))
    if grid * grid != patch_tokens:
        raise ValueError(
            f"pos_embed carries {patch_tokens} patch tokens (not a perfect square); "
            "cannot infer the pretraining grid for resampling."
        )
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if grid == target_h and grid == target_w:
        return tensor

    prefix = tensor[:, :num_prefix_tokens]
    patch = tensor[:, num_prefix_tokens:]
    patch = patch.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
    patch = F.interpolate(
        patch, size=(target_h, target_w), mode=mode, align_corners=False, antialias=True
    )
    patch = patch.permute(0, 2, 3, 1).reshape(1, target_h * target_w, dim)
    return torch.cat([prefix, patch], dim=1)


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
@dataclass
class SSLSettings:
    """Normalised, family-agnostic view of the SSL-related config fields."""

    method: str = "none"
    enabled: bool = False
    weights: str = ""
    checkpoint_key: str = ""
    strict: bool = True
    source: str = "default"
    extra: dict = field(default_factory=dict)

    @property
    def is_ssl(self) -> bool:
        return self.enabled and self.method not in ("", "none")


def _cfg_get(cfg, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if node is None or not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node if node is not None else default


def _infer_method_from_name(cfg) -> str:
    name = str(_cfg_get(cfg, "MODEL.BACKBONE.NAME", "") or "").lower()
    if "vit" in name:
        return "dinov2"
    if "resnet" in name or "simclr" in name:
        return "simclr"
    if "swin" in name:
        return "swinmae"
    return "none"


def resolve_ssl_settings(cfg) -> SSLSettings:
    """Merge the new ``SSL_*`` fields with the legacy ``PRETRAINED_*`` fields.

    The new schema wins whenever it is populated; otherwise the legacy
    Swin-MAE-style fields (``USE_PRETRAINED`` / ``PRETRAINED_FORMAT`` /
    ``PRETRAINED_WEIGHTS`` / ``PRETRAINED_CHECKPOINT_KEY``) are translated so
    every existing config keeps working unchanged.
    """
    method = str(_cfg_get(cfg, "MODEL.BACKBONE.SSL_METHOD", "") or "").strip().lower()
    enabled_new = bool(_cfg_get(cfg, "MODEL.BACKBONE.USE_SSL_PRETRAINED", False))
    weights_new = str(_cfg_get(cfg, "MODEL.BACKBONE.SSL_WEIGHTS", "") or "")
    key_new = str(_cfg_get(cfg, "MODEL.BACKBONE.SSL_CHECKPOINT_KEY", "") or "")
    strict = bool(_cfg_get(cfg, "MODEL.BACKBONE.SSL_STRICT_LOAD", True))

    enabled_old = bool(_cfg_get(cfg, "MODEL.BACKBONE.USE_PRETRAINED", False))
    legacy_format = str(_cfg_get(cfg, "MODEL.BACKBONE.PRETRAINED_FORMAT", "") or "").lower()
    weights_old = str(_cfg_get(cfg, "MODEL.BACKBONE.PRETRAINED_WEIGHTS", "") or "")
    key_old = str(_cfg_get(cfg, "MODEL.BACKBONE.PRETRAINED_CHECKPOINT_KEY", "") or "")

    source = "new"
    if not method or method == "none":
        if enabled_old and legacy_format:
            method = _LEGACY_FORMAT_TO_METHOD.get(legacy_format, legacy_format)
            source = "legacy"
        elif enabled_new:
            method = _infer_method_from_name(cfg)
            source = "inferred"
        else:
            method = "none"
            source = "default"

    enabled = enabled_new or enabled_old
    weights = weights_new or weights_old
    checkpoint_key = key_new or key_old

    if method == "imagenet" and not weights:
        # ImageNet init pulls its own weights; an explicit path is optional.
        enabled = True

    return SSLSettings(
        method=method,
        enabled=enabled,
        weights=weights,
        checkpoint_key=checkpoint_key,
        strict=strict,
        source=source,
    )


# --------------------------------------------------------------------------- #
# Detector / module lookup + top-level dispatcher
# --------------------------------------------------------------------------- #
def find_detector_backbone(model):
    """Return ``(inner_model, backbone, bottom_up)`` unwrapping DDP and FPN."""
    inner = model.module if hasattr(model, "module") else model
    backbone = getattr(inner, "backbone", None)
    if backbone is None:
        raise TypeError(
            "The model does not expose a `.backbone`; SSL encoder initialisation "
            "expects a Detectron2-style detector."
        )
    bottom_up = getattr(backbone, "bottom_up", backbone)
    return inner, backbone, bottom_up


def load_ssl_pretrained_backbone(model, cfg):
    """Initialise only the image encoder from the configured SSL checkpoint.

    Returns the per-encoder audit summary produced by ``load_state_flexible``
    (a dict) when a checkpoint was consumed, and ``False`` when SSL
    initialisation is not configured for this backbone. Kept deliberately
    separate from ``MODEL.WEIGHTS``: the latter restores a *complete* detector
    while this path only maps the pretrained SSL encoder into
    ``model.backbone``.
    """
    settings = resolve_ssl_settings(cfg)
    if not settings.is_ssl:
        return False

    name = str(_cfg_get(cfg, "MODEL.BACKBONE.NAME", "") or "")
    if name not in SSL_BACKBONE_BUILDERS:
        if settings.source == "new":
            raise ValueError(
                f"MODEL.BACKBONE.SSL_METHOD={settings.method!r} requires one of "
                f"{', '.join(SSL_BACKBONE_BUILDERS)} as MODEL.BACKBONE.NAME, "
                f"got {name!r}."
            )
        # Legacy Swin-MAE configs set USE_PRETRAINED/PRETRAINED_* on the native
        # D2SwinMAEBackbone; leave those to load_swin_mae_pretrained_backbone.
        SSL_LOG.info(
            "MODEL.BACKBONE.NAME=%r is not one of the SSL backbones; skipping the "
            "unified SSL loader for method=%r.",
            name,
            settings.method,
        )
        return False

    method = settings.method
    audit = None
    if method in ("swinmae", "dentmag_swin"):
        from .swin_ssl_backbone import load_swin_ssl_pretrained_backbone

        audit = load_swin_ssl_pretrained_backbone(model, cfg, settings)
    elif method == "simclr":
        from .resnet_ssl_backbone import load_resnet_ssl_pretrained_backbone

        audit = load_resnet_ssl_pretrained_backbone(model, cfg, settings)
    elif method in ("dinov2", "vitmae", "dentmag_vit", "imagenet_vit"):
        from .vit_ssl_backbone import load_vit_ssl_pretrained_backbone

        audit = load_vit_ssl_pretrained_backbone(model, cfg, settings)
    elif method == "imagenet":
        # ImageNet init is handled by the family-specific builder (weights are
        # resolved from the canonical name), so there is nothing to load here.
        SSL_LOG.info(
            "SSL_METHOD='imagenet': the encoder keeps its ImageNet initialisation; "
            "no checkpoint is loaded by the SSL path."
        )
    else:
        raise ValueError(
            f"Unsupported MODEL.BACKBONE.SSL_METHOD={method!r}. "
            f"Choose one of {', '.join(SSL_METHODS)}."
        )
    return audit if audit is not None else True


# --------------------------------------------------------------------------- #
# Shared backbone scaffolding
# --------------------------------------------------------------------------- #
class FeaturePyramidBackbone(Backbone):
    """Base class for SSL backbones that emit ``res2``-``res5`` as NCHW tensors.

    Subclasses fill :attr:`stage_channels` / :attr:`native_strides` and implement
    ``forward``; the feature contract (names, channels, strides) is declared once
    here so the smoke tests and ``output_shape()`` cannot drift apart.
    """

    out_features: Sequence[str] = FEATURE_NAMES

    def __init__(self, out_features: Sequence[str] = FEATURE_NAMES):
        super().__init__()
        self._out_features = [
            name for name in out_features if name in FEATURE_NAMES
        ] or list(FEATURE_NAMES)
        #: Populated by subclasses with the true channel count of each stage.
        self.stage_channels: dict = {}
        #: Populated by subclasses when a stage's stride differs from the default.
        self._native_strides: dict = {}

    @property
    def native_strides(self) -> dict:
        return {
            name: self._native_strides.get(name, FEATURE_STRIDES[name])
            for name in self._out_features
        }

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.stage_channels[name], stride=self.native_strides[name])
            for name in self._out_features
        }

    def num_parameters(self, trainable: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable)
        )


def align_fpn_size_divisibility(fpn, bottom_up) -> int:
    """Make ``fpn`` advertise the padding ``bottom_up`` needs; return the value.

    ``detectron2.modeling.backbone.fpn.FPN.__init__`` hard-codes
    ``_size_divisibility = strides[-1]`` (32) regardless of what the bottom-up
    backbone requires, and ``GeneralizedRCNN`` reads that attribute to pad the
    input image. A windowed / patched encoder therefore receives an image it
    cannot partition: the Swin SSL backbone needs
    ``patch_size * window_size * 2 ** (num_layers - 1)`` (224) and the ViT SSL
    backbone needs ``lcm(patch_size, 32)`` (224 at ``patch_size=14``). Widening
    the padding is always safe -- it only adds zeros the encoders ignore -- so
    the floor stays at the FPN's own stride when the bottom-up needs less.
    """
    required = int(getattr(bottom_up, "size_divisibility", 1) or 1)
    current = int(getattr(fpn, "size_divisibility", 1) or 1)
    if required > current:
        # No public setter exists; ``_size_divisibility`` is the documented
        # source of the read-only ``size_divisibility`` property.
        fpn._size_divisibility = required
        SSL_LOG.info(
            "fpn.size_divisibility raised from %d to %d to match %s.",
            current,
            required,
            type(bottom_up).__name__,
        )
    return max(required, current)


def validate_feature_contract(
    outputs: Mapping,
    expected: Sequence[str] = FEATURE_NAMES,
    *,
    name: str = "backbone",
) -> None:
    """Assert ``outputs`` contains exactly ``expected`` NCHW feature maps."""
    missing = [key for key in expected if key not in outputs]
    if missing:
        raise ValueError(f"[{name}] forward() did not produce {missing}; got {sorted(outputs)}.")
    for key in expected:
        value = outputs[key]
        if not torch.is_tensor(value) or value.dim() != 4:
            raise ValueError(
                f"[{name}] feature {key} must be a 4D NCHW tensor, got "
                f"{type(value).__name__}{tuple(getattr(value, 'shape', ()))}."
            )
