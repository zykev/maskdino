# Copyright (c) IDEA, Inc. and its affiliates.
from .backbone import (  # noqa: F401  (registers the SSL backbones)
    FEATURE_NAMES,
    FEATURE_STRIDES,
    ResNetSSLBackbone,
    SwinSSLBackbone,
    ViTSSLBackbone,
    load_ssl_pretrained_backbone,
    resolve_ssl_settings,
)
from .backbone.swin import D2SwinTransformer
from .backbone.swin_mae import D2SwinMAEBackbone
from .pixel_decoder.maskdino_encoder import MaskDINOEncoder
from .meta_arch.maskdino_head import MaskDINOHead

