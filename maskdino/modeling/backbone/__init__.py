# Copyright (c) Facebook, Inc. and its affiliates.
"""Backbone zoo.

Importing this package registers every backbone with Detectron2's
``BACKBONE_REGISTRY``. The SSL-adapted backbones live alongside the native ones
and share the same ``res2``-``res5`` output contract:

* :mod:`~maskdino.modeling.backbone.swin_ssl_backbone` -- Swin-MAE / DentMaG-Swin
* :mod:`~maskdino.modeling.backbone.resnet_ssl_backbone` -- SimCLRv2 ResNet
* :mod:`~maskdino.modeling.backbone.vit_ssl_backbone` -- DINOv2 / ViT-MAE / DentMaG-ViT

:mod:`~maskdino.modeling.backbone.ssl_common` holds the shared checkpoint
loading helpers and the ``SSL_*`` config resolution; it contains no
architecture-specific logic.
"""

from . import ssl_common  # noqa: F401  (helpers; registers safe globals)
from . import simclrv2_resnet  # noqa: F401
from . import resnet_ssl_backbone  # noqa: F401
from . import swin_ssl_backbone  # noqa: F401
from . import vit_ssl_backbone  # noqa: F401

from .resnet_ssl_backbone import (  # noqa: F401
    ResNetSSLBackbone,
    build_resnet_ssl_backbone,
    build_resnet_ssl_fpn_backbone,
    load_resnet_ssl_pretrained_backbone,
)
from .ssl_common import (  # noqa: F401
    FEATURE_NAMES,
    FEATURE_STRIDES,
    SSL_METHODS,
    load_ssl_pretrained_backbone,
    resolve_ssl_settings,
)
from .swin_ssl_backbone import (  # noqa: F401
    SwinSSLBackbone,
    build_swin_ssl_backbone,
    build_swin_ssl_fpn_backbone,
    load_swin_ssl_pretrained_backbone,
)
from .vit_ssl_backbone import (  # noqa: F401
    ViTSSLBackbone,
    build_vit_ssl_backbone,
    build_vit_ssl_fpn_backbone,
    load_vit_ssl_pretrained_backbone,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_STRIDES",
    "SSL_METHODS",
    "ResNetSSLBackbone",
    "SwinSSLBackbone",
    "ViTSSLBackbone",
    "build_resnet_ssl_backbone",
    "build_resnet_ssl_fpn_backbone",
    "build_swin_ssl_backbone",
    "build_swin_ssl_fpn_backbone",
    "build_vit_ssl_backbone",
    "build_vit_ssl_fpn_backbone",
    "load_resnet_ssl_pretrained_backbone",
    "load_ssl_pretrained_backbone",
    "load_swin_ssl_pretrained_backbone",
    "load_vit_ssl_pretrained_backbone",
    "resolve_ssl_settings",
]
