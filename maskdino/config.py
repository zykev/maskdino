# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN
from datasets.augmentations import add_augmentation_config
from utils.orth_class_balance import add_class_balance_config


def add_swin_mae_backbone_config(cfg):
    """Add the shared Swin-MAE image-encoder configuration.

    ``MODEL.WEIGHTS`` deliberately remains reserved for a complete detection
    checkpoint.  A Swin-MAE checkpoint initializes only the image encoder and
    is controlled independently through ``MODEL.BACKBONE``.
    """
    cfg.MODEL.BACKBONE.USE_PRETRAINED = False
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ""
    cfg.MODEL.BACKBONE.PRETRAINED_FORMAT = "swin_mae"
    cfg.MODEL.BACKBONE.PRETRAINED_CHECKPOINT_KEY = "model"

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False


def add_ssl_backbone_config(cfg):
    """Add the family-agnostic SSL backbone configuration."""
    cfg.MODEL.BACKBONE.SSL_METHOD = "none"
    cfg.MODEL.BACKBONE.USE_SSL_PRETRAINED = False
    cfg.MODEL.BACKBONE.SSL_WEIGHTS = ""
    cfg.MODEL.BACKBONE.SSL_CHECKPOINT_KEY = ""
    cfg.MODEL.BACKBONE.SSL_STRICT_LOAD = True
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.BACKBONE.FEATURE_NAMES = ["res2", "res3", "res4", "res5"]

    # Swin family. Empty/zero sentinels mean "not configured", so the legacy
    # MODEL.SWIN node keeps working when only SSL_METHOD/PATCH_SIZE is overridden.
    cfg.MODEL.SWIN_SSL = CN()
    cfg.MODEL.SWIN_SSL.EMBED_DIM = 0
    cfg.MODEL.SWIN_SSL.DEPTHS = []
    cfg.MODEL.SWIN_SSL.NUM_HEADS = []
    cfg.MODEL.SWIN_SSL.PATCH_SIZE = 0
    cfg.MODEL.SWIN_SSL.WINDOW_SIZE = 0
    cfg.MODEL.SWIN_SSL.MLP_RATIO = 4.0
    cfg.MODEL.SWIN_SSL.QKV_BIAS = True
    cfg.MODEL.SWIN_SSL.DROP_RATE = 0.0
    cfg.MODEL.SWIN_SSL.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN_SSL.DROP_PATH_RATE = 0.0
    cfg.MODEL.SWIN_SSL.PATCH_NORM = True
    # "" means "derive from SSL_METHOD": swinmae -> "native_fixed" (the learned
    # absolute table of the Swin-MAE encoder), dentmag_swin -> "dentmag_sincos".
    # Explicit values: none | native_fixed | dentmag_sincos.
    cfg.MODEL.SWIN_SSL.POSITION_ENCODING = ""
    # Input resolution the native_fixed table defaults to when no checkpoint
    # supplies one (the table is resampled to the real grid at forward time).
    cfg.MODEL.SWIN_SSL.IMG_SIZE = 224

    # SimCLRv2 ResNet family.
    cfg.MODEL.RESNET_SSL = CN()
    cfg.MODEL.RESNET_SSL.DEPTH = 0  # 0 -> fall back to MODEL.RESNETS.DEPTH
    cfg.MODEL.RESNET_SSL.WIDTH_MULTIPLIER = 1
    cfg.MODEL.RESNET_SSL.STEM_TYPE = "simclrv2"
    cfg.MODEL.RESNET_SSL.SK_RATIO = 0.0625

    # ViT family (DINOv2 / ViT-MAE / DentMaG-ViT).
    cfg.MODEL.VIT_SSL = CN()
    cfg.MODEL.VIT_SSL.ARCH = ""  # vit_tiny|vit_small|vit_base|vit_large|dinov2_vits14|...
    cfg.MODEL.VIT_SSL.PATCH_SIZE = 0
    cfg.MODEL.VIT_SSL.EMBED_DIM = 0
    cfg.MODEL.VIT_SSL.DEPTH = 0
    cfg.MODEL.VIT_SSL.NUM_HEADS = 0
    cfg.MODEL.VIT_SSL.NUM_REGISTER_TOKENS = 0
    cfg.MODEL.VIT_SSL.OUT_BLOCKS = [3, 6, 9, 12]
    cfg.MODEL.VIT_SSL.PYRAMID_CHANNELS = 256
    cfg.MODEL.VIT_SSL.PYRAMID_NORM = "LN"
    cfg.MODEL.VIT_SSL.PYRAMID_SMOOTHING = False
    # Resolution the checkpoint's pos_embed was trained with; only used to size
    # the placeholder parameter before the checkpoint is loaded.
    cfg.MODEL.VIT_SSL.IMG_SIZE = 448


def add_maskdino_config(cfg):
    """
    Add config for MaskDINO.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "MaskDINO_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # MaskDINO model config
    cfg.MODEL.MaskDINO = CN()
    cfg.MODEL.MaskDINO.LEARN_TGT = False

    # loss
    cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
    cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.1
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.DEC_LAYERS = 6
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.BOX_LOSS = True
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.MaskDINO.DN="seg"
    cfg.MODEL.MaskDINO.DN_NOISE_SCALE=0.4
    cfg.MODEL.MaskDINO.DN_NUM=100
    cfg.MODEL.MaskDINO.PRED_CONV=False

    cfg.MODEL.MaskDINO.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    # Compatibility only for upstream YAMLs; MaskDINO uses TOTAL_NUM_FEATURE_LEVELS.
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # MaskDINO inference config
    cfg.MODEL.MaskDINO.TEST = CN()
    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
    # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

    # transformer module
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75

    add_swin_mae_backbone_config(cfg)

    add_ssl_backbone_config(cfg)

    add_augmentation_config(cfg)

    # Orthodontic detection class-balance options, left disabled by default
    # for other tasks. Only ENABLED/BETA/CLIP_MIN/CLIP_MAX/FOCAL_GAMMA/
    # CLASS_WEIGHTS affect MaskDINO training; see add_class_balance_config.
    add_class_balance_config(cfg)
