# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import json # Added: needed for dataset loading
import random
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set

import warnings
# 忽略掉所有的 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.structures import BoxMode # Added: needed for dataset registration
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskDINO Imports
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)

# ==============================================================================
#  1. 自定义数据集处理函数 (User Provided Logic)
def get_caries_dicts(root_dir, is_train=False, keep_healthy_ratio=0.1):
    """
    img_dir: 包含 single_tooth 文件夹的根目录
    json_path: 转换后的 coco json 文件路径
    """
    # 确保路径存在
    img_dir = os.path.join(root_dir, "single_tooth")
    json_path = os.path.join(root_dir, "caries_sample_dataset.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with open(json_path) as f:
        coco_data = json.load(f)

    # 建立 image_id 到 image 信息的映射
    images = {img["id"]: img for img in coco_data["images"]}
    
    # 建立 image_id 到 annotations 的映射
    img_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    dataset_dicts = []
    
    for img_id, img_info in images.items():

        anns = img_to_anns.get(img_id, [])
        # --- 新增：健康样本降采样逻辑 ---
        if len(anns) == 0:
            # 如果是训练集，且是健康图片，按概率决定是否保留
            if is_train and random.random() > keep_healthy_ratio:
                continue
        # ------------------------------

        record = {}
        # 拼接图片完整路径    
        record["file_name"] = img_info["file_name"]
        record["image_id"] = img_id
        record["height"] = img_info["height"]
        record["width"] = img_info["width"]
      
        objs = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann["segmentation"],
                # 重要：Detectron2 内部类别从 0 开始。
                # 请确保你的 json 中 category_id 是 1-9，这里减 1 变成 0-8
                "category_id": ann["category_id"] - 1, 
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

def register_caries_dataset():
    """
    注册 caries 数据集到 Detectron2 Catalog
    """
    CLASS_NAMES = [
        "caries", "white_spot_lesion", "filling_no_caries", 
        "filling_with_caries", "fissure_sealant", "non_caries_hard_tissue", 
        "staining", "abnormal_central_cusp", "palatal_radicular_groove"
    ]
    
    # 修改为你的实际数据路径
    data_root = ".datasets/intraoral_anno/single_ch_0225"

    for d in ["train", "val"]:
        dataset_name = "tooth_" + d
        # 避免重复注册
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
            
        # --- 新增：区分训练集和验证集的注册 ---
        is_train_set = (d == "train")
        DatasetCatalog.register(
            dataset_name, 
            lambda is_t=is_train_set, i=data_root: get_caries_dicts(os.path.join(i), is_train=is_t, keep_healthy_ratio=0.1)
        )
        # ------------------------------------
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES, evaluator_type="coco")
        
    print(f"Successfully registered datasets: tooth_train, tooth_val with {len(CLASS_NAMES)} classes.")

# ==============================================================================
#  2. Trainer Class (Provided by MaskDINO)

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer/MaskDINO.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            'trainer': weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        # 对于自定义数据集，通常 evaluator_type 为空或者默认，强制使用 COCO Evaluator
        if evaluator_type == "coco" or evaluator_type == "":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            
        # Keep other evaluators logic from MaskDINO
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        if len(evaluator_list) == 0:
            # Fallback to COCO Evaluator for custom datasets if type is not set
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)
            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class MaskDINOPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.input_format = self.cfg.INPUT.FORMAT

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            return self.model(inputs)[0]


def save_confidence_score_distribution(cfg, predictor, dataset_dicts, class_names):
    scores_per_class = {idx: [] for idx in range(len(class_names))}
    gt_counts_per_class = {idx: 0 for idx in range(len(class_names))}

    print("Saving MaskDINO confidence score distribution...")
    for record in dataset_dicts:
        for ann in record.get("annotations", []):
            cat_id = ann["category_id"]
            if 0 <= cat_id < len(class_names):
                gt_counts_per_class[cat_id] += 1

        image = cv2.imread(record["file_name"])
        if image is None:
            continue
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        if not instances.has("scores"):
            continue
        for cls_id, score in zip(instances.pred_classes.tolist(), instances.scores.tolist()):
            if 0 <= cls_id < len(class_names):
                scores_per_class[cls_id].append(score)

    cols = 4
    rows = int(np.ceil(len(class_names) / cols))
    plt.figure(figsize=(cols * 5, rows * 4))
    plt.suptitle("Confidence Score Distribution per Class", fontsize=18)

    for cls_id, class_name in enumerate(class_names):
        plt.subplot(rows, cols, cls_id + 1)
        scores = scores_per_class[cls_id]
        title = f"{class_name}\nGT={gt_counts_per_class[cls_id]} Pred={len(scores)}"
        if scores:
            plt.hist(scores, bins=30, color="skyblue", edgecolor="black", alpha=0.75)
            plt.axvline(np.mean(scores), color="red", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(scores):.2f}")
            plt.legend(fontsize=8)
        else:
            plt.text(0.5, 0.5, "No Predictions", ha="center", va="center")
        plt.title(title)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Confidence")
        plt.ylabel("Count")

    output_dir = os.path.join(cfg.OUTPUT_DIR, "inference", "custom_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "confidence_score_distribution.png"))
    plt.close()


def add_label(image_bgr, text, color):
    labeled = image_bgr.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(labeled, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return labeled


def save_prediction_visualizations(cfg, predictor, dataset_dicts, metadata, class_names, max_samples=5, conf_thresh=0.5):
    vis_save_dir = os.path.join(cfg.OUTPUT_DIR, "inference", "custom_visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)

    category_to_samples = {idx: [] for idx in range(len(class_names))}
    for record in dataset_dicts:
        seen = set()
        for ann in record.get("annotations", []):
            cat_id = ann["category_id"]
            if 0 <= cat_id < len(class_names) and cat_id not in seen:
                category_to_samples[cat_id].append(record)
                seen.add(cat_id)

    for cls_id, class_name in enumerate(class_names):
        print(f"Category {cls_id} ({class_name}): Found {len(category_to_samples[cls_id])} samples")

    target_size = (640, 640)
    for cls_id, class_name in enumerate(class_names):
        samples = category_to_samples[cls_id]
        if not samples:
            continue

        class_dir = os.path.join(vis_save_dir, f"{cls_id:02d}_{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        selected = random.sample(samples, min(len(samples), max_samples))

        row_orig, row_gt, row_pred = [], [], []
        for idx, record in enumerate(selected):
            image_bgr = cv2.imread(record["file_name"])
            if image_bgr is None:
                continue

            image_rgb = image_bgr[:, :, ::-1]
            gt_rgb = Visualizer(image_rgb, metadata=metadata, scale=0.8).draw_dataset_dict(record).get_image()

            outputs = predictor(image_bgr)
            instances = outputs["instances"].to("cpu")
            if instances.has("scores"):
                instances = instances[instances.scores >= conf_thresh]
            pred_rgb = Visualizer(image_rgb, metadata=metadata, scale=0.8).draw_instance_predictions(instances).get_image()

            gt_bgr = gt_rgb[:, :, ::-1]
            pred_bgr = pred_rgb[:, :, ::-1]

            cv2.imwrite(os.path.join(class_dir, f"sample_{idx}_0_orig.jpg"), image_bgr)
            cv2.imwrite(os.path.join(class_dir, f"sample_{idx}_1_gt.jpg"), gt_bgr)
            cv2.imwrite(os.path.join(class_dir, f"sample_{idx}_2_pred.jpg"), pred_bgr)

            orig_resized = add_label(cv2.resize(image_bgr, target_size), f"Sample {idx}", (255, 255, 255))
            gt_resized = add_label(cv2.resize(gt_bgr, target_size), "GT", (0, 255, 0))
            pred_resized = add_label(cv2.resize(pred_bgr, target_size), f"Pred score>={conf_thresh:g}", (0, 0, 255))

            row_orig.append(orig_resized)
            row_gt.append(gt_resized)
            row_pred.append(pred_resized)

        if row_orig:
            summary_matrix = np.vstack([np.hstack(row_orig), np.hstack(row_gt), np.hstack(row_pred)])
            cv2.imwrite(os.path.join(class_dir, f"00_{class_name}_summary_matrix.jpg"), summary_matrix)

    print(f"Saved MaskDINO visualizations to: {vis_save_dir}")


def save_post_train_visualizations(cfg):
    if not comm.is_main_process():
        return

    weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if not os.path.exists(weights_path):
        print(f"Skip visualization: checkpoint not found: {weights_path}")
        return

    vis_cfg = cfg.clone()
    vis_cfg.defrost()
    vis_cfg.MODEL.WEIGHTS = weights_path
    vis_cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    vis_cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    vis_cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    vis_cfg.freeze()

    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes

    predictor = MaskDINOPredictor(vis_cfg)
    save_confidence_score_distribution(vis_cfg, predictor, dataset_dicts, class_names)
    save_prediction_visualizations(vis_cfg, predictor, dataset_dicts, metadata, class_names)

# ==============================================================================
#  3. Setup & Main (Modified for Custom Dataset)
# ==============================================================================

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    # 1. 注册数据集，以便配置可以引用 "tooth_train"
    register_caries_dataset()
    
    # 2. 加载命令行传入的 config (通常是 maskdino 的 yaml)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.OUTPUT_DIR = "output/maskdino_caries_v1"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 3. --- 强制覆盖为 Caries 数据集配置 ---
    cfg.DATASETS.TRAIN = ("tooth_train",)
    cfg.DATASETS.TEST = ("tooth_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # --- 关键新增：处理类别不平衡 ---
    # 允许保留下来的健康样本进入 DataLoader（作为负样本训练）
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False 
    
    # 启用重复因子采样，拉升稀有类别 (如 Caries) 的出场率
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.05 
    # ------------------------------
    
    # 训练超参数 (根据你的需求调整)
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch Size
    cfg.SOLVER.BASE_LR = 0.00025  # Learning Rate
    cfg.SOLVER.MAX_ITER = 3000    # Iterations
    cfg.SOLVER.STEPS = []         # No decay step
    
    # 4. --- 关键：修改类别数量 ---
    # MaskDINO/Mask2Former 架构依赖 SemSegHead.NUM_CLASSES
    # 必须设置为你的实际类别数 (9)
    num_classes = 9
    # 如果 config 中包含 MaskDINO 特有的 key，也一并修改以防万一
    try:
        cfg.MODEL.SemSegHead.NUM_CLASSES = num_classes
        cfg.MODEL.RetinaNet.NUM_CLASSES = num_classes 
        cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    except AttributeError:
        pass
        
    # 为了兼容部分 Evaluator，也设置 ROI_HEADS (尽管 MaskDINO 不直接用它)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    
    print("Command cfg:", cfg)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    train_result = trainer.train()
    comm.synchronize()
    save_post_train_visualizations(cfg)
    return train_result


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    
    # random port for distributed training
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
