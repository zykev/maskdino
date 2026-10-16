# 配置目录树

这里每一次可训练的检测 / 分割实验都由恰好一个 YAML 描述。目录按
**检测器 → 任务 → 编码器变体** 组织，因此一次实验的身份可以直接从它的路径读出来：

```
configs/
  maskrcnn/
    caries/{_base, no_ssl, simclr, swinmae, dentmag_swin, dinov2}.yaml
    orth/{_base, no_ssl, simclr, swinmae, dentmag_swin, dinov2}.yaml
  maskdino/
    caries/{_base, no_ssl, simclr, swinmae, dentmag_swin, dinov2}.yaml
    orth/{_base, no_ssl, simclr, swinmae, dentmag_swin, dinov2}.yaml
  _legacy/          # 已退役的扁平配置，保留用于旧的实验记录
  coco/ ade20k/ cityscapes/   # 上游 MaskDINO 配置，未做改动
```

* `_base.yaml` — 只包含任务 + 检测器的超参数。它**不**包含 backbone 选择，因此永远不能单独运行；每个变体都以 `_BASE_: _base.yaml` 开头。
* `<variant>.yaml` — backbone 配置块外加 `SWANLAB.NAME` / `OUTPUT_DIR`。
  其余一切都被继承，这正是让五个变体成为受控对比的原因：相同的数据、相同的增强、相同的求解器、相同的任务头；**只有图像编码器的初始化不同**。

## 五个变体

| 变体 | `SSL_METHOD` | 编码器 | Checkpoint |
| --- | --- | --- | --- |
| `no_ssl` | `none` | 原生 Detectron2 ResNet-50 FPN | ImageNet（MaskDINO）/ 完整 COCO Mask R-CNN（Mask R-CNN） |
| `simclr` | `simclr` | SimCLRv2 ResNet-50，`sk_ratio=0.0625` | `.checkpoints/pretrain/simclr/resnet50_checkpoint-050.pth` |
| `swinmae` | `swinmae` | Swin-Small（Swin-MAE） | `.checkpoints/pretrain/swinmae/swins_checkpoint-50.pth` |
| `dentmag_swin` | `dentmag_swin` | Swin-Small（DentMaG 高分辨率） | `.checkpoints/pretrain/dentmap/swins_highres_checkpoint-50.pth` |
| `dinov2` | `dinov2` | DINOv2 ViT-S/14 | `.checkpoints/pretrain/dinov2/dinov2_vits_train249999_teacher_checkpoint.pth` |

`no_ssl` 是对照组。`simclr` 与 `no_ssl` 拥有*相同的架构*（ResNet-50 C2–C5），
因此这一对隔离出了 SimCLRv2 预训练的影响；`swinmae` 与 `dentmag_swin` 的对比则在
完全相同的 Swin-Small 编码器上隔离出领域特定（DentMaG）预训练的影响。

### 从零训练的 backbone

随机初始化的编码器是一种变体，而不是一个目录：取对应的 SSL 变体并把权重关掉即可。

```yaml
MODEL:
  BACKBONE:
    USE_SSL_PRETRAINED: false
    SSL_WEIGHTS: ""
```

架构（深度、宽度、patch/window 尺寸、位置编码、FFN 比例）不受影响，
因此与预训练运行唯一的差别就是初始化。

### 各检测器使用的构建函数

两个检测器消费同一个适配器，但它们在不同位置需要金字塔，
因此 `MODEL.BACKBONE.NAME` 不同：

| 编码器系列 | Mask R-CNN（`*_fpn_backbone`） | MaskDINO（`*_backbone`） |
| --- | --- | --- |
| ResNet / SimCLR | `build_resnet_ssl_fpn_backbone` | `build_resnet_ssl_backbone` |
| Swin | `build_swin_ssl_fpn_backbone` | `build_swin_ssl_backbone` |
| ViT / DINOv2 | `build_vit_ssl_fpn_backbone` | `build_vit_ssl_backbone` |

Mask R-CNN 把 `res2–res5` 包进 Detectron2 的 `FPN`（+ `LastLevelMaxPool`）；
MaskDINO 则把原始的 `res2–res5` 送入它的 deformable pixel decoder，
并自行构建 `p2–p6`。

## 输入填充：`SIZE_DIVISIBILITY`

基于窗口和 patch 的编码器只能消费能被它们整除划分的图像。Detectron2 的 `FPN`
硬编码了 `_size_divisibility = 32`，因此 SSL 的 FPN 构建函数会把它提升到
bottom-up backbone 所需的任意值（这些辅助函数会在构建时自动完成这一步）：

| Backbone | 所需填充 | MaskDINO `SIZE_DIVISIBILITY` |
| --- | --- | --- |
| ResNet / SimCLR | 32 | `32` |
| Swin（`patch_size · window_size · 2^(layers−1)` = 4·7·8） | 224 | `224` |
| ViT（`lcm(patch_size, 32)` = lcm(14, 32)） | 224 | `224` |

已经正确的 `SIZE_DIVISIBILITY: 32` 值永远不会被调低，
而放宽填充只会增加编码器会忽略的零。

## Checkpoint 键

`SSL_CHECKPOINT_KEY` 是编码器在 checkpoint 中的点分路径。
这些 checkpoint 是*扁平*的点分键映射，唯一的嵌套就是顶层容器，
共享加载器同时处理这两种形态。

| 变体 | `SSL_CHECKPOINT_KEY` |
| --- | --- |
| `simclr` | `model.backbone` |
| `swinmae` | `model` |
| `dentmag_swin` | `encoder_teacher` |
| `dinov2` | `teacher` |

将键留空（`""`）可让加载器从
`encoder_teacher` / `encoder` / `teacher` / `model` / `state_dict` / `resnet` /
`net` 中自动探测。自动探测会记录它选择的键；无法解析的键会被上报，
并带着警告回退到自动探测。

## 运行一个配置

两个入口都会相对仓库根目录解析默认配置，因此它们可以从任意工作目录运行。
默认是 `no_ssl` 对照组 —— 每一次 SSL 运行都需要显式选择启用。

```bash
# Mask R-CNN, orth, Swin-MAE encoder
python maskrcnn_unify.py \
  --task orth \
  --config_file configs/maskrcnn/orth/swinmae.yaml \
  --input_dir .datasets/intraoral_anno/orth_0616 \
  --output_dir output/maskrcnn_orth_swinmae \
  --swanlab_name maskrcnn_orth_swinmae

# MaskDINO, caries, DINOv2 encoder
python maskdino_unify.py \
  --task caries \
  --config_file configs/maskdino/caries/dinov2.yaml \
  --input_dir .datasets/intraoral_anno/single_ch_0225
```

`--output_dir` / `--swanlab_name` 会覆盖 YAML 中的值。现成的启动脚本位于
`scripts/`；覆盖全部 20 种组合的完整矩阵为：

```bash
for detector in maskrcnn maskdino; do
  for task in caries orth; do
    for variant in no_ssl simclr swinmae dentmag_swin dinov2; do
      python "${detector}_unify.py" --task "$task" \
        --config_file "configs/${detector}/${task}/${variant}.yaml"
    done
  done
done
```

## 日志与面板（SwanLab）

每个 `_base.yaml` 都设置了 `LOGGER: swanlab`，于是 Detectron2 `EventStorage`
里的每个标量都会被镜像到 SwanLab，归属 workspace `zykev` / 项目 `dentmag`：

* 训练标量：`total_loss` 与模型 `loss_dict` 的每一项（MaskDINO 的
  `loss_ce` / `loss_bbox` / `loss_giou` / `loss_ce_dn` …，Mask R-CNN 的
  `loss_cls` / `loss_box_reg` / `loss_mask` / `loss_rpn_*` …），外加
  `lr`、`time`、`data_time`、`rank_data_time` 等；
* 验证标量：`val/<loss key>`、`val/total_loss`，以及 COCO 指标
  `bbox/AP*`（龋齿任务还含 `segm/AP*`）。

面板上的**超参数表**来自写入 run config 的完整配置：`SwanLabWriter` 会把
Detectron2 的 `cfg` 转成普通字典传给 `swanlab.init(config=...)`，所以每个实验
在网页上都能直接查到它实际使用的那一份配置，无需再翻 YAML。

本地日志写在**每个实验自己**的 `<OUTPUT_DIR>/swanlab/` 下（与 checkpoint、
`metrics.json` 同级），不同实验之间不会互相污染。run 名取自 `SWANLAB.NAME`，
可用 `--swanlab_name` 覆盖。运行前确保在 `intraoral` 环境里执行过一次
`swanlab login`（key 会缓存在本地），否则 `online` 模式无法上传。

## 运行与校验

每个 `(detector, task)` 组合都有一个启动脚本，脚本会遍历 5 个变体
（`no_ssl` / `simclr` / `swinmae` / `dentmag_swin` / `dinov2`），并从变体 YAML
里读取 `OUTPUT_DIR`，因此不再需要手写 `--output_dir`：

```bash
# 训练：5 个变体依次跑，MaskDINO + orth
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_maskdino_orth.sh

# 只跑部分变体
VARIANTS="no_ssl swinmae" bash scripts/train/train_maskrcnn_caries.sh

# 评估：输出目录自动取自 config，可选评估 split
EVAL_SPLITS="train test" bash scripts/eval/eval_maskdino_orth.sh

# 确认将执行哪些命令（不真正运行）
DRY_RUN=1 bash scripts/train/train_maskrcnn_orth.sh
```

文件名与变体列表通过 `scripts/common.sh` 统一维护（`ALL_VARIANTS`、
`config_output_dir` 等）。完整的运行矩阵与环境变量见
[`scripts/README.md`](../scripts/README.md)。

## 编辑时需谨记的约定

* **任务差异写在 `_base.yaml` 中，绝不写在变体里。** `caries` 是实例分割
  （`MASK_ON: true`）；`orth` 只做框 —— Mask R-CNN 根本不会收到 `gt_masks`，
  而 MaskDINO 训练时使用
  `MASK_WEIGHT: 0.0, DICE_WEIGHT: 0.0, COST_MASK_WEIGHT: 0.0,
  COST_DICE_WEIGHT: 0.0`。
* **`ORTH_CLASS_BALANCE.CLASS_WEIGHTS` 在所有 YAML 中都被有意省略。**
  两个入口都会从训练标注中重新计算它
  （`compute_effective_class_weights`），因为这些权重是逐数据集的类别频率。
* **`AUGMENTATION` 的每个任务策略必须保持同步。** `AUGMENTATION.CARIES`
  与 `AUGMENTATION.ORTH` 各自在
  `configs/maskrcnn/<task>/_base.yaml` 与 `configs/maskdino/<task>/_base.yaml`
  之间逐项一致；拼图脚本 `datasets/augmentations.py`
  （`representative_orth_augmentation_transforms`）是 orth 策略的第三处镜像。
  入口按数据集名（`<task>_train`）自动选择对应策略，YAML 只需写自己任务的那一个节点。
* **Checkpoint 使用绝对路径。** `SSL_WEIGHTS` 指向
  `maskdino/.checkpoints/pretrain/...`；它们不会相对配置文件解析。
* **新增一个变体**意味着每个 `(detector, task)` 文件夹下都要新增一个 YAML，
  并在 `scripts/common.sh` 的 `ALL_VARIANTS` 中加入变体名，同时在变体 YAML 里
  写好 `SWANLAB.NAME` 与 `OUTPUT_DIR`（启动脚本会从后者推导 `--output_dir`）。

关于适配器的设计，请参见 `Docs/ssl_backbone_adapter_implementation_plan.md`。
旧的扁平 `default_*_config.yaml` 及其启动方式已全部移除，请不要再引用。
