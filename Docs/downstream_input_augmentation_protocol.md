# Downstream 输入分辨率与数据增强统一协议

## 1. 目的

本协议用于比较不同 SSL 预训练 encoder 在两个 downstream 框架上的迁移表现：

- Mask R-CNN；
- MaskDINO。

比较对象包括 ImageNet 初始化、COCO 完整模型工程基线、Swin-MAE、SimCLR、DINOv2、ViT-MAE、DentMag-Swin 与 DentMag-ViT。预训练阶段的输入分辨率、crop 策略和 augmentation 已经不同；这些差异属于 checkpoint 的历史和方法定义，不应在 downstream 训练时再次变成变量。

本协议的目标是让 downstream 结果主要反映 encoder 初始化的可迁移性，而不是不同方法享受了不同的训练图像尺度或增强强度。

## 2. 总体规则

### 2.1 同一任务内必须统一

在同一个任务内，下列组合必须使用逐项相同的 downstream 数据处理：

```text
所有 SSL method
× Mask R-CNN / MaskDINO
× 相同 train/val/test split
```

必须统一的项目包括：

- 训练与测试输入分辨率；
- resize 规则、最大边限制与 padding 规则；
- 几何 augmentation；
- 光度与质量 augmentation；
- 负样本保留策略、类别重采样策略；
- 训练迭代数、评估脚本与测试时 resize。

不允许出现“MaskDINO 使用强增强而 Mask R-CNN 使用弱增强”，或“DINOv2 沿用 multi-crop、SimCLR 沿用双视图，而 DentMag 使用另一套 crop”的主实验设置。

### 2.2 caries 与 orth 不要求共用一套尺寸或增强

caries 是细粒度实例分割任务，重点是病灶边界、小区域纹理和 mask 质量；orth 是 bbox-only 检测任务，目标在当前统一 resize 后已经偏大。两种任务应该各自采用适合目标尺度的统一协议。

```text
同一任务内统一：是
跨任务强行统一：否
```

### 2.3 SSL 原生处理不在 downstream 复现

不在 downstream 主实验中复现以下 SSL 专属机制：

- DINOv2 的 global/local multi-crop、solarize、双 global view；
- SimCLR 的双视图 NT-Xent 输入、强 RandomResizedCrop、灰度化；
- DentMag 的 full-image + tooth/sextant/random crop 对齐与 FDI/bbox crop 组织；
- MAE / Swin-MAE 的随机 mask 与 reconstruction 视图。

这些机制用于构建 SSL loss，不是下游 bbox/mask 监督训练的通用 augmentation。直接复用会改变每种方法的 supervised learning objective，并破坏可比性。

## 3. 推荐输入分辨率

### 3.1 caries：实例分割

推荐作为第一轮统一实验设置：

```yaml
INPUT:
  MIN_SIZE_TRAIN: [352, 400, 448]
  MIN_SIZE_TRAIN_SAMPLING: choice
  MAX_SIZE_TRAIN: 672
  MIN_SIZE_TEST: 448
  MAX_SIZE_TEST: 672
  MASK_FORMAT: polygon
```

选择理由：

- 当前 caries 配置使用 `[300, 400, 500]`，测试短边为 `400`；
- 将测试短边提升到 `448` 能提供更高的病灶边界采样密度；
- `448 = 14 × 32`，与 DINOv2 ViT/14 的 token grid 和 32 倍检测尺度天然对齐；
- `448` 也是 DentMag 常用的 full-image 预训练画布之一；
- 最大边 `672` 为统一 padding 后的可控上界，并可容纳训练期较大的 resize。

如果显存不足，唯一允许的降级方案是所有 method 和两个 downstream model 一起退回当前配置；不可针对某一 encoder 或某一检测器单独降低分辨率。

### 3.2 orth：bbox 检测

推荐保持目前已经较合理的统一几何设置：

```yaml
INPUT:
  MIN_SIZE_TRAIN: [640, 800, 1024]
  MIN_SIZE_TRAIN_SAMPLING: choice
  MAX_SIZE_TRAIN: 1333
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1333
```

选择理由：

- 现有统计表明，短边 resize 至 `800` 后，orth bbox sqrt-area 的约为 p10=`176`、p50=`264`、p90=`375` px；
- 这已是适合 FPN anchors、MaskDINO multi-scale feature 与 bbox loss 的目标尺度；
- 直接升至 DentMag high-resolution 的 `896` 不一定增加有效细节，却会显著增加显存和 ViT token 成本；
- 训练短边 `[640, 800, 1024]` 提供足够的尺度扰动，且当前 Mask R-CNN / MaskDINO 都已使用该组设置。

## 4. padding 与 encoder 兼容性

padding 必须满足每个编码器自身的整除约束，只向右侧和底部补零，不改变 resize
后的有效图像内容或 bbox/mask 坐标：

| Backbone | 硬性整除要求 | 实际 `size_divisibility` |
| --- | --- | --- |
| ResNet / SimCLR | `res5` stride 32 | **32** |
| Swin | `patch·window·2^(layers−1)` = 4·7·8 | 224 |
| DINOv2 ViT | `lcm(patch 14, 32)` | 224 |

```text
224 = lcm(DINOv2 patch size 14, detection pyramid stride 32)
```

**最终决定：ResNet / SimCLR 保留原生 `32`，不抬到 `224`。** 理由：`32` 已完全满足
ResNet 的整除需求（无窗口/patch 约束），抬到 `224` 只会让基线多补大量零、
增加显存与算力却不增加任何有效信息。因此本项目的统一点是**每个 encoder 使用其
最小合法 padding**，而非强行统一到同一个数值：ResNet 与 Swin/ViT 之间仍存在
padding 粒度差异，但这是确定性的、由编码器契约决定、且不改变有效内容与坐标；
它不属于“可被某个方法独占利用”的随机增强，故不影响方法间的公平比较。

对 caries 的 `448` 测试输入，`448 / 14 = 32`，可以得到整齐的 DINOv2 token grid；
对 orth，padding 后的长宽也可被 patch/window 安全整除。

padding 粒度会影响小图或长宽比极端图像的显存占用（尤其 `224`）。因此训练日志
必须记录：实际 padding 后 shape、每卡 batch size、峰值显存和有效像素占比。

## 5. 推荐训练 augmentation

两条任务策略都挂在统一的 `AUGMENTATION` 节点下（`AUGMENTATION.CARIES` 与
`AUGMENTATION.ORTH`），由 `datasets/augmentations.py` 的
`add_augmentation_config` 注册。入口按任务（`cfg.ORTH_BBOX_ONLY` 或数据集名
前缀）自动选取对应子节点，因此**同一份 YAML 可以同时被 caries / orth 脚本加载**：
每个 `_base.yaml` 只写自己任务的那个子节点，另一任务回退到模块默认值。

各子节点的键名按 transform 拆分了概率，例如亮度/对比度/饱和度/gamma 各自有
`*_PROB`，因此 orth 的 `COLOR_PROB: 0.2` 现在展开为
`BRIGHTNESS_PROB/CONTRAST_PROB/SATURATION_PROB = 0.2`。

### 5.1 caries augmentation

主协议使用保守、mask/bbox 同步且尽量保护纹理的增强：

| 类型 | 推荐 | 备注 |
|---|---|---|
| 多尺度 resize | 第 3.1 节的训练短边 choice | 所有模型一致 |
| 水平翻转 | `p=0.5`，前提是类别不编码左右牙位 | 先对类别语义做一次审计 |
| 小角度旋转 | 可选，`p≤0.2`、绝对角度 `≤5°` | 必须同步变换 polygon mask 与 bbox |
| 亮度/对比度/gamma | 轻度，`p≤0.2` | 防止破坏龋病细纹理与颜色线索 |
| 裁剪 | 主协议关闭 | 不应随机裁掉病灶实例 |
| 灰度化/强模糊/强 JPEG | 主协议关闭 | 不应把 SimCLR/DINO SSL augmentation 移植到下游 |

水平翻转是否启用取决于 caries 类别是否表达解剖学左右关系；若标签仅表达病灶类型，通常可启用。该结论必须通过样本可视化与标签定义审计确认。

主协议已把 caries 策略写进配置（`configs/{maskrcnn,maskdino}/caries/_base.yaml`
的 `AUGMENTATION.CARIES`，两个检测器逐项一致）：

```yaml
AUGMENTATION:
  CARIES:
    ENABLED: true
    HORIZONTAL_FLIP_PROB: 0.5
    ROTATION_PROB: 0.2          # 绝对角度 ≤ 5°
    ROTATION_ANGLE: 5.0
    BRIGHTNESS_PROB: 0.2
    BRIGHTNESS_MIN: 0.90
    BRIGHTNESS_MAX: 1.10
    CONTRAST_PROB: 0.2
    CONTRAST_MIN: 0.90
    CONTRAST_MAX: 1.10
    SATURATION_PROB: 0.0        # 主协议关闭
    GAMMA_PROB: 0.2
    GAMMA_MIN: 0.90
    GAMMA_MAX: 1.10
    QUALITY_PROB: 0.0           # 模糊 / JPEG 也关闭
```

### 5.2 orth augmentation

为避免当前 MaskDINO 比 Mask R-CNN 接受更强 augmentation，主协议统一采用较保守的 orth 设置：

```yaml
AUGMENTATION:
  ORTH:
    ENABLED: true
    HORIZONTAL_FLIP_PROB: 0.5
    ROTATION_PROB: 0.3
    ROTATION_ANGLE: 7.0
    BRIGHTNESS_PROB: 0.2
    BRIGHTNESS_MIN: 0.85
    BRIGHTNESS_MAX: 1.15
    CONTRAST_PROB: 0.2
    CONTRAST_MIN: 0.90
    CONTRAST_MAX: 1.10
    SATURATION_PROB: 0.2
    SATURATION_MIN: 0.90
    SATURATION_MAX: 1.10
    GAMMA_PROB: 0.2
    GAMMA_MIN: 0.90
    GAMMA_MAX: 1.10
    QUALITY_PROB: 0.2
    BLUR_SIGMA_MIN: 0.10
    BLUR_SIGMA_MAX: 0.80
    JPEG_QUALITY_MIN: 75
    JPEG_QUALITY_MAX: 100
```

该设置来自当前较保守的 Mask R-CNN orth 配置。它应同时用于 MaskDINO，使检测器间的性能差异不来自 MaskDINO 独占的更强颜色、模糊、压缩或更大旋转范围。

若需要研究强 augmentation，应建立一个单独的 task-level ablation：同一组强增强同时应用到所有 SSL method、Mask R-CNN 与 MaskDINO，不与主表混合。

水平翻转仅在 orth 类别不编码患者左右侧时有效。现有配置已经使用它，因此应保留一次标签语义审计记录。

## 6. 可保留的 encoder-specific 兼容处理

以下不是 data augmentation，可因 checkpoint 的原生契约而不同：

- RGB/BGR 输入顺序；
- checkpoint 所需 pixel mean/std normalization；
- ViT position embedding 的插值；
- CLS/register token 的移除；
- DINOv2 patch=14 与 Swin window attention 所需的内部 padding；
- encoder 结构本身的 fixed positional encoding。

这些处理应封装在 SSL backbone adapter 内，并写入配置和日志。它们必须是固定、确定性的；不得成为只对某一方法开启的随机 augmentation。

## 7. 实验记录与验证

每个实验配置至少记录：

```text
task / downstream model / SSL method / checkpoint hash
train/test resize policy / augmentation policy id
size_divisibility / padding 后的实际输入 shape
pixel normalization 方案 / batch size / peak memory
dataset split id / SSL-pretraining-to-test overlap audit result
```

开始完整实验前，对每个 `(task, downstream model, SSL method)` 执行一次 smoke test：

1. 检查 resize 后 bbox/mask 与图像一致；
2. 检查 padding 不改变标注坐标；
3. 检查输入可以通过 Swin、ResNet、ViT adapter；
4. 检查 Mask R-CNN 与 MaskDINO 都能完成一次 loss forward 和 inference；
5. 可视化同一原始样本经不同 adapter 后的有效内容区域，确认差异仅是确定性的 normalization / padding。

## 8. 与预训练设置的关系

[预训练输入分辨率与数据增强对比](pretraining_input_augmentation_comparison.md) 仅记录各 checkpoint 的训练历史。下游主实验不得按表中各方法的 SSL 增强重新分叉数据管线。

本协议与预训练历史的关系为：

```text
预训练：各方法保留原始输入/增强差异，形成各自 checkpoint
下游：统一使用本任务的输入分辨率和标注安全 augmentation
例外：仅保留 checkpoint 必需的确定性 normalization / token/window 兼容处理
```
