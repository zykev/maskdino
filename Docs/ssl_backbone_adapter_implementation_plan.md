# SSL 预训练 Encoder 接入 Mask R-CNN 与 MaskDINO：统一 Backbone Adapter 实施方案

## 1. 目标与边界

本方案将 DentMag 自监督预训练模型及其基线模型接入本仓库的两个下游框架：

- Mask R-CNN：目标检测与实例分割；
- MaskDINO：query-based 目标检测与实例分割。

目标不是将各预训练项目的下游 head 迁入本仓库，而是**仅复用图像 encoder 的预训练参数与原生特征提取路径**。所有检测、分类、定位、实例分割能力仍由现有 Mask R-CNN 或 MaskDINO 学习。因此，不迁移 MAE decoder、DentMag 的 alignment/reconstruction head、DINO projection head、SimCLR projection head，也不迁移任何既有分割 decoder。

最终应支持下列比较：

| Encoder 家族 | 通用默认初始化 | 外部/基线 SSL | DentMag SSL |
|---|---:|---:|---:|
| Swin-T | ImageNet 预训练 Swin-T | Swin-MAE | DentMag-Swin |
| ResNet-50 | Detectron2 ImageNet 预训练 R50 | SimCLR-R50 | — |
| ViT | 同架构 ImageNet 预训练 ViT | DINOv2、ViT-MAE | DentMag-ViT |

每个 encoder 行必须分别用于：

```text
<SSL Backbone> → Mask R-CNN
<SSL Backbone> → MaskDINO
```

龋齿任务使用实例分割标注；正畸任务可使用 bbox-only 配置。两种任务只切换数据集、类别数、损失开关和训练超参，不能更换 adapter 的结构。

## 2. 核心原则

### 2.1 一个 backbone adapter，同时服务两种下游模型

Detectron2 模型通过 `MODEL.BACKBONE.NAME` 构建 backbone。统一 adapter 只需实现 Detectron2 `Backbone` 契约：

```python
forward(images: Tensor) -> dict[str, Tensor]
output_shape() -> dict[str, ShapeSpec]
size_divisibility -> int
```

输出固定为：

```text
res2: [B, C2, H/4,  W/4]
res3: [B, C3, H/8,  W/8]
res4: [B, C4, H/16, W/16]
res5: [B, C5, H/32, W/32]
```

Mask R-CNN 在该输出之上接现有 FPN、RPN、ROI box head 与 mask head；MaskDINO 则把同一 feature dict 输入既有 pixel decoder 与 transformer decoder。adapter 不感知 `task=caries/orth`，也不感知当前下游模型是 Mask R-CNN 还是 MaskDINO。

### 2.2 预训练初始化与完整下游恢复必须分离

`MODEL.WEIGHTS` 仅用于恢复完整 Mask R-CNN / MaskDINO checkpoint；SSL 权重通过独立字段初始化 image encoder。优先级为：

```text
resume downstream checkpoint
  > complete downstream MODEL.WEIGHTS
  > ImageNet encoder initialization（若适用）
  > random initialization（仅保留为可选诊断，不作为主实验行）
```

这避免把 SSL checkpoint 错当作检测模型 checkpoint，并确保下游训练继续训练时不会重复覆盖已经微调过的 backbone。

### 2.3 主结论以“通用默认初始化”与同架构对照为准

跨家族比较回答的是实际选型问题；Swin、ResNet、ViT 的参数量、特征步长和归纳偏置不同，不能将其简单解释为 SSL 算法本身的优劣。

主实验使用各 family 的通用默认权重作为参照：ImageNet encoder 初始化优先；对于已有完整 COCO 检测模型的配置，COCO checkpoint 作为单独的工程基线。主比较为：

```text
ImageNet Swin-T  vs  Swin-MAE Swin-T  vs  DentMag-Swin-T
ImageNet ViT     vs  DINOv2 / ViT-MAE  vs  DentMag-ViT
ImageNet R50     vs  SimCLR-R50
```

其中每一行的下游模型、feature adapter、训练数据、增强、输入尺度、优化器、训练步数均保持一致。COCO 完整 Mask R-CNN / MaskDINO checkpoint 除加载 backbone 外还携带检测、类别甚至 mask 监督，因此必须单列为“COCO full-model engineering baseline”，不得混入只比较 encoder 初始化的 SSL 排名。

## 3. 模块命名与职责规范

目录建议：

```text
maskdino/modeling/backbone/
  ssl_common.py
  swin_ssl_backbone.py
  resnet_ssl_backbone.py
  vit_ssl_backbone.py
  adapters/
    swin_adapter.py
    resnet_adapter.py
    vit_adapter.py
```

名称和责任严格区分：

| 名称 | 责任 | 不负责的事情 |
|---|---|---|
| `SwinSSLBackbone` | 构造 Swin encoder、加载 Swin SSL 权重、输出原生四级 stage feature | 不实现 Detectron2 任务头，不加载 MAE decoder |
| `SwinAdapter` | 把 Swin stage feature 标准化为 Detectron2 的 `res2–res5` | 不包含 DentMag / Swin-MAE checkpoint 解析逻辑 |
| `ResNetSSLBackbone` | 构造 SimCLR 对应 ResNet、加载其 encoder 权重、导出四级 residual stage | 不使用 SimCLR projection head |
| `ResNetAdapter` | 规范化 ResNet stage 输出为 `res2–res5` | 不伪造高分辨率 feature |
| `ViTSSLBackbone` | 构造 DINOv2、ViT-MAE 或 DentMag-ViT encoder，按指定 block 取 token | 不承载下游检测/分割逻辑 |
| `ViTAdapter` | 使用统一 token-to-feature-pyramid，把 token maps 转为 `res2–res5` | 不使用复杂的 DINO ViTAdapter / deformable interaction 模块 |

建议在 Detectron2 注册的实际 backbone 名为：

```text
build_swin_ssl_backbone
build_swin_ssl_fpn_backbone
build_resnet_ssl_backbone
build_resnet_ssl_fpn_backbone
build_vit_ssl_backbone
build_vit_ssl_fpn_backbone
```

其中 `*_fpn_backbone` 仅供 Mask R-CNN 使用；MaskDINO 使用无 FPN 版本，直接消费 `res2–res5`。这与现有 `build_swin_mae_fpn_backbone` 的使用方式一致。

## 4. 统一配置与 checkpoint 描述

所有 SSL backbone 使用同一套配置字段，避免不同方法各自发明 YAML 结构：

```yaml
MODEL:
  BACKBONE:
    NAME: "build_swin_ssl_backbone"
    SSL_METHOD: "dentmag_swin"      # imagenet | swinmae | dentmag_swin | simclr | dinov2 | vitmae | dentmag_vit
    USE_SSL_PRETRAINED: true
    SSL_WEIGHTS: "/absolute/path/to/checkpoint.pth"
    SSL_CHECKPOINT_KEY: "encoder_teacher"
    SSL_STRICT_LOAD: true
    FREEZE_AT: 0
    FEATURE_NAMES: ["res2", "res3", "res4", "res5"]
```

不同 family 增加专属字段：

```yaml
MODEL:
  SWIN_SSL:
    EMBED_DIM: 96
    DEPTHS: [2, 2, 6, 2]
    NUM_HEADS: [3, 6, 12, 24]
    PATCH_SIZE: 4
    WINDOW_SIZE: 7
    POSITION_ENCODING: "dentmag_sincos"  # none | dentmag_sincos | native_fixed

  RESNET_SSL:
    DEPTH: 50
    WIDTH_MULTIPLIER: 1
    STEM_TYPE: "simclrv2"
    SK_RATIO: 0.0625

  VIT_SSL:
    ARCH: "vit_base"
    PATCH_SIZE: 14
    EMBED_DIM: 768
    DEPTH: 12
    NUM_HEADS: 12
    NUM_REGISTER_TOKENS: 0
    OUT_BLOCKS: [3, 6, 9, 12]
    PYRAMID_CHANNELS: 256
    PYRAMID_NORM: "LN"
```

`SSL_METHOD=imagenet` 表示只初始化与该 encoder 同构的 ImageNet 权重，检测/分割 task head 仍随机初始化。完整 COCO 下游 checkpoint 不通过 `SSL_METHOD` 加载，而是使用 `MODEL.WEIGHTS` 恢复，并在实验记录中标记为 full-model baseline。可选的随机初始化只用于 smoke test 或训练诊断，不能替代 ImageNet/COCO 主基线。

## 5. 公共加载器：`ssl_common.py`

公共加载逻辑应独立于各 family，包含：

1. 安全加载可信的本地 checkpoint；
2. 根据点分路径获取状态字典，例如 `encoder_teacher`、`encoder`、`teacher`、`model`、`resnet`；
3. 删除训练时专用模块：decoder、projection head、predictor、optimizer、EMA step 等；
4. 去除公共前缀：`module.`、`model.`、`backbone.`；
5. 严格比较共享 key 的数量、shape 和加载比例；
6. 仅在明确允许的情况下对 ViT `pos_embed` 做 bicubic 插值；
7. 输出可审计日志：checkpoint 路径、哈希、选择的 key、加载/跳过 tensor 数、shape mismatch 列表。

禁止“`strict=False` 后静默继续”。若未加载任何关键 encoder 层，或共享 tensor 的 mismatch 超过预设阈值，应直接失败。DentMag 的 [backbones.py](../../dentmag/application/models/backbones.py) 中已有 fail-loud loader 和 ViT position embedding 重采样逻辑，可作为参考，但新实现不应依赖该应用侧分割框架。

## 6. Swin 路径

### 6.1 `SwinSSLBackbone`

Swin 是本项目第一优先级，因为天然提供层次化 feature pyramid。每级 feature 必须在 `PatchMerging` 前截取：

```text
patch embedding → stage 1 → res2 (stride 4,  C=96)
              → merge → stage 2 → res3 (stride 8,  C=192)
                      → merge → stage 3 → res4 (stride 16, C=384)
                              → merge → stage 4 → res5 (stride 32, C=768)
```

`SwinSSLBackbone` 支持：

- `swinmae`：加载 baseline Swin-MAE 的 `model` 或 `state_dict` 中 encoder 部分；丢弃 `layers_up`、`norm_up`、`decoder_pred`；
- `dentmag_swin`：加载 DentMag checkpoint 的 `encoder_teacher`，缺失时回退 `encoder`；丢弃 decoder、global/patch projection heads、optimizer；
- `imagenet`：加载同结构 ImageNet Swin 权重；下游 task head 不加载。

架构必须精确匹配 checkpoint；例如 Swin-T 为 `96/[2,2,6,2]/[3,6,12,24]`，不能用当前某些旧配置中的 `[2,2,2,2]` 作为它的下游容器。

### 6.2 `SwinAdapter`

`SwinAdapter` 应尽量薄，只做：

- BHWC 到 NCHW；
- 可选 LayerNorm；
- feature 名称和 stride 的标准化；
- 对 window attention 所需输入 padding 的处理。

DentMag-Swin 的 encoder 在 patch embedding 后叠加动态二维 sin-cos positional embedding；该行为必须通过 `POSITION_ENCODING=dentmag_sincos` 保留。Swin-MAE 路径遵循其原生 feature forward 行为。不能为了方便让两者都走“无位置编码”的当前 wrapper，否则 DentMag 权重可加载但不再是原预训练 encoder 的推理分布。

### 6.3 下游连接

- Mask R-CNN：`build_swin_ssl_fpn_backbone` 在 `SwinSSLBackbone + SwinAdapter` 外包 Detectron2 FPN，生成 P2–P6；
- MaskDINO：`build_swin_ssl_backbone` 直接提供 `res2–res5` 给现有 pixel decoder。

## 7. ResNet / SimCLR 路径

### 7.1 `ResNetSSLBackbone`

SimCLR 只贡献 ResNet image encoder。projection head 不进入 downstream。adapter 必须能取到真实 stage 输出：

```text
stem + layer1 → res2
layer2        → res3
layer3        → res4
layer4        → res5
```

DentMag 当前 [SimCLRv2Wrapper](../../dentmag/application/models/backbones.py) 只返回最终 stride-32 feature，足以服务其原来的单尺度分割 head，但不适合目标检测和实例分割。

需要先检查 SimCLRv2 ResNet 和 Detectron2 ResNet 是否逐层同构：

- 若 stem、bottleneck、normalization、stride placement、SK block 均一致，可写 checkpoint key converter 并复用 Detectron2 ResNet；
- 若不一致，保留 SimCLRv2 原生 encoder，写 `ResNetSSLBackbone` 作为 Detectron2 `Backbone`，并显式返回四级 feature。

后者是更稳妥的默认方案。不得只上采样 `res5` 伪造 `res2–res4`，因为这会抹去 SimCLR 的低层局部信息并对小病灶、边界 mask 和小 bbox 极其不利。

### 7.2 `ResNetAdapter`

`ResNetAdapter` 仅负责将原生 stage 输出转为标准 names/ShapeSpec，并处理通道和 stride 元信息。对标准 R50：

```text
res2=256, res3=512, res4=1024, res5=2048
```

如 `width_multiplier != 1`，channels 必须由 encoder 实际输出推导，禁止写死标准 ResNet-50 通道数。

### 7.3 下游连接

- Mask R-CNN：沿用 FPN；
- MaskDINO：沿用现有 `res2–res5` 及 pixel decoder 配置。

## 8. ViT 路径：统一轻量 token-to-feature-pyramid

### 8.1 设计取舍

DINOv2、ViT-MAE 和 DentMag-ViT 原生输出为单尺度 token grid。若直接交给 Mask R-CNN / MaskDINO，会缺少 feature pyramid。

本项目不采用 DINO 仓库中复杂的 `ViTAdapter`，因为它含 spatial prior、deformable attention、interaction blocks 和额外随机初始化参数。该方案更像一个新的大 backbone；它可能提高性能，却会混淆“SSL encoder 增益”和“复杂 adapter 增益”。

替代方案为一个所有 ViT family 共用的轻量 `ViTAdapter`，其全称为 **token-to-feature-pyramid adapter**。

### 8.2 `ViTSSLBackbone`

职责：构造原生 ViT 并在若干深度抽取 patch tokens。

| 方法 | checkpoint key 的优先顺序 | token 提取方式 |
|---|---|---|
| ImageNet-ViT | 实际 checkpoint schema | 同一 ViT 配置下按来源实现 |
| DINOv2 | `teacher` → `model` → `state_dict` | `get_intermediate_layers` 或等价 hooks |
| ViT-MAE | `model` → `state_dict` | 无 mask 的 patch embedding + blocks + norm |
| DentMag-ViT | `encoder_teacher` → `encoder` | `forward_clean` 或等价 block collector |

所有路径均要：

1. 删除 CLS token 与 register token；
2. 将 `[B, L, C]` 恢复为 `[B, C, Hpatch, Wpatch]`；
3. 抽取相同数量的 block，例如 12 层 ViT 用 `[3, 6, 9, 12]`；
4. 对非正方形输入按真实 `Hpatch/Wpatch` reshape，不得使用 `sqrt(L)` 假定正方形；
5. 复用各模型原生的 position embedding 插值逻辑，或在确有必要时由公共 loader 进行安全重采样。

### 8.3 `ViTAdapter`：统一 token-to-feature-pyramid

adapter 输入为来自多个 transformer 深度的 token maps，输出为 Detectron2 标准四级 feature。

建议的轻量结构：

```text
block 3  token map ─→ 1×1 projection ─→ resize to H/4  ─→ res2
block 6  token map ─→ 1×1 projection ─→ resize to H/8  ─→ res3
block 9  token map ─→ 1×1 projection ─→ resize to H/16 ─→ res4
block 12 token map ─→ 1×1 projection ─→ resize to H/32 ─→ res5
```

每一级使用相同规则的：

```text
1×1 Conv(Cvit → Cpyramid) → LayerNorm / GroupNorm → 可选 3×3 smoothing Conv
```

推荐第一版不加入跨尺度 attention、deformable attention、CNN spatial-prior 分支或 encoder-token interaction。若加入 3×3 smoothing，其结构、通道数、初始化、训练策略必须在 ImageNet-ViT、DINOv2、ViT-MAE 与 DentMag-ViT 中完全一致。

对于 DINOv2 patch size=14，原生 token stride 不是 2 的幂。adapter 仍将输出 resize 到 `H/4,H/8,H/16,H/32` 并在 `output_shape()` 中声明 stride `4,8,16,32`。这是为了满足 Detectron2 FPN 和 MaskDINO pixel decoder 的尺度契约；因此 adapter 是模型的一部分，必须对所有 ViT 方法保持不变。

输入 padding 应设为 `lcm(patch_size, 32)`：

```text
DINOv2 p14: lcm(14, 32) = 224
ViT-MAE p16: lcm(16, 32) = 32
```

为支持混合实验，建议 ViT 配置默认 `size_divisibility=224`。这会增加小图 padding 和显存成本，需在实验表中记录实际 batch size 与有效输入尺寸。

### 8.4 为什么这仍然是可比的

该 `ViTAdapter` 会引入少量可训练投影参数，但它：

- 在所有 ViT 方法之间逐参数同构；
- 不读取特定 SSL 方法的 checkpoint；
- 不承担检测/实例分割预测；
- 将主要任务能力仍交由 Mask R-CNN 或 MaskDINO 学习。

报告中应把模型写为：

```text
DINOv2 + common ViTAdapter + Mask R-CNN
DINOv2 + common ViTAdapter + MaskDINO
ViT-MAE + common ViTAdapter + Mask R-CNN / MaskDINO
DentMag-ViT + common ViTAdapter + Mask R-CNN / MaskDINO
```

## 9. Mask R-CNN 与 MaskDINO 的接入方式

### 9.1 Mask R-CNN

Mask R-CNN 的配置指向相应 `build_*_ssl_fpn_backbone`。该 builder：

```text
<Family SSLBackbone> + <Family Adapter>
  → {res2,res3,res4,res5}
  → Detectron2 FPN
  → P2–P6
  → RPN / ROI box head / mask head
```

检测和 mask head 重新初始化并参与下游训练。FPN 对所有同一 encoder family 的对照应相同。

### 9.2 MaskDINO

MaskDINO 的配置指向 `build_*_ssl_backbone`。链路为：

```text
<Family SSLBackbone> + <Family Adapter>
  → {res2,res3,res4,res5}
  → MaskDINO pixel decoder
  → transformer decoder
  → classes / boxes / masks
```

保持当前 `SEM_SEG_HEAD.IN_FEATURES` 和 `DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES` 的名称一致。MaskDINO 的 pixel decoder 已负责通道投影、多尺度融合和 query-side 预测，不应由 SSL adapter 复制这些工作。

## 10. 训练策略

每个 `(encoder method, downstream model, task)` 运行三种可选训练模式：

| 模式 | Backbone | 新建 adapter / FPN / task heads | 用途 |
|---|---|---|---|
| linear-style warm-up | 冻结短周期 | 训练 | 排查权重与 feature 接口，非最终主结果 |
| full fine-tune | 训练，较小 LR multiplier | 训练 | 主结果 |
| ImageNet baseline | 加载同构 ImageNet encoder，较小 LR multiplier | 训练 | 与通用视觉预训练比较 |
| COCO full-model baseline | 恢复完整 COCO checkpoint | 按 checkpoint 恢复 | 工程参考；不进入 encoder-only SSL 排名 |

主结果建议使用 full fine-tune，并设置 backbone LR multiplier（如 `0.1`）而非长期冻结。DentMag 自监督价值最终应通过可迁移后的全模型能力体现。

应统一：

- train/val/test 的病例级划分；
- 下游数据增强；
- 输入尺度、padding 策略；
- optimizer、global batch、学习率 schedule、训练 iteration；
- 类别不平衡策略、负样本保留比例；
- Mask R-CNN 与 MaskDINO 各自的检测/分割损失设置。

SSL 预训练数据必须审计是否含 downstream test 病例。若包含，结果只能标为 transductive/in-domain SSL，不能作为严格未见病例泛化结果。

## 11. 验证与失败保护

### 11.1 单元级验证

每个 family 都必须在训练前验证：

1. checkpoint 关键 tensors 成功加载；
2. `forward()` 返回恰好四项 `res2–res5`；
3. batch、channel、空间尺寸、stride 与 `output_shape()` 一致；
4. 不同长宽比的图像可以通过；
5. 含 padding 的输入不会在 window attention 或 token reshape 出错；
6. Mask R-CNN 可以跑一次 loss forward 和一次 inference；
7. MaskDINO 可以跑一次 loss forward 和一次 inference。

### 11.2 推荐 shape smoke test

对输入 `[2, 3, 896, 896]`，断言：

```text
res2 spatial = 224 × 224
res3 spatial = 112 × 112
res4 spatial = 56 × 56
res5 spatial = 28 × 28
```

对非正方形输入也测试 `[2, 3, 672, 896]`。ViT 路径应依据真实 token grid reshape，不能以 `sqrt(num_tokens)` 推断。

### 11.3 训练级 sanity checks

- ImageNet 初始化与 SSL 初始化的可训练参数数目一致；
- SSL checkpoint 不改变 task head 的随机初始化；
- 训练第一个 iteration 的 loss 有限；
- backbone 梯度在 full fine-tune 模式中非零；
- 所有方法用相同评估脚本与 COCOEvaluator；
- 记录 checkpoint hash、adapter 版本、配置文件快照。

## 12. 实施顺序

1. 抽取现有 Swin loader 的公共部分，建立 `ssl_common.py` 和统一 config schema。
2. 完成 `SwinSSLBackbone + SwinAdapter`，先验证 ImageNet / Swin-MAE / DentMag-Swin 在 Mask R-CNN 与 MaskDINO 的 smoke test。
3. 确定 SimCLRv2 源码与具体 checkpoint 格式，完成真实多级 `ResNetSSLBackbone + ResNetAdapter`。
4. 实现 `ViTSSLBackbone + common ViTAdapter`，先接 ImageNet-ViT，再接 ViT-MAE、DINOv2、DentMag-ViT。
5. 为每种 family 添加 Mask R-CNN、MaskDINO、caries、orth 的最小配置组合。
6. 建立权重加载、feature shape、两类下游模型 forward 的自动 smoke tests。
7. 先跑 Swin 三元组的小规模正畸实验，确认评估链路后再扩展全矩阵。

## 13. 交付物

- 三个 Detectron2 SSL backbone：`SwinSSLBackbone`、`ResNetSSLBackbone`、`ViTSSLBackbone`；
- 三个标准化 feature adapter：`SwinAdapter`、`ResNetAdapter`、`ViTAdapter`；
- 统一 checkpoint loader 与严格加载报告；
- 每个 family 的 Mask R-CNN / MaskDINO 配置；
- ImageNet baseline / COCO full-model baseline / baseline SSL / DentMag SSL 的可复现实验矩阵；
- feature contract、加载日志和 forward smoke tests；
- 病例级 SSL 数据泄漏审计记录。

该设计保证 adapter 只解决“把预训练表征变成 Detectron2 多尺度输入”的问题，检测与分割的主要建模能力持续由 Mask R-CNN 和 MaskDINO 承担。
