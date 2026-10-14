# 预训练输入分辨率与数据增强对比

本文汇总当前配置与训练代码中的输入分辨率、视图构造和数据增强。`no_odontify` 指病例级数据划分：排除 Odontify，保留自有采集者与 AIDC；它本身不是一种 data augmentation。

| 方法 | 实际/配置输入 | 训练 augmentation 与视图 |
| --- | --- | --- |
| 我们 Swin hybrid | full 448²；tooth 224²；sextant 448×224；random 224² | 同一次 full-image 几何变换同步作用于所有 bbox；再从该图重新裁 tooth/sextant/random crop，并 letterbox 到各自画布。 |
| 我们 Swin highres | full 896²；tooth 224²；sextant 896×448；random 448² | augmentation 规则不变，仅提高 full、多牙 crop 分辨率。 |
| 我们 ViT hybrid | full 448²；tooth 224²；sextant 448×224；random 224² | 与 Swin hybrid 同一套数据与增强逻辑。 |
| 我们 ViT highres | full 518²；tooth 224²；sextant/random 448² | augmentation 不变；518 是 DINOv2 ViT/14 的原生 37×37 patch grid。 |
| SimCLRv2 `no_odontify` | 默认 448² | 真正的 continued SSL：每张 full-mouth image 独立生成两个 SimCLR 视图，使用 RandomResizedCrop、水平翻转、ColorJitter、灰度和 GaussianBlur，并优化 symmetric NT-Xent；不使用 FDI/tooth/sextant/view 标签，也不使用 annotation bbox crop。 |
| Swin-MAE | 当前仓库基础配置是 224²；registry 声明牙科 adapted run 为 448² | `RandomResizedCrop(.5–1, .9–1.1)` + flip + ColorJitter(p=.5) + ±10° rotation + GaussianBlur(p=.2)。 |
| DINOv2 | 通用 DINO 默认：2×224 global + 8×96 local；ViT/14 常见 local 为 98 | global/local RandomResizedCrop + flip；强 color jitter、灰度、blur，第二 global crop 额外 solarize。 |

## 说明

- 我们的方法采用轻量、bbox 同步的缩放（0.8–1.25）、平移（每轴最多原图 10%）和 ColorJitter；不使用随机旋转，以保持 full 图、局部 crop 与 bbox 的几何对应关系。
- SimCLRv2 的默认 shell 配方与原 `finetune_dental_no_odontify.sh` 的 full-mode 优化设置对齐：50 epochs、5 epochs warmup、AdamW `lr=3e-4` 至 `3e-5`、`weight_decay=.05`、每 10 epoch 保存。其训练目标和数据视图已改为 SSL，不再是原先的多任务监督微调。
- Swin-MAE 与 DINOv2 的 `*_no_odontify.sh` 在当前 checkout 中缺失，因此表中 baseline 的分辨率/增强分别来自当前可见的基础实现、registry 以及通用 DINO 配置；不能将其视作缺失 wrapper 的逐项运行审计。
