# scripts

`intraoral` 编码器对比实验的启动脚本。目录结构与 config 树一一对应：每个
`(detector, task)` 组合一个脚本，脚本内部遍历该目录下的 5 个编码器变体。

```
scripts/
  common.sh                       # 公共函数 run_train / run_eval（仅用于 source）
  train/
    train_maskrcnn_caries.sh      # configs/maskrcnn/caries/*.yaml
    train_maskrcnn_orth.sh        # configs/maskrcnn/orth/*.yaml
    train_maskdino_caries.sh      # configs/maskdino/caries/*.yaml
    train_maskdino_orth.sh        # configs/maskdino/orth/*.yaml
  eval/
    eval_maskrcnn_caries.sh       # maskrcnn_pred.py 读取 configs/maskrcnn/caries/*.yaml
    eval_maskrcnn_orth.sh
    eval_maskdino_caries.sh
    eval_maskdino_orth.sh
```

启动脚本本身只有一行循环，入口文件名由 detector 推导
（训练 `<detector>_unify.py`，评估 `<detector>_pred.py`）：

```bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

for variant in "${VARIANTS[@]}"; do
  run_train maskdino orth "${variant}"   # 或 run_eval maskdino orth "${variant}"
done
```

## 运行矩阵

| Detector | Task | 入口（训练 / 评估） |
| --- | --- | --- |
| Mask R-CNN | caries（实例分割） | `maskrcnn_unify.py` / `maskrcnn_pred.py` |
| Mask R-CNN | orth（仅检测框） | `maskrcnn_unify.py` / `maskrcnn_pred.py` |
| MaskDINO | caries（实例分割） | `maskdino_unify.py` / `maskdino_pred.py` |
| MaskDINO | orth（仅检测框） | `maskdino_unify.py` / `maskdino_pred.py` |

每个脚本都会遍历 5 个变体
`no_ssl simclr swinmae dentmag_swin dinov2`，并解析
`configs/<detector>/<task>/<variant>.yaml`。**`OUTPUT_DIR` / `SWANLAB.NAME`
都取自变体 YAML** —— 脚本里不做任何硬编码，因此一次运行的身份就等同于它的
config 路径。

## 用法

先激活环境（`conda activate intraoral`），或用 `conda run -n intraoral` 包裹
调用。脚本会自行定位仓库根目录，因此在任何工作目录下都能运行。

```bash
# 单卡训练，全部 5 个变体，MaskDINO + orth
CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_maskdino_orth.sh

# 双卡，只跑两个 Swin 变体
CUDA_VISIBLE_DEVICES=1,2 VARIANTS="swinmae dentmag_swin" \
  bash scripts/train/train_maskrcnn_orth.sh

# 评估所有已训练的 Mask R-CNN caries checkpoint，只跑 test
CUDA_VISIBLE_DEVICES=0 EVAL_SPLITS=test bash scripts/eval/eval_maskrcnn_caries.sh

# 只预览命令，不真正运行
DRY_RUN=1 bash scripts/train/train_maskdino_caries.sh
```

## 可调参数

全部可选，均为环境变量（详见 `scripts/common.sh`）：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | 暴露哪些 GPU。`DISTRIBUTED.NUM_GPUS` 由可见卡数推导，因此 N 个 id ⇒ N 路数据并行。 |
| `VARIANTS` | 全部 5 个 | 编码器变体子集，如 `VARIANTS="no_ssl simclr"`。 |
| `INPUT_DIR` | `single_ch_0225`（caries）/ `orth_0616`（orth） | 数据集根目录。 |
| `IMAGE_SUBDIR` | `images` | 每个样本内的图像子目录；orth 运行可传 `tooth_mask`。 |
| `BATCH_SIZE` | config 中的 `DISTRIBUTED.IMS_PER_GPU` | 单卡 batch size（仅训练）。 |
| `EVAL_SPLITS` | `train test` | 要评估的 split（仅评估）。 |
| `EXTRA_ARGS` | – | 原样透传的额外命令行参数，如 `EXTRA_ARGS="--resume"`。 |
| `DRY_RUN` | `0` | 设为 `1` 时只打印命令而不执行。 |

各变体的 `OUTPUT_DIR` 就是 config 树里写的值，例如
`output/maskdino_orth_swinmae`、`output/maskrcnn_caries_dinov2`。
