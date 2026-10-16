## MaskDINO 推理可视化 Demo

`demo.py` 加载本仓库的 config 和训练好的 checkpoint，对图片 / 视频 / 摄像头
逐帧推理，并在 OpenCV 窗口里显示检测框、掩码和类别名。底层就是 detectron2 的
`DefaultPredictor` + `Visualizer`，更多背景见
[Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)。

### 前置条件

1. 一条可用的 config：任意 `configs/<detector>/<task>/<variant>.yaml`，默认
   `configs/maskdino/orth/no_ssl.yaml`。
2. 一个训练好的 checkpoint，例如 `output/maskdino_orth_resnet/model_final.pth`，
   用 `--opts MODEL.WEIGHTS <path>` 传入（必须给，否则 `DefaultPredictor` 直接报错）。
3. 与训练相同的环境（见 [`../INSTALL.md`](../INSTALL.md)），因为 demo 要
   `import maskdino` 和 `utils`；另外需要 `opencv-python`(`cv2`) 和 `tqdm`。

> **config 和 checkpoint 必须配套**：`--config-file` 决定模型结构，checkpoint 必须
> 是用同一份 config 训出来的。加载时会打印 `Skip loading parameter ...` 之类的
> warning，说明结构对不上，这时的预测结果没有意义。
> 已知一处历史漂移：`output/maskdino_orth_resnet/` 是在
> `FEATURE_ORDER: high2low` 时训练的，而 `configs/maskdino/orth/_base.yaml` 后来
> 改成了 `low2high`。所以用默认 config 加载这个 resnet checkpoint 会跳过两个
> `pixel_decoder.input_proj.*` 参数，输出的是乱框（数量正好是
> `TEST.DETECTIONS_PER_IMAGE`）。要复现该实验请用 `output/maskdino_orth_resnet/
> config.yaml` 作为 `--config-file`；`output/` 下另外三个 MaskDINO run 与当前
> config 一致。

### 运行

在**仓库根目录**执行：

```bash
mkdir -p /tmp/demo_out   # 脚本不会自动建目录，目录不存在时保存会静默失败
python demo/demo.py \
  --config-file configs/maskdino/orth/no_ssl.yaml \
  --task orth \
  --input ".datasets/intraoral_anno/orth_0616/orth_0616/*/images/*.png" \
  --output /tmp/demo_out \
  --opts MODEL.WEIGHTS output/maskdino_orth_resnet/model_final.pth
```

`--config-file` 默认就是上面这个 orth baseline，所以可以省略。`--input` 接受单个
glob 模式（会被 `glob.glob` 展开），或者用空格分隔的多个文件；不传 `--output`
时会弹 OpenCV 窗口显示结果（按 `Esc` 退出）。

### 常用参数

* `--config-file` — 模型结构完全由它决定，默认 `configs/maskdino/orth/no_ssl.yaml`。
* `--task` — `orth`（默认）或 `caries`。只影响数据集元数据的注册，也就是可视化的
  类别名和配色，不改变模型本身。
* `--input_dir` / `--image_subdir` — 数据集根目录和图片子目录名，默认分别是
  `utils/task_paths.py` 里的 `DEFAULT_INPUT_DIRS[task]` 和 `images`。
* `--confidence-threshold` — 低于该分数的实例不画，默认 `0.5`。
* `--opts` — 透传给 config 的 `KEY VALUE` 对，最常用的是
  `MODEL.WEIGHTS <checkpoint>`；在 CPU 上跑再加 `MODEL.DEVICE cpu`。
  注意 `--opts` 是 `argparse.REMAINDER`，必须放在命令行最后。
* `--webcam` / `--video-input <file>` — 摄像头 / 视频输入，替代 `--input`。

> 类别名来自 `--task` 对应 COCO 标注文件里的 `categories`。如果标注文件不在
> `--input_dir` 下，脚本会打一条 warning 并跳过注册，此时只画框和掩码、不写
> 类别名。


