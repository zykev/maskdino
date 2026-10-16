# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path

# fmt: off
import sys
# 让脚本无论从哪个目录启动，都能 import 仓库里的 maskdino / utils 包。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(REPO_ROOT))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo

from utils.orth_class_balance import compute_effective_class_weights
from utils.task_paths import DEFAULT_INPUT_DIRS, add_input_dir_arg, resolve_task_paths
from utils.unify_common import load_categories, register_datasets


# constants
WINDOW_NAME = "maskdino demo"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "maskdino" / "orth" / "no_ssl.yaml"


def apply_task_config(args, cfg):
    """把训练脚本注入的、推理时同样必需的配置项补齐。

    这些值在 yaml 里没有写死，只有 `maskdino_unify.setup` 会根据数据集算出来：

    * ``MODEL.SEM_SEG_HEAD.NUM_CLASSES``：yaml 缺省是 80，不改会导致加载权重时
      类别数对不上；
    * ``ORTH_BBOX_ONLY``：orth 只有框标注，用它关掉 mask 分支；
    * ``ORTH_CLASS_BALANCE.CLASS_WEIGHTS``：``SetCriterion`` 会校验长度必须等于
      类别数，训练脚本按有效样本数权重算出，这里复用同一套算法。

    另外注册本项目的数据集，可视化时才能画出类别名。返回 val 数据集名。
    """
    task_args = resolve_task_paths(
        argparse.Namespace(
            task=args.task,
            input_dir=args.input_dir or REPO_ROOT / DEFAULT_INPUT_DIRS[args.task],
            image_subdir=args.image_subdir,
            keep_negative_ratio=0.2,
            seed=42,
        )
    )
    class_names = load_categories(task_args.train_json, args.task)
    register_datasets(task_args, class_names, include_masks=args.task == "caries")

    num_classes = len(class_names)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.ORTH_BBOX_ONLY = args.task == "orth"
    balance = getattr(cfg, "ORTH_CLASS_BALANCE", None)
    if args.task == "orth" and balance is not None and balance.ENABLED:
        cfg.ORTH_CLASS_BALANCE.CLASS_WEIGHTS = compute_effective_class_weights(
            task_args.train_json,
            num_classes,
            beta=float(balance.BETA),
            clip_min=float(balance.CLIP_MIN),
            clip_max=float(balance.CLIP_MAX),
        )
    return f"{args.task}_val"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    # 与 maskdino_unify._build_base_cfg 保持一致：本项目在 config 里自定义了
    # DATALOADER.KEEP_NEGATIVE_RATIO / AUGMENTATION / ORTH_CLASS_BALANCE /
    # SWANLAB 等键，必须先放开未知键才能 merge。
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # 指向本项目注册的 val 数据集；标注文件缺失时降级为不带类别名的可视化。
    try:
        cfg.DATASETS.TEST = (apply_task_config(args, cfg),)
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(f"跳过数据集注册，可视化将不显示类别名：{exc}")
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="intraoral maskdino / maskrcnn demo")
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG),
        metavar="FILE",
        help="path to config file (defaults to the intraoral orth baseline)",
    )
    parser.add_argument(
        "--task",
        default="orth",
        choices=("orth", "caries"),
        help="intraoral task of the checkpoint; drives the dataset metadata",
    )
    add_input_dir_arg(parser)
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
