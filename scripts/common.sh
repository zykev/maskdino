#!/usr/bin/env bash
# scripts/{train,eval} 下所有启动脚本的公共函数。只能 source，不要直接执行。
#
# 每个启动脚本负责一个 (detector, task) 组合，内部遍历 5 个编码器变体：
#     no_ssl  simclr  swinmae  dentmag_swin  dinov2
# 配置路径就是运行身份：configs/<detector>/<task>/<variant>.yaml
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES  暴露哪些 GPU（默认 0），N 个 id 即 N 路数据并行
#   VARIANTS              变体子集，如 VARIANTS="no_ssl simclr"（默认全部 5 个）
#   INPUT_DIR             数据集根目录（默认按 task 取，见 default_input_dir）
#   IMAGE_SUBDIR          样本内的图像子目录（默认 images；orth 可用 tooth_mask）
#   BATCH_SIZE            单卡 batch size（仅训练，默认用配置里的值）
#   EVAL_SPLITS           要评估的 split（仅评估，默认 "train test"）
#   EXTRA_ARGS            额外命令行参数，原样透传
#   DRY_RUN=1             只打印命令，不执行

set -euo pipefail

# 切到仓库根目录，这样在任何工作目录下调用都能找到 configs/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ALL_VARIANTS=(no_ssl simclr swinmae dentmag_swin dinov2)
read -r -a VARIANTS <<<"${VARIANTS:-${ALL_VARIANTS[*]}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 两个任务各自的数据集根目录
# （orth_0616 是正畸的标准数据；orth_test 的 JSON 与磁盘不一致）
default_input_dir() {
  case "$1" in
    caries) echo .datasets/intraoral_anno/single_ch_0225 ;;
    orth)   echo .datasets/intraoral_anno/orth_0616 ;;
  esac
}

# 打印并执行命令；DRY_RUN=1 时只打印
run() {
  printf '==> CUDA_VISIBLE_DEVICES=%s %s\n' "${CUDA_VISIBLE_DEVICES}" "$*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

# 训练一个变体，入口是 <detector>_unify.py
#   run_train <detector> <task> <variant>
run_train() {
  local detector="$1" task="$2" variant="$3"
  local config="configs/${detector}/${task}/${variant}.yaml"
  [[ -f "${config}" ]] || { echo "缺少配置：${config}" >&2; return 1; }

  local -a args=(--task "${task}" --config_file "${config}"
                 --input_dir "${INPUT_DIR:-$(default_input_dir "${task}")}")
  [[ -n "${IMAGE_SUBDIR:-}" ]] && args+=(--image_subdir "${IMAGE_SUBDIR}")
  [[ -n "${BATCH_SIZE:-}" ]] && args+=(--batch_size "${BATCH_SIZE}")

  run python "${detector}_unify.py" "${args[@]}" ${EXTRA_ARGS:-}
}

# 评估一个变体，入口是 <detector>_pred.py
#   run_eval <detector> <task> <variant>
# --output_dir 直接读变体 YAML 顶层的 OUTPUT_DIR（checkpoint 即
# <OUTPUT_DIR>/model_final.pth），所以这里不用再写一遍路径。
run_eval() {
  local detector="$1" task="$2" variant="$3"
  local config="configs/${detector}/${task}/${variant}.yaml"
  [[ -f "${config}" ]] || { echo "缺少配置：${config}" >&2; return 1; }

  local -a args=(--task "${task}" --config_file "${config}"
                 --input_dir "${INPUT_DIR:-$(default_input_dir "${task}")}"
                 --output_dir "$(awk '/^OUTPUT_DIR:/{print $2; exit}' "${config}")")
  [[ -n "${IMAGE_SUBDIR:-}" ]] && args+=(--image_subdir "${IMAGE_SUBDIR}")
  if [[ -n "${EVAL_SPLITS:-}" ]]; then
    local -a splits
    read -r -a splits <<<"${EVAL_SPLITS}"
    args+=(--eval_splits "${splits[@]}")
  fi

  run python "${detector}_pred.py" "${args[@]}" ${EXTRA_ARGS:-}
}
