#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Set up the shared conda env for the intraoral project suite
# (maskdino / intraoral / dentmag) on a fresh machine.
#
# Target env (matches the reference env of 2026-10-15):
#   python 3.12 | torch 2.7.0+cu118 | conda gcc/gxx 11.2 | system CUDA 11.8
#
# Usage:
#   bash setup.sh              # env name defaults to "intraoral"
#   bash setup.sh my_env       # custom env name
#
# Optional environment variables:
#   DETECTRON2_DIR=/path/to/detectron2  default: <parent of repo>/detectron2
#   CUDA_HOME=/usr/local/cuda-11.8      must contain bin/nvcc (system toolkit)
#   SKIP_SAM3=1                         skip the optional sam3 install
#   FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6"
#                                       needed only when compiling on a node
#                                       without a visible GPU
# ---------------------------------------------------------------------------
set -e

ENV_NAME="${1:-intraoral}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DETECTRON2_DIR="${DETECTRON2_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/detectron2}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo ">> conda env '${ENV_NAME}' already exists, reusing it"
else
    conda create -y -n "${ENV_NAME}" python=3.12 pip
fi
conda activate "${ENV_NAME}"

pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# PyTorch with CUDA 11.8 support (default PyPI wheel is cu12x, so pin the index)
# torchaudio is kept only for parity with the reference env; none of the three
# projects import it.
# ---------------------------------------------------------------------------
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------------------------
# C/C++ compilers (reference env uses conda gcc/gxx 11.2)
# ---------------------------------------------------------------------------
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
ln -sfn "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" "${CONDA_PREFIX}/bin/gcc"
ln -sfn "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" "${CONDA_PREFIX}/bin/g++"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# CUDA toolkit check -- nvcc is required by BOTH detectron2 and the MSDeformAttn
# op below. It comes from a *system* install, not from conda/pip.
# ---------------------------------------------------------------------------
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at ${CUDA_HOME}/bin/nvcc"
    echo "       Install a CUDA 11.8 toolkit or set CUDA_HOME to the right path."
    exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
echo ">> using CUDA toolkit: ${CUDA_HOME} ($("${CUDA_HOME}/bin/nvcc" --version | tail -n1))"

# ---------------------------------------------------------------------------
# detectron2 -- required by all three projects. Upstream repo (not on PyPI),
# installed editable so the sibling projects import it as-is.
# ---------------------------------------------------------------------------
if [ ! -d "${DETECTRON2_DIR}" ]; then
    git clone https://github.com/facebookresearch/detectron2.git "${DETECTRON2_DIR}"
fi
# Upstream caps iopath at <0.1.10 while the env ships iopath 0.1.10, which
# makes `pip install -e .` fail during dependency resolution. Relax the bound.
sed -i 's/"iopath>=0\.1\.7,<0\.1\.10"/"iopath>=0.1.7"/' "${DETECTRON2_DIR}/setup.py"
if ! grep -q '"iopath>=0.1.7",' "${DETECTRON2_DIR}/setup.py"; then
    echo "ERROR: could not relax the iopath cap in ${DETECTRON2_DIR}/setup.py"
    echo "       Edit the 'iopath>=0.1.7,<0.1.10' line manually and re-run."
    exit 1
fi
pip install -e "${DETECTRON2_DIR}"

# ---------------------------------------------------------------------------
# MSDeformAttn CUDA kernel (MaskDINO pixel decoder) -> MultiScaleDeformableAttention
# ---------------------------------------------------------------------------
( cd "${SCRIPT_DIR}/maskdino/modeling/pixel_decoder/ops" && sh make.sh )

# ---------------------------------------------------------------------------
# Optional: sam3 (not imported by maskdino/intraoral/dentmag)
# ---------------------------------------------------------------------------
if [ "${SKIP_SAM3:-0}" != "1" ]; then
    pip install --no-build-isolation git+https://github.com/facebookresearch/sam3.git
else
    echo ">> skipping sam3 (SKIP_SAM3=1)"
fi

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
