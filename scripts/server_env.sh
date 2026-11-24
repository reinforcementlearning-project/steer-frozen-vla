# source scripts/server_env.sh [GPU index]; does not install or change existing environments.
RF_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RF_BASE="${RF_BASE:-$HOME/rtx3090-migration/sy}"
export PY="${RF_PYTHON:-$RF_BASE/.runtime/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12}"
export PYTHONPATH="$RF_PROJECT:$HOME/rf/python-deps:$RF_BASE/lerobot/src:$RF_BASE/.venv/lib/python3.12/site-packages"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HOME="$HOME/rf/.cache/huggingface" XDG_CACHE_HOME="$HOME/rf/.cache"
export TORCH_HOME="$HOME/rf/.cache/torch" NUMBA_CACHE_DIR="$HOME/rf/.cache/numba"
export LIBERO_CONFIG_PATH="$RF_PROJECT/configs/libero"
export CUDA_VISIBLE_DEVICES="${1:-0}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
