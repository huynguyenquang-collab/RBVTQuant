#!/usr/bin/env bash
set -euo pipefail

# Install the Python 3.12 environment used by the LeanQuant server runner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# These values are intentionally fixed for the supported server driver (535).
# Do not inherit stale PYTORCH_* variables from the login shell.
PYTORCH_VERSION="2.12.0"
TORCHVISION_VERSION="0.27.0"
PYTORCH_CUDA_RUNTIME="12.6"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"

export UV_CACHE_DIR
export PIP_CACHE_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

echo "=== RBVTQuant server setup ==="
echo "Repository: $ROOT_DIR"
echo "Repository commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python: $PYTHON_VERSION"
echo "PyTorch: $PYTORCH_VERSION"
echo "Torchvision: $TORCHVISION_VERSION"
echo "PyTorch CUDA runtime: $PYTORCH_CUDA_RUNTIME"
echo "PyTorch index: $PYTORCH_INDEX_URL"
echo "Virtual environment: $VENV_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi is unavailable. A Linux server with an NVIDIA GPU is required." >&2
  exit 1
fi
nvidia-smi

if [ -f .gitmodules ]; then
  echo "Initializing upstream submodules ..."
  git submodule update --init --recursive
fi

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "Installing uv ..."
  python3 -m pip install --user --upgrade uv
  UV_BIN="$(python3 -m site --user-base)/bin/uv"
fi

if [ ! -x "$UV_BIN" ]; then
  echo "Error: uv was not found after installation." >&2
  exit 1
fi

"$UV_BIN" python install "$PYTHON_VERSION"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

TORCH_MATCHES=0
if "$VENV_DIR/bin/python" - \
  "$PYTORCH_VERSION" \
  "$PYTORCH_CUDA_RUNTIME" \
  "$TORCHVISION_VERSION" <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError:
    raise SystemExit(1)

expected_version, expected_cuda, expected_torchvision = sys.argv[1:]
installed_version = torch.__version__.split("+", 1)[0]
installed_torchvision = torchvision.__version__.split("+", 1)[0]
raise SystemExit(
    0
    if (
        installed_version == expected_version
        and torch.version.cuda == expected_cuda
        and installed_torchvision == expected_torchvision
    )
    else 1
)
PY
then
  TORCH_MATCHES=1
fi

if [ "$TORCH_MATCHES" = "1" ]; then
  echo "Reusing compatible PyTorch installation."
else
  echo "Installing CUDA-compatible PyTorch after all other dependencies ..."
  "$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    --reinstall \
    "torch==$PYTORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url "$PYTORCH_INDEX_URL"
fi

echo "Installing requirements-server.txt and restoring pinned NumPy ..."
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --upgrade \
  --constraints constraints-server.txt \
  -r requirements-server.txt

"$VENV_DIR/bin/python" - <<'PY'
import sys

import numpy as np
import torch
import torchvision

print("Python:", sys.version.replace("\n", " "))
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Python 3.12 is required")
if np.__version__ != "1.26.4":
    raise SystemExit(f"NumPy 1.26.4 is required, got {np.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(
        "The installed PyTorch build cannot access CUDA. Check that its CUDA "
        "runtime is not newer than the maximum CUDA version shown by nvidia-smi."
    )
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GiB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
)

# Exercise the CUDA operations used by Hessian collection and LeanQuant.
device = torch.device("cuda:0")
matrix = torch.randn(256, 256, device=device, dtype=torch.float32)
gram = matrix @ matrix.t()
positive_definite = gram + torch.eye(256, device=device) * 1e-3
factor = torch.linalg.cholesky(positive_definite)
if not torch.isfinite(factor).all():
    raise SystemExit("CUDA Cholesky smoke test produced non-finite values")
torch.cuda.synchronize()
print("CUDA GEMM/Cholesky smoke test: OK")
PY

echo "Testing upstream SqueezeLLM KMeans ..."
"$VENV_DIR/bin/python" - <<'PY'
import numpy as np

from quantizers.upstream_imports import load_squeezellm_kmeans

kmeans_fit = load_squeezellm_kmeans()
values = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(-1, 1)
weights = np.ones(32, dtype=np.float32)
centers, labels = kmeans_fit((values, weights, 8))
if centers.shape != (8,) or labels.shape != (32,):
    raise SystemExit(
        f"Unexpected SqueezeLLM KMeans output: {centers.shape}, {labels.shape}"
    )
print("SqueezeLLM KMeans smoke test: OK")
PY

echo "Testing lm-eval PIQA dataset compatibility ..."
"$VENV_DIR/bin/python" lm_eval_dataset_smoke.py --download-piqa

echo "Server setup complete."
