#!/usr/bin/env bash
set -euo pipefail

# Install the Python 3.12 environment used by the LeanQuant server runner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
PYTORCH_CUDA_RUNTIME="${PYTORCH_CUDA_RUNTIME:-12.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"

export UV_CACHE_DIR
export PIP_CACHE_DIR

echo "=== RBVTQuant server setup ==="
echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON_VERSION"
echo "PyTorch: $PYTORCH_VERSION"
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
if "$VENV_DIR/bin/python" - "$PYTORCH_VERSION" "$PYTORCH_CUDA_RUNTIME" <<'PY'
import sys

try:
    import torch
except ImportError:
    raise SystemExit(1)

expected_version, expected_cuda = sys.argv[1:]
installed_version = torch.__version__.split("+", 1)[0]
raise SystemExit(
    0
    if installed_version == expected_version and torch.version.cuda == expected_cuda
    else 1
)
PY
then
  TORCH_MATCHES=1
fi

echo "Installing requirements-server.txt ..."
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --upgrade \
  --constraints constraints-server.txt \
  -r requirements-server.txt

if [ "$TORCH_MATCHES" = "1" ]; then
  echo "Reusing compatible PyTorch installation."
else
  echo "Installing CUDA-compatible PyTorch after all other dependencies ..."
  "$UV_BIN" pip install \
    --python "$VENV_DIR/bin/python" \
    --reinstall \
    "torch==$PYTORCH_VERSION" \
    --index-url "$PYTORCH_INDEX_URL"
fi

"$VENV_DIR/bin/python" - <<'PY'
import sys

import torch

print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Python 3.12 is required")
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
PY

echo "Server setup complete."
