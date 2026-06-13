#!/usr/bin/env bash
set -euo pipefail

# Fast adapter and CLI checks. This does not download or quantize Llama-3.1-8B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_colab_codebooks.sh" >&2
  exit 1
fi

echo "=== Codebook adapter tests ==="
"$PYTHON_BIN" -m unittest -v test_codebook_adapters.py

echo
echo "=== Benchmark CLI check ==="
"$PYTHON_BIN" codebook_benchmark.py --help >/dev/null

echo "Smoke tests passed."
