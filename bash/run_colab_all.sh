#!/usr/bin/env bash
set -euo pipefail

# One-command Colab entrypoint after cloning the repository.
#
# Required before running:
# - select a GPU runtime;
# - mount Google Drive when USE_DRIVE=1;
# - export HF_TOKEN for the gated Llama-3.1-8B model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SETUP="${RUN_SETUP:-1}"
RUN_TESTS="${RUN_TESTS:-1}"

echo "=== RBVTQuant Colab end-to-end runner ==="
echo "Repository: $ROOT_DIR"

if [ "$RUN_SETUP" = "1" ]; then
  bash bash/setup_colab_codebooks.sh
fi

if [ "$RUN_TESTS" = "1" ]; then
  bash bash/test_colab_codebooks.sh
fi

# run_colab_codebooks.sh performs preflight checks by default.
bash bash/run_colab_codebooks.sh
