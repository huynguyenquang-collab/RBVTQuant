#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python lm_eval_smoke.py \
  --model-path "${MODEL_PATH:-sshleifer/tiny-gpt2}" \
  --device "${DEVICE:-cpu}" \
  --task "${TASK:-piqa}" \
  --limit "${LIMIT:-5}"
