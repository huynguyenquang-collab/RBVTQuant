#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python || true)"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"

read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"

"$PYTHON_BIN" lm_eval_smoke.py \
  --model-path "${LM_EVAL_SMOKE_MODEL_PATH:-sshleifer/tiny-gpt2}" \
  --device "${LM_EVAL_SMOKE_DEVICE:-cuda:0}" \
  --tasks "${LM_EVAL_TASK_ARRAY[@]}" \
  --limit "${LM_EVAL_SMOKE_LIMIT:-1}" \
  --batch-size "${LM_EVAL_SMOKE_BATCH_SIZE:-1}" \
  --num-fewshot "${LM_EVAL_SMOKE_NUM_FEWSHOT:-0}"
