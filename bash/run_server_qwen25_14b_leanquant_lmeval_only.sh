#!/usr/bin/env bash
set -euo pipefail

# Memory/disk-conscious Qwen2.5-14B LeanQuant eval runner.
# - Runs 4/3-bit x RTN/RBVT by default.
# - Runs lm-eval only, including MMLU and GSM8K; skips perplexity.
# - Refuses to rebuild missing LeanQuant codebooks by default.
# - Deletes old PPL caches (WikiText/C4/eval cache) before starting.
# - Runs one combo at a time and removes saved model weights after each summary.
# - Keeps HuggingFace model cache so Qwen weights are not re-downloaded each job.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen2.5-14B}"
MODEL_SLUG="${MODEL_SLUG:-Qwen2p5-14B}"
DEVICE="${DEVICE:-cuda:0}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/leanquant_server}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

CACHE_ROOT="${CACHE_ROOT:-/workspace/rbvtquant_runtime_cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$CACHE_ROOT/evaluation}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
LEANQUANT_EXPONENT="${LEANQUANT_EXPONENT:-4.0}"
LEANQUANT_PERCDAMP="${LEANQUANT_PERCDAMP:-0.1}"
KMEANS_SEED="${KMEANS_SEED:-0}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_SETUP="${RUN_SETUP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
FORCE_EVAL="${FORCE_EVAL:-1}"
REQUIRE_CODEBOOKS="${REQUIRE_CODEBOOKS:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-35}"

export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

if [ "$RUN_SETUP" = "1" ]; then
  VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" bash bash/setup_server_leanquant.sh
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run setup first or set PYTHON_BIN." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$STATISTICS_CACHE_DIR" "$LOG_DIR" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" "$LM_EVAL_OUTPUT_DIR"

read -r -a BITS_ARRAY <<< "$BITS"
read -r -a METHOD_ARRAY <<< "$METHODS"
read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"

free_gib() {
  df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

print_space() {
  echo "Disk: $(df -h / /workspace 2>/dev/null | tail -n +2 | tr '\n' ' ')"
  du -sh "$OUTPUT_ROOT" "$CACHE_ROOT" /root/.cache 2>/dev/null || true
}

cleanup_ppl_cache() {
  echo "Cleaning PPL/eval caches while preserving codebooks and HF model cache ..."
  rm -rf "$EVAL_CACHE_DIR"
  mkdir -p "$EVAL_CACHE_DIR"

  find "$HF_DATASETS_CACHE" -maxdepth 2 -type d \( \
    -iname '*wikitext*' -o \
    -iname '*c4*' -o \
    -iname '*allenai___c4*' -o \
    -iname '*Salesforce___wikitext*' \
  \) -prune -exec rm -rf {} + 2>/dev/null || true

  rm -rf "$HF_MODULES_CACHE/datasets_modules" 2>/dev/null || true
  rm -rf /root/.cache/huggingface /root/.cache/pip /root/.cache/torch 2>/dev/null || true
}

cleanup_stale_weight_outputs() {
  echo "Removing stale saved model weights from old incomplete runs ..."
  find "$OUTPUT_ROOT" -maxdepth 2 -type f \( \
    -name 'model.safetensors' -o \
    -name '*.safetensors' -o \
    -name 'pytorch_model*.bin' \
  \) -not -path "$STATISTICS_CACHE_DIR/codebooks/*" -delete 2>/dev/null || true
}

remove_run_weights_if_summarized() {
  local run_dir="$1"
  if [ -f "$run_dir/run_summary.json" ]; then
    find "$run_dir" -maxdepth 1 -type f \( \
      -name 'model.safetensors' -o \
      -name '*.safetensors' -o \
      -name 'pytorch_model*.bin' \
    \) -delete 2>/dev/null || true
  fi
}

verify_codebooks() {
  [ "$REQUIRE_CODEBOOKS" = "1" ] || return 0
  for bits in "${BITS_ARRAY[@]}"; do
    local manifest="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/leanquant_${bits}bit_direct_upstream/manifest.json"
    "$PYTHON_BIN" - "$manifest" "$bits" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
bits = sys.argv[2]
if not manifest.exists():
    raise SystemExit(f"Missing {bits}-bit codebook manifest: {manifest}")
data = json.loads(manifest.read_text(encoding="utf-8"))
if not data.get("complete"):
    raise SystemExit(f"Incomplete {bits}-bit codebook manifest: {manifest}")
print(f"Codebook OK: {bits}-bit -> {manifest.parent}")
PY
  done
}

run_one() {
  local bits="$1"
  local method="$2"
  local run_dir="$OUTPUT_ROOT/leanquant_${bits}bit_${method}"
  local free
  free="$(free_gib || echo 0)"
  if [ "${free:-0}" -lt "$MIN_FREE_GIB" ]; then
    echo "Error: only ${free}GiB free on /workspace; need at least ${MIN_FREE_GIB}GiB." >&2
    print_space
    exit 1
  fi

  echo
  echo "=== LeanQuant Qwen2.5-14B | ${bits}-bit ${method^^} | lm-eval only ==="
  print_space

  local args=(
    --model-path "$MODEL"
    --device "$DEVICE"
    --output-root "$OUTPUT_ROOT"
    --resume
    --codebooks leanquant
    --bits "$bits"
    --methods "$method"
    --calib-dataset c4
    --n-calib "$N_CALIB"
    --max-length "$MAX_LENGTH"
    --seed "$SEED"
    --group-size -1
    --row-chunk "$ROW_CHUNK"
    --leanquant-exponent "$LEANQUANT_EXPONENT"
    --leanquant-percdamp "$LEANQUANT_PERCDAMP"
    --kmeans-seed "$KMEANS_SEED"
    --statistics-cache-dir "$STATISTICS_CACHE_DIR"
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --gap-floor "$GAP_FLOOR"
    --skip-perplexity
    --include-lm-eval
    --lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}"
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
    --eval-cache-dir "$EVAL_CACHE_DIR"
  )

  if [ "$FORCE_EVAL" = "1" ]; then
    args+=(--force-eval)
  fi
  if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
    args+=(--lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT")
  fi
  if [ -n "$LM_EVAL_LIMIT" ]; then
    args+=(--lm-eval-limit "$LM_EVAL_LIMIT")
  fi
  if [ "$USE_WANDB" = "1" ]; then
    args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_ENTITY" ]; then
      args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  else
    args+=(--no-wandb)
  fi

  set +e
  "$PYTHON_BIN" codebook_benchmark.py "${args[@]}"
  local status=$?
  set -e

  remove_run_weights_if_summarized "$run_dir"
  print_space
  return "$status"
}

if [ "$RUN_TESTS" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" VENV_DIR="$VENV_DIR" LM_EVAL_TASKS="$LM_EVAL_TASKS" \
    bash bash/test_colab_codebooks.sh
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  "$PYTHON_BIN" - "$MODEL" "$DEVICE" <<'PY'
import sys

import torch
from transformers import AutoConfig

model, device = sys.argv[1:3]
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(f"{device} requested but CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("Model config:", AutoConfig.from_pretrained(model, trust_remote_code=True).model_type)
PY
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/qwen25_14b_lmeval_only_${TIMESTAMP}.log"

{
  echo "=== Qwen2.5-14B LeanQuant lm-eval-only runner ==="
  echo "Model: $MODEL"
  echo "Bits: $BITS"
  echo "Methods: $METHODS"
  echo "Tasks: $LM_EVAL_TASKS"
  echo "Output: $OUTPUT_ROOT"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
  echo "HF cache: $HF_HOME"
  echo "Require precomputed codebooks: $REQUIRE_CODEBOOKS"
} 2>&1 | tee -a "$LOG_FILE"

cleanup_ppl_cache 2>&1 | tee -a "$LOG_FILE"
cleanup_stale_weight_outputs 2>&1 | tee -a "$LOG_FILE"
verify_codebooks 2>&1 | tee -a "$LOG_FILE"

for bits in "${BITS_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    run_one "$bits" "$method" 2>&1 | tee -a "$LOG_FILE"
  done
done

echo "Done. Log: $LOG_FILE"
