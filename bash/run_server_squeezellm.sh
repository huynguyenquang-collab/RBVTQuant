#!/usr/bin/env bash
set -euo pipefail

# Resumable SqueezeLLM benchmark for a Linux GPU server:
# SqueezeLLM dense+sparse+sensitive x 3/4-bit x RTN/RBVT on Llama-3.1-8B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
RUN_SETUP="${RUN_SETUP:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"
SQUEEZELLM_MODE="${SQUEEZELLM_MODE:-hybrid}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/squeezellm_server}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$CACHE_ROOT/evaluation}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
KMEANS_SEED="${KMEANS_SEED:-0}"
SQUEEZELLM_SENSITIVITY="${SQUEEZELLM_SENSITIVITY:-}"
SQUEEZELLM_FISHER_SAMPLES="${SQUEEZELLM_FISHER_SAMPLES:-100}"
SQUEEZELLM_FISHER_LENGTH="${SQUEEZELLM_FISHER_LENGTH:-512}"
SQUEEZELLM_OUTLIER_RANGE="${SQUEEZELLM_OUTLIER_RANGE:-1.8}"
SQUEEZELLM_SENSITIVE_PERCENT="${SQUEEZELLM_SENSITIVE_PERCENT:-0.05}"

# Fixed by request so all SqueezeLLM RBVT rows use the same setting.
RBVT_LAMBDA="1.0"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
KEEP_MODEL="${KEEP_MODEL:-0}"

MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-30}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"

export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# SqueezeLLM runs row-wise sklearn KMeans in multiprocessing workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

if [ "$RUN_SETUP" = "1" ]; then
  VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" \
    bash bash/setup_server_leanquant.sh
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_server_leanquant.sh" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

if [ "$RUN_TESTS" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" VENV_DIR="$VENV_DIR" \
    bash bash/test_colab_codebooks.sh
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  "$PYTHON_BIN" - "$MODEL" "$MIN_GPU_MEMORY_GIB" "$ALLOW_LOW_VRAM" "$SQUEEZELLM_MODE" <<'PY'
import sys

import torch
from transformers import AutoConfig

model = sys.argv[1]
required_gib = float(sys.argv[2])
allow_low_vram = sys.argv[3] == "1"
mode = sys.argv[4]

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 is required, got {sys.version_info.major}.{sys.version_info.minor}"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

available_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("GPU:", torch.cuda.get_device_name(0))
print(f"GPU memory: {available_gib:.2f} GiB")
if available_gib < required_gib:
    message = (
        f"GPU memory is {available_gib:.2f} GiB; the default workflow expects "
        f"about {required_gib:.0f} GiB."
    )
    if allow_low_vram:
        print("Warning:", message)
    else:
        raise SystemExit(message + " Set ALLOW_LOW_VRAM=1 to bypass this check.")

config = AutoConfig.from_pretrained(model, trust_remote_code=True)
print("Model config:", config.model_type)

from quantizers.upstream_imports import (
    load_squeezellm_gradients,
    load_squeezellm_kmeans,
    load_squeezellm_model_parse,
)

get_loaders, get_modules, square_grad_hook = load_squeezellm_gradients()
kmeans_fit = load_squeezellm_kmeans()
model_parse = load_squeezellm_model_parse()
print("SqueezeLLM KMeans:", kmeans_fit.__module__)
print("SqueezeLLM model parser:", model_parse.__name__)
print("SqueezeLLM gradients loader:", get_loaders.__module__)
print("SqueezeLLM gradients modules:", get_modules.__upstream_source__)
print("SqueezeLLM gradients hook:", square_grad_hook.__upstream_source__)
if mode == "hybrid":
    from quantizers.upstream_imports import load_squeezellm_remove_outliers

    remove_outliers = load_squeezellm_remove_outliers()
    print("SqueezeLLM sparse extractor:", remove_outliers.__module__)
PY
fi

mkdir -p \
  "$OUTPUT_ROOT" \
  "$STATISTICS_CACHE_DIR" \
  "$LOG_DIR" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR"

read -r -a BITS_ARRAY <<< "$BITS"
read -r -a METHOD_ARRAY <<< "$METHODS"

COMMON_ARGS=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --resume
  --calib-dataset c4
  --n-calib "$N_CALIB"
  --max-length "$MAX_LENGTH"
  --seed "$SEED"
  --group-size -1
  --row-chunk "$ROW_CHUNK"
  --kmeans-seed "$KMEANS_SEED"
  --squeezellm-fisher-samples "$SQUEEZELLM_FISHER_SAMPLES"
  --squeezellm-fisher-length "$SQUEEZELLM_FISHER_LENGTH"
  --squeezellm-mode "$SQUEEZELLM_MODE"
  --squeezellm-outlier-range "$SQUEEZELLM_OUTLIER_RANGE"
  --squeezellm-sensitive-percent "$SQUEEZELLM_SENSITIVE_PERCENT"
  --statistics-cache-dir "$STATISTICS_CACHE_DIR"
  --rbvt-lambda "$RBVT_LAMBDA"
  --rbvt-topk "$RBVT_TOPK"
  --gap-floor "$GAP_FLOOR"
  --eval-stride "$EVAL_STRIDE"
  --eval-max-length "$EVAL_MAX_LENGTH"
  --eval-samples "$EVAL_SAMPLES"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
  --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
)

if [ -n "$SQUEEZELLM_SENSITIVITY" ]; then
  COMMON_ARGS+=(--squeezellm-sensitivity "$SQUEEZELLM_SENSITIVITY")
fi
if [ "$INCLUDE_LM_EVAL" = "1" ]; then
  COMMON_ARGS+=(--include-lm-eval)
else
  COMMON_ARGS+=(--no-lm-eval)
fi
if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
  COMMON_ARGS+=(--lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT")
fi
if [ -n "$LM_EVAL_LIMIT" ]; then
  COMMON_ARGS+=(--lm-eval-limit "$LM_EVAL_LIMIT")
fi
if [ "$KEEP_MODEL" = "1" ]; then
  COMMON_ARGS+=(--keep-model)
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/squeezellm_${TIMESTAMP}.log"

{
  echo "=== RBVTQuant SqueezeLLM server benchmark ==="
  echo "Repository: $ROOT_DIR"
  echo "Model: $MODEL"
  echo "Device: $DEVICE"
  echo "Bits: $BITS"
  echo "Methods: $METHODS"
  echo "SqueezeLLM mode: $SQUEEZELLM_MODE"
  echo "SqueezeLLM Fisher: C4/${SQUEEZELLM_FISHER_SAMPLES}x${SQUEEZELLM_FISHER_LENGTH}"
  echo "SqueezeLLM outlier range: $SQUEEZELLM_OUTLIER_RANGE"
  echo "SqueezeLLM sensitive values: $SQUEEZELLM_SENSITIVE_PERCENT%"
  echo "RBVT lambda: $RBVT_LAMBDA"
  echo "Worker threads: OMP=$OMP_NUM_THREADS | MKL=$MKL_NUM_THREADS"
  echo "RBVT calibration: C4/${N_CALIB}x${MAX_LENGTH}"
  echo "Output: $OUTPUT_ROOT"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
  nvidia-smi
} 2>&1 | tee -a "$LOG_FILE"

run_index=0
total_runs=$((${#BITS_ARRAY[@]} * ${#METHOD_ARRAY[@]}))
for bits in "${BITS_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    run_index=$((run_index + 1))
    {
      echo
      echo "=== Job ${run_index}/${total_runs}: SqueezeLLM ${bits}-bit ${method} ==="
      "$PYTHON_BIN" codebook_benchmark.py \
        "${COMMON_ARGS[@]}" \
        --codebooks squeezellm \
        --bits "$bits" \
        --methods "$method"
    } 2>&1 | tee -a "$LOG_FILE"
  done
done

{
  echo
  echo "=== Building combined SqueezeLLM report ==="
  "$PYTHON_BIN" codebook_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --codebooks squeezellm \
    --bits "${BITS_ARRAY[@]}" \
    --methods "${METHOD_ARRAY[@]}"
} 2>&1 | tee -a "$LOG_FILE"

echo "Benchmark complete."
echo "Results: $OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
