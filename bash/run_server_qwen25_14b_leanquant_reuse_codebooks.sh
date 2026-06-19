#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-14B LeanQuant benchmark that reuses precomputed codebooks.
#
# This wrapper intentionally refuses to run if requested LeanQuant codebooks are
# missing, so codebook_benchmark.py cannot spend disk/time collecting Hessians or
# rebuilding upstream codebooks. It also keeps HuggingFace/eval caches outside of
# the repository by default, which is safer on Vast.ai images with small root or
# venv disks.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-14B}"
MODEL_SLUG="${MODEL_SLUG:-Qwen2p5-14B}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/rbvtquant_outputs/leanquant_qwen25_14b}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
CODEBOOK_IMPORT_ROOT="${CODEBOOK_IMPORT_ROOT:-/root/leanquant_qwen25_14b_codebooks}"

RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-$WORKSPACE_ROOT/rbvtquant_runtime_cache}"
HF_HOME="${HF_HOME:-$RUNTIME_CACHE_ROOT/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$RUNTIME_CACHE_ROOT/evaluation}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
SKIP_PERPLEXITY="${SKIP_PERPLEXITY:-0}"
FORCE_EVAL="${FORCE_EVAL:-1}"
REFRESH_DATASETS_CACHE="${REFRESH_DATASETS_CACHE:-0}"
CLEAN_STATISTICS_CACHE="${CLEAN_STATISTICS_CACHE:-0}"
RUN_SETUP="${RUN_SETUP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
USE_WANDB="${USE_WANDB:-1}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande}"

export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE
export TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p \
  "$OUTPUT_ROOT" \
  "$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR"

copy_codebook_if_needed() {
  local bits="$1"
  local name="leanquant_${bits}bit_direct_upstream"
  local dst="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/$name"
  local manifest="$dst/manifest.json"

  if [ -f "$manifest" ]; then
    return 0
  fi

  local src=""
  for candidate in \
    "$CODEBOOK_IMPORT_ROOT/$MODEL_SLUG/$name" \
    "$CODEBOOK_IMPORT_ROOT/$name" \
    "$WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$MODEL_SLUG/$name" \
    "$WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$name"
  do
    if [ -f "$candidate/manifest.json" ]; then
      src="$candidate"
      break
    fi
  done

  if [ -z "$src" ]; then
    echo "Missing precomputed LeanQuant ${bits}-bit codebook." >&2
    echo "Expected one of:" >&2
    echo "  $CODEBOOK_IMPORT_ROOT/$MODEL_SLUG/$name" >&2
    echo "  $CODEBOOK_IMPORT_ROOT/$name" >&2
    echo "  $WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$MODEL_SLUG/$name" >&2
    echo "  $WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$name" >&2
    return 1
  fi

  mkdir -p "$dst"
  rsync -a --ignore-existing "$src/" "$dst/"
}

verify_codebook_complete() {
  local bits="$1"
  local manifest="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/leanquant_${bits}bit_direct_upstream/manifest.json"
  python - "$manifest" "$bits" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
bits = sys.argv[2]
if not manifest.exists():
    raise SystemExit(f"Missing {bits}-bit manifest: {manifest}")
data = json.loads(manifest.read_text(encoding="utf-8"))
if not data.get("complete"):
    raise SystemExit(f"{bits}-bit codebook manifest is not complete: {manifest}")
pt_count = len(list(manifest.parent.glob("*.pt")))
print(f"Reusing LeanQuant {bits}-bit codebook: {manifest.parent} ({pt_count} tensors)")
PY
}

read -r -a BITS_ARRAY <<< "$BITS"
for bits in "${BITS_ARRAY[@]}"; do
  copy_codebook_if_needed "$bits"
  verify_codebook_complete "$bits"
done

MODEL="$MODEL" \
BITS="$BITS" \
METHODS="$METHODS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
STATISTICS_CACHE_DIR="$STATISTICS_CACHE_DIR" \
HF_HOME="$HF_HOME" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
HF_MODULES_CACHE="$HF_MODULES_CACHE" \
TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
EVAL_CACHE_DIR="$EVAL_CACHE_DIR" \
LM_EVAL_OUTPUT_DIR="$LM_EVAL_OUTPUT_DIR" \
INCLUDE_LM_EVAL="$INCLUDE_LM_EVAL" \
LM_EVAL_TASKS="$LM_EVAL_TASKS" \
SKIP_PERPLEXITY="$SKIP_PERPLEXITY" \
FORCE_EVAL="$FORCE_EVAL" \
REFRESH_DATASETS_CACHE="$REFRESH_DATASETS_CACHE" \
CLEAN_STATISTICS_CACHE="$CLEAN_STATISTICS_CACHE" \
RUN_SETUP="$RUN_SETUP" \
RUN_TESTS="$RUN_TESTS" \
RUN_PREFLIGHT="$RUN_PREFLIGHT" \
USE_WANDB="$USE_WANDB" \
bash bash/run_server_leanquant.sh
