#!/usr/bin/env bash
# Full-resolution, teacher-forced history-attention observation study.
# Override MODEL_PATH or AndroidControl roots only when your local layout differs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Ensure this checkout wins over any globally exported VLMEvalKit checkout.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The system Python on this host has a NumPy/Pandas ABI mismatch.  Keep this
# overrideable for other machines, but default to the project evaluation env.
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_history_attention_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_history_attention_cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
# Helpful for the remaining normal SDPA activations on long full-history inputs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_attention_test/${DATASET}}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
HISTORY_STEPS="${HISTORY_STEPS:-4}"
ROI_JSON="${ROI_JSON:-}"
SEED="${SEED:-42}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
SAVE_ATTENTION_TENSORS="${SAVE_ATTENTION_TENSORS:-0}"

# AndroidControl-Curated's normal full-history benchmark inputs.
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
# Explicitly prevent optional state-packet thumbnail/crop inputs: use full history.
export GUI_ODYSSEY_STATE_PACKET_ENABLE=0
export ANDROID_CONTROL_STATE_PACKET_ENABLE=0
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS=1
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="$HISTORY_STEPS"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CMD=("$PYTHON_BIN" utils/history_attention_diagnostic.py
  --data "$DATASET"
  --model "$MODEL_PATH"
  --work-dir "$WORK_DIR"
  --num-samples "$NUM_SAMPLES"
  --history-steps "$HISTORY_STEPS"
  --sample-by task
  --samples-per-task 1
  --layers last_third
  --action-token-mode all
  --random-roi-samples 100
  --save-attention-tensors "$SAVE_ATTENTION_TENSORS"
  --seed "$SEED")

if [[ -n "$ROI_JSON" ]]; then CMD+=(--roi-json "$ROI_JSON"); fi
if [[ -n "$SAMPLE_IDS" ]]; then CMD+=(--sample-ids "$SAMPLE_IDS"); fi
"${CMD[@]}"
