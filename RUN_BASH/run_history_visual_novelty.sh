#!/usr/bin/env bash
# Offline, task-balanced visual novelty pilot. It does not alter normal evaluation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_history_novelty_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_history_novelty_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_visual_novelty/${DATASET}}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

# Small, deterministic task-balanced pilot. Override CUDA_VISIBLE_DEVICES externally.
EXTRA_ARGS=()
if [[ -n "${SAMPLE_IDS:-}" ]]; then
  EXTRA_ARGS+=(--sample-ids "$SAMPLE_IDS")
fi
"$PYTHON_BIN" utils/history_visual_novelty.py \
  --data "$DATASET" --model "$MODEL_PATH" --work-dir "$WORK_DIR" \
  --num-samples "${NUM_SAMPLES:-50}" --sample-by "${SAMPLE_BY:-task}" \
  --samples-per-task "${SAMPLES_PER_TASK:-1}" --history-steps "${HISTORY_STEPS:-4}" \
  --max-pairs-per-sample "${MAX_PAIRS_PER_SAMPLE:-0}" --seed "${SEED:-42}" \
  "${EXTRA_ARGS[@]}"
