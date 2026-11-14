#!/usr/bin/env bash
# Render raw frames plus patch/token similarity overlays for specified samples.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_history_similarity_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_history_similarity_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
SAMPLE_IDS="${SAMPLE_IDS:-6732,1937,1516,1464}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_similarity_samples}"
GPU_ID="${GPU_ID:-3}"

export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

echo "[history-similarity] GPU=$GPU_ID samples=$SAMPLE_IDS output=$WORK_DIR" >&2
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" utils/render_history_similarity_sample.py \
  --data "$DATASET" \
  --model "$MODEL_PATH" \
  --sample-ids "$SAMPLE_IDS" \
  --work-dir "$WORK_DIR" \
  --history-steps "${HISTORY_STEPS:-4}" \
  --local-scale "${LOCAL_SCALE:-4}"

echo "[history-similarity] complete: $WORK_DIR/figures" >&2
