#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

export GUI_ODYSSEY_ROOT="${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export GUI_ODYSSEY_DEBUG_HISTORY_PROMPT="${GUI_ODYSSEY_DEBUG_HISTORY_PROMPT:-0}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ROI_PRUNE_USE_CACHE="${QWEN3VL_ROI_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE="${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE:-1}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT="${GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT:-1}"

export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export GUI_ODYSSEY_STATE_PACKET_ENABLE="1"
export GUI_ODYSSEY_STATE_PACKET_DEBUG="1"
export GUI_ODYSSEY_STATE_PACKET_CACHE_DIR="${GUI_ODYSSEY_STATE_PACKET_CACHE_DIR:-/tmp/gui_odyssey_state_packet_cache}"
export GUI_ODYSSEY_STATE_PACKET_PATCH_SIZE="${GUI_ODYSSEY_STATE_PACKET_PATCH_SIZE:-16}"
export GUI_ODYSSEY_STATE_PACKET_MERGE_SIZE="${GUI_ODYSSEY_STATE_PACKET_MERGE_SIZE:-2}"
# Keep the global thumbnail unchanged; make ROI slightly tighter but render it a bit larger.
export GUI_ODYSSEY_STATE_PACKET_THUMB_LONG_EDGE="${GUI_ODYSSEY_STATE_PACKET_THUMB_LONG_EDGE:-256}"
export GUI_ODYSSEY_STATE_PACKET_ROI_LONG_EDGE="${GUI_ODYSSEY_STATE_PACKET_ROI_LONG_EDGE:-288}"
export GUI_ODYSSEY_STATE_PACKET_ROI_SHORT_SIDE_RATIO="${GUI_ODYSSEY_STATE_PACKET_ROI_SHORT_SIDE_RATIO:-0.26}"
export GUI_ODYSSEY_STATE_PACKET_ROI_MIN_SIDE_PX="${GUI_ODYSSEY_STATE_PACKET_ROI_MIN_SIDE_PX:-224}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-GUIOdyssey_high_random_split}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-hist4_keep_system_prompt_state_packet_changeratio2}"
WORK_DIR="${WORK_DIR:-OUTPUT/ablation_gui_odyssey_${TAG}_qwen3_5_node5}"
LOG_FILE="run_output_${TS}_gui_odyssey_${TAG}.log"

mkdir -p "${WORK_DIR}/${DATASET}"
{
  echo "[Run] tag=${TAG}"
  echo "[Run] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
  echo "[Run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "[Run] model=${MODEL}"
  echo "[Run] dataset=${DATASET}"
  echo "[Run] work_dir=${WORK_DIR}"
  echo "[Run] use_history=${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS}"
  echo "[Run] max_history=${GUI_ODYSSEY_MAX_HISTORY_IMAGES}"
  echo "[Run] keep_system_prompt=${GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT}"
  echo "[Run] state_packet_enable=${GUI_ODYSSEY_STATE_PACKET_ENABLE}"
  echo "[Run] state_packet_debug=${GUI_ODYSSEY_STATE_PACKET_DEBUG}"
  echo "[Run] thumb_long_edge=${GUI_ODYSSEY_STATE_PACKET_THUMB_LONG_EDGE}"
  echo "[Run] roi_long_edge=${GUI_ODYSSEY_STATE_PACKET_ROI_LONG_EDGE}"
  echo "[Run] roi_short_side_ratio=${GUI_ODYSSEY_STATE_PACKET_ROI_SHORT_SIDE_RATIO}"
  echo "[Run] roi_min_side_px=${GUI_ODYSSEY_STATE_PACKET_ROI_MIN_SIDE_PX}"
  echo "[Run] flops_profile=${QWEN3VL_PROFILE_FLOPS}"
  echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
  echo "[Run] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
  echo "[Run] sample_seed=${VLM_EVAL_SAMPLE_SEED}"
} | tee -a "${LOG_FILE}"

"${TORCHRUN_BIN}" --standalone --nproc_per_node=1 run.py \
  --data "${DATASET}" \
  --model "${MODEL}" \
  --work-dir "${WORK_DIR}/${DATASET}" \
  --mode all \
  --reuse 2>&1 | tee -a "${LOG_FILE}"
