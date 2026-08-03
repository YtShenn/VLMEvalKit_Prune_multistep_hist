#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

export GUI_ODYSSEY_ROOT="${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export GUI_ODYSSEY_DEBUG_HISTORY_PROMPT=0 #"${GUI_ODYSSEY_DEBUG_HISTORY_PROMPT:-0}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-0}"

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ROI_PRUNE_USE_CACHE="${QWEN3VL_ROI_PRUNE_USE_CACHE:-1}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE="${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE:-1}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-GUIOdyssey_high_task_split}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/torchrun}"
TS="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local tag="$1"
  local use_history="$2"
  local max_history="$3"
  local keep_system="$4"
  local port="$5"
  local work_dir="OUTPUT/ablation_gui_odyssey_${tag}_qwen3_5_0731"
  local log_file="run_output_${TS}_gui_odyssey_ablation_${tag}_qwen3_5.log"

  export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${use_history}"
  export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${max_history}"
  export GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT="${keep_system}"

  mkdir -p "${work_dir}/${DATASET}"
  {
    echo "[Ablation] tag=${tag}"
    echo "[Ablation] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
    echo "[Ablation] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "[Ablation] model=${MODEL}"
    echo "[Ablation] dataset=${DATASET}"
    echo "[Ablation] work_dir=${work_dir}"
    echo "[Ablation] use_history=${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS}"
    echo "[Ablation] max_history=${GUI_ODYSSEY_MAX_HISTORY_IMAGES}"
    echo "[Ablation] keep_system_prompt=${GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT}"
    echo "[Ablation] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
    echo "[Ablation] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
    echo "[Ablation] sample_seed=${VLM_EVAL_SAMPLE_SEED}"
  } | tee -a "${log_file}"

  "${TORCHRUN_BIN}" --standalone --nproc_per_node=1 run.py \
    --data "${DATASET}" \
    --model "${MODEL}" \
    --work-dir "${work_dir}/${DATASET}" \
    --mode all \
    --reuse 2>&1 | tee -a "${log_file}"
}

run_one nohist_fixed 0 0 0 29620
# run_one hist1_original_prompt 1 1 0 29621
# run_one hist4_keep_system_prompt 1 4 1 29622
