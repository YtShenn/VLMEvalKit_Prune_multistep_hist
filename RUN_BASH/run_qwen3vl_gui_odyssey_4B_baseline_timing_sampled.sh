#!/usr/bin/env bash
set -euo pipefail
# 有问题：没显示开kv cache，统计出来的时间大概率不对

# Baseline timing run for GUIOdyssey.
# No ROI pruning is applied. We only enable the timing-capable Qwen3-VL wrapper
# so we can record encode / prefill / decode stats and write summary.json.

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export CUDA_VISIBLE_DEVICES=4,5

NPROC_PER_NODE=2
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
WORK_DIR="${WORK_DIR:-OUTPUT/outputs_qwen3vl_gui_odyssey_4B_baseline_timing_node3}"

export GUI_ODYSSEY_ROOT="${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-0}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-0}"

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_USE_TIMING_MODEL=1
export QWEN3VL_PRINT_PER_SAMPLE="${QWEN3VL_PRINT_PER_SAMPLE:-1}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-200}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

if [[ ! -d "${GUI_ODYSSEY_ROOT}/test_anno" ]]; then
  echo "[ERROR] Missing test_anno under GUI_ODYSSEY_ROOT=${GUI_ODYSSEY_ROOT}"
  exit 1
fi

if [[ ! -d "${GUI_ODYSSEY_ROOT}/screenshots" ]]; then
  echo "[ERROR] Missing screenshots under GUI_ODYSSEY_ROOT=${GUI_ODYSSEY_ROOT}"
  exit 1
fi

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DS_LIST=(
  "${DS_LIST:-GUIOdyssey_high_task_split}"
)

echo "[Run] MODEL=${MODEL} WORK_DIR=${WORK_DIR}"
echo "[Run] TIMING_MODEL=${QWEN3VL_USE_TIMING_MODEL}"
echo "[Run] ROI_PRUNE_ENABLED=${QWEN3VL_ENABLE_ROI_PRUNE}"
echo "[Run] PRINT_PER_SAMPLE=${QWEN3VL_PRINT_PER_SAMPLE}"
echo "[Run] PER_SAMPLE_LOG=GenerateSample includes prompt_seq_tokens and decode_tokens"
echo "[Run] EVAL_SAMPLE_MODE=${VLM_EVAL_SAMPLE_MODE}"
echo "[Run] EVAL_SAMPLE_TASKS=${VLM_EVAL_SAMPLE_TASKS}"
echo "[Run] EVAL_SAMPLE_SEED=${VLM_EVAL_SAMPLE_SEED}"

for ds in ${DS_LIST}; do
  mkdir -p "${WORK_DIR}/${ds}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=29515 run.py \
    --data "${ds}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}/${ds}" \
    --mode all \
    --reuse \
    2>&1 | tee -a "run_output_${TIMESTAMP}_gui_odyssey_baseline_timing_${ds}_node3.log"
done
