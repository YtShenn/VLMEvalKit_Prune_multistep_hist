#!/usr/bin/env bash
set -euo pipefail

# ROI / top4 driven prune for GUIOdyssey.
# Default is explicit opt-in via env; this script enables the prune path.

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export CUDA_VISIBLE_DEVICES=5

NPROC_PER_NODE=1
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
WORK_DIR="${WORK_DIR:-OUTPUT/outputs_qwen3vl_gui_odyssey_4B_baseline_10task_prunebash_detailtime_node5_hist}"

export GUI_ODYSSEY_ROOT="${GUI_ODYSSEY_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS=1
export GUI_ODYSSEY_MAX_HISTORY_IMAGES=4
export GUI_ODYSSEY_DEBUG_HISTORY_PROMPT=1

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
export QWEN3VL_ROI_PRUNE_JSON="${QWEN3VL_ROI_PRUNE_JSON:-/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep/OUTPUT/outputs_qwen3vl_gui_odyssey_attn_top4_4B_first15/GUIOdyssey_high_task_split/per_sample.json}"
export QWEN3VL_ROI_PRUNE_LAYER_ORDER="${QWEN3VL_ROI_PRUNE_LAYER_ORDER:-16}"
export QWEN3VL_ROI_PRUNE_TOPK_KEEP="${QWEN3VL_ROI_PRUNE_TOPK_KEEP:-4}"
export QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_EVERY="${QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_EVERY:-8}"
export QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_OFFSET="${QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_OFFSET:-0}"
export QWEN3VL_ROI_PRUNE_DEBUG="${QWEN3VL_ROI_PRUNE_DEBUG:-0}"
export QWEN3VL_ROI_PRUNE_PRINT_LAYER_ATTN_TOKENS="${QWEN3VL_ROI_PRUNE_PRINT_LAYER_ATTN_TOKENS:-0}"
export QWEN3VL_ROI_PRUNE_ALLOW_NONCLICK="${QWEN3VL_ROI_PRUNE_ALLOW_NONCLICK:-0}"
export QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE=1
export VLM_EVAL_SAMPLE_MODE="task" #"${VLM_EVAL_SAMPLE_MODE:-off}"
export VLM_EVAL_SAMPLE_TASKS="10"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"
export QWEN3VL_ROI_PRUNE_USE_CACHE=1

if [[ -z "${QWEN3VL_ROI_PRUNE_JSON}" ]]; then
  echo "[ERROR] QWEN3VL_ROI_PRUNE_JSON is required."
  echo "Example:"
  echo "  QWEN3VL_ROI_PRUNE_JSON=/path/to/guiodyssey_roi.json bash $0"
  exit 1
fi

if [[ ! -f "${QWEN3VL_ROI_PRUNE_JSON}" ]]; then
  echo "[ERROR] ROI json not found: ${QWEN3VL_ROI_PRUNE_JSON}"
  exit 1
fi

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
echo "[Run] ROI_JSON=${QWEN3VL_ROI_PRUNE_JSON}"
echo "[Run] PRUNE_LAYER_ORDER=${QWEN3VL_ROI_PRUNE_LAYER_ORDER}"
echo "[Run] TOPK_KEEP=${QWEN3VL_ROI_PRUNE_TOPK_KEEP}"
echo "[Run] UNIFORM_KEEP_EVERY=${QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_EVERY}"
echo "[Run] UNIFORM_KEEP_OFFSET=${QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_OFFSET}"
echo "[Run] DEBUG=${QWEN3VL_ROI_PRUNE_DEBUG}"
echo "[Run] PRINT_PER_SAMPLE=${QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE}"
echo "[Run] EVAL_SAMPLE_MODE=${VLM_EVAL_SAMPLE_MODE}"
echo "[Run] EVAL_SAMPLE_TASKS=${VLM_EVAL_SAMPLE_TASKS}"
echo "[Run] EVAL_SAMPLE_SEED=${VLM_EVAL_SAMPLE_SEED}"

for ds in ${DS_LIST}; do
  mkdir -p "${WORK_DIR}/${ds}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=29512 run.py \
    --data "${ds}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}/${ds}" \
    --mode all \
    --reuse \
    2>&1 | tee -a "run_output_${TIMESTAMP}_gui_odyssey_baseline_10task_prunebash_${ds}_detailtime_node5_hist.log"
done
