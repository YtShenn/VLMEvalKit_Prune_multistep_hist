#!/usr/bin/env bash
set -euo pipefail

# Template-prefill timing run for AndroidControl Curated.
# ROI prune is disabled by default; this script focuses on fixed-template partial prefill.

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export CUDA_VISIBLE_DEVICES=7

NPROC_PER_NODE=1
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
WORK_DIR="${WORK_DIR:-OUTPUT/outputs_qwen3vl_android_control_curated_4B_template_prefill_timing_sample_once_gen_0726}"

export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING=1
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-0}"

export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_USE_TIMING_MODEL=1
export QWEN3VL_PRINT_PER_SAMPLE=1
export QWEN3VL_ROI_PRUNE_JSON="OUTPUT/outputs_qwen3vl_android_control_attn_top4_4B_first15_ONLY_CLICK_LONGPRESS_node5/AndroidControl_Curated_High_Task_Improved/per_sample.json"
export QWEN3VL_ROI_PRUNE_USE_CACHE=1

# Force the intended template-prefill path for apples-to-apples timing with ROI prune.
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=1
export QWEN3VL_TEMPLATE_PREFILL_IMPL=single_generate # or legacy
export QWEN3VL_TEMPLATE_PREFILL_DATASETS=androidcontrol
export QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE=action_first_json
export QWEN3VL_TEMPLATE_PREFILL_DEBUG=0

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-200}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

if [[ ! -d "${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource" ]]; then
  echo "[ERROR] Missing benchmark_resource under ANDROID_CONTROL_CURATED_ROOT=${ANDROID_CONTROL_CURATED_ROOT}"
  exit 1
fi

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DS_LIST=(
  "${DS_LIST:-AndroidControl_Curated_High_Task_Improved}"
)

echo "[Run] MODEL=${MODEL} WORK_DIR=${WORK_DIR}"
echo "[Run] TEMPLATE_PREFILL=${QWEN3VL_ENABLE_TEMPLATE_PREFILL}"
echo "[Run] TEMPLATE_IMPL=${QWEN3VL_TEMPLATE_PREFILL_IMPL}"
echo "[Run] TEMPLATE_DATASETS=${QWEN3VL_TEMPLATE_PREFILL_DATASETS}"
echo "[Run] TEMPLATE_ANDROID_MODE=${QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE}"
echo "[Run] ROI_PRUNE_ENABLED=${QWEN3VL_ENABLE_ROI_PRUNE}"
echo "[Run] ROI_PRUNE_USE_CACHE=${QWEN3VL_ROI_PRUNE_USE_CACHE}"
echo "[Run] PRINT_PER_SAMPLE=${QWEN3VL_PRINT_PER_SAMPLE}"
echo "[Run] EVAL_SAMPLE_MODE=${VLM_EVAL_SAMPLE_MODE}"
echo "[Run] EVAL_SAMPLE_TASKS=${VLM_EVAL_SAMPLE_TASKS}"
echo "[Run] EVAL_SAMPLE_SEED=${VLM_EVAL_SAMPLE_SEED}"

for ds in ${DS_LIST}; do
  mkdir -p "${WORK_DIR}/${ds}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=29525 run.py \
    --data "${ds}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}/${ds}" \
    --mode all \
    --reuse \
    2>&1 | tee -a "run_output_${TIMESTAMP}_android_control_template_prefill_${ds}_sample_once_gen_0726.log"
done
