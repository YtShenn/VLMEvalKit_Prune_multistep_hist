#!/usr/bin/env bash
set -euo pipefail

# GUIPruner-reproduction (unofficial reproduction), four-dataset runner.
# Default runs are sequential, which keeps a single GPU's memory bounded.
# Examples:
#   bash RUN_BASH/run_qwen3vl_guipruner_4datasets_4B.sh
#   DATASETS='Mind2Web_test_task GUIOdyssey_high_task_split' CUDA_VISIBLE_DEVICES=0 \
#     bash RUN_BASH/run_qwen3vl_guipruner_4datasets_4B.sh
#   VLM_EVAL_SAMPLE_MODE=off bash RUN_BASH/run_qwen3vl_guipruner_4datasets_4B.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct-GUIPruner}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WORK_DIR="${WORK_DIR:-OUTPUT_GUIPRUNER/4B_hist4_lambda40_mu752_change_flops}"
# DATASETS="${DATASETS:-AITW_all Mind2Web_test_task GUIOdyssey_high_task_split AndroidControl_Curated_High_Task_Improved}"
DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"
# DATASETS="${DATASETS:-AITW_all}"
# DATASETS="${DATASETS:-AITW_all Mind2Web_test_task}"
# DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"

# Paper hyperparameters, mirrored in the registered model configuration.
export GUI_PRUNER_HISTORY_STEPS="${GUI_PRUNER_HISTORY_STEPS:-4}"
export GUI_PRUNER_HISTORY_KEEP_RATIO="${GUI_PRUNER_HISTORY_KEEP_RATIO:-0.02360}"
export GUI_PRUNER_TEMPORAL_DECAY="${GUI_PRUNER_TEMPORAL_DECAY:-0.350}"
export GUI_PRUNER_CURRENT_KEEP_RATIO="${GUI_PRUNER_CURRENT_KEEP_RATIO:-0.75}"
# Optional eta: history + current global retention. It is disabled by default,
# so lambda and mu remain independent paper-style budgets.  When explicitly
# set, eta derives the current SSP ratio and takes precedence over mu.
export GUI_PRUNER_OVERALL_KEEP_RATIO="${GUI_PRUNER_OVERALL_KEEP_RATIO-0.35}"
# Keep eta on feasible samples; otherwise saturate current-image SSP at 100%.
# Set GUI_PRUNER_GLOBAL_BUDGET_POLICY=strict for per-sample exact failure.
export GUI_PRUNER_GLOBAL_BUDGET_POLICY="${GUI_PRUNER_GLOBAL_BUDGET_POLICY:-clip}"
export GUI_PRUNER_BACKGROUND_SALIENCY="${GUI_PRUNER_BACKGROUND_SALIENCY:-0.30}"
export GUI_PRUNER_PRUNE_LAYER="${GUI_PRUNER_PRUNE_LAYER:-2}" # paper numbering, one-based

# Four history screenshots, using only the existing dataset prompt builders.
export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-${GUI_PRUNER_HISTORY_STEPS}}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-${GUI_PRUNER_HISTORY_STEPS}}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-${GUI_PRUNER_HISTORY_STEPS}}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-${GUI_PRUNER_HISTORY_STEPS}}"

# Explicitly disable every unrelated accelerator for an isolated comparison.
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export QWEN3VL_USE_ATTN_PRUNE_MODEL=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE=0
export AITW_STATE_PACKET_ENABLE=0
export MIND2WEB_STATE_PACKET_ENABLE=0
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

# Optional measurement; these do not claim paper-equivalent speed/FLOPs.
export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

# A small deterministic smoke subset by default. Set MODE=off for full data.
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

export AITW_ANN_ROOT="${AITW_ANN_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_annots}"
export AITW_IMAGE_ROOT="${AITW_IMAGE_ROOT:-/mnt/storage2/Datasets/aitw_data/aitw_images}"
export AITW_SPLIT="${AITW_SPLIT:-test}"
export AITW_WITH_NO_HISTORY="${AITW_WITH_NO_HISTORY:-0}"
export AITW_SEMANTIC_ACTION_PROMPT="${AITW_SEMANTIC_ACTION_PROMPT:-0}"

export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/mnt/storage2/Datasets/Mind2Web}"
export MIND2WEB_ANN_ROOT="${MIND2WEB_ANN_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_annots}"
if [[ -z "${MIND2WEB_IMAGE_ROOT:-}" && -d "${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images" ]]; then
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images"
else
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_IMAGE_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_images}"
fi
export MIND2WEB_WITH_NO_HISTORY="${MIND2WEB_WITH_NO_HISTORY:-0}"
export MIND2WEB_STRICT_OUTPUT_PROMPT="${MIND2WEB_STRICT_OUTPUT_PROMPT:-1}"

export DATA_ROOT="${DATA_ROOT:-/mnt/storage2/users/ytshen_data/GUIOdyssey}"
export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

require_dataset_files() {
  local dataset="$1"
  case "${dataset}" in
    AITW_*)
      [[ -f "${AITW_ANN_ROOT}/aitw_data_${AITW_SPLIT}.json" ]] || { echo "[ERROR] Missing AITW annotation."; return 1; }
      [[ -d "${AITW_IMAGE_ROOT}" ]] || { echo "[ERROR] Missing AITW image root."; return 1; }
      ;;
    Mind2Web_test_*)
      local split="${dataset#Mind2Web_test_}"
      [[ -f "${MIND2WEB_ANN_ROOT}/mind2web_data_test_${split}.json" ]] || { echo "[ERROR] Missing Mind2Web annotation for ${split}."; return 1; }
      [[ -d "${MIND2WEB_IMAGE_ROOT}" ]] || { echo "[ERROR] Missing Mind2Web image root."; return 1; }
      ;;
    GUIOdyssey_*)
      local split="${dataset#GUIOdyssey_}"
      [[ -d "${DATA_ROOT}/screenshots" ]] || { echo "[ERROR] Missing GUI-Odyssey screenshots."; return 1; }
      [[ -f "${DATA_ROOT}/test_anno/${split}.json" ]] || { echo "[ERROR] Missing GUI-Odyssey annotation: ${DATA_ROOT}/test_anno/${split}.json"; return 1; }
      ;;
    AndroidControl_Curated_*)
      [[ -d "${ANDROID_CONTROL_CURATED_ROOT}/benchmark_resource" ]] || { echo "[ERROR] Missing AndroidControl benchmark_resource."; return 1; }
      [[ -d "${ANDROID_CONTROL_CURATED_IMAGE_ROOT}" ]] || { echo "[ERROR] Missing AndroidControl image root."; return 1; }
      ;;
    *) echo "[ERROR] Unsupported dataset: ${dataset}"; return 1 ;;
  esac
}

mkdir -p "${WORK_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"


for dataset in ${DATASETS}; do
  require_dataset_files "${dataset}"
  output_dir="${WORK_DIR}/${dataset}"
  LOG_FILE="${WORK_DIR}/run_output_${STAMP}_${dataset}_guipruner.log"
  mkdir -p "${output_dir}"
  {
    echo "[GUIPruner-reproduction] unofficial reproduction"
    echo "[Run] model=${MODEL} dataset=${dataset} output=${output_dir}"
    echo "[Run] history_steps=${GUI_PRUNER_HISTORY_STEPS} lambda=${GUI_PRUNER_HISTORY_KEEP_RATIO} gamma=${GUI_PRUNER_TEMPORAL_DECAY} mu=${GUI_PRUNER_CURRENT_KEEP_RATIO} overall_eta=${GUI_PRUNER_OVERALL_KEEP_RATIO:-disabled} global_budget_policy=${GUI_PRUNER_GLOBAL_BUDGET_POLICY} rho=${GUI_PRUNER_BACKGROUND_SALIENCY} prune_layer_one_based=${GUI_PRUNER_PRUNE_LAYER}"
    echo "[Run] isolated_backbone: attn_prune=${QWEN3VL_ENABLE_ATTN_PRUNE} attn_prune_model=${QWEN3VL_USE_ATTN_PRUNE_MODEL} roi_prune=${QWEN3VL_ENABLE_ROI_PRUNE}"
    echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE} sample_tasks=${VLM_EVAL_SAMPLE_TASKS} sample_count=${VLM_EVAL_SAMPLE_COUNT}"
  } | tee -a "${LOG_FILE}"
  if [[ "${NPROC_PER_NODE}" == "1" ]]; then
    "${PYTHON_BIN}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${LOG_FILE}"
  else
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py --data "${dataset}" --model "${MODEL}" --work-dir "${output_dir}" --mode all 2>&1 | tee -a "${LOG_FILE}"
  fi
done
