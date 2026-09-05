#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

export MIND2WEB_DATA_ROOT="${MIND2WEB_DATA_ROOT:-/mnt/storage2/Datasets/Mind2Web}"
export MIND2WEB_ANN_ROOT="${MIND2WEB_ANN_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_annots}"
if [[ -z "${MIND2WEB_IMAGE_ROOT:-}" && -d "${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images" ]]; then
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_DATA_ROOT}/mind2web_images/ming2web_images"
else
  export MIND2WEB_IMAGE_ROOT="${MIND2WEB_IMAGE_ROOT:-${MIND2WEB_DATA_ROOT}/mind2web_images}"
fi
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"
export MIND2WEB_WITH_NO_HISTORY="${MIND2WEB_WITH_NO_HISTORY:-0}"
export MIND2WEB_STRICT_OUTPUT_PROMPT="${MIND2WEB_STRICT_OUTPUT_PROMPT:-1}"

export MIND2WEB_STATE_PACKET_ENABLE="${MIND2WEB_STATE_PACKET_ENABLE:-0}"
export MIND2WEB_STATE_PACKET_DEBUG="${MIND2WEB_STATE_PACKET_DEBUG:-0}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_TIMING_VERBOSE="${VLM_TIMING_VERBOSE:-1}"
export VLM_TIMING_SYNC="${VLM_TIMING_SYNC:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export VLM_STAGE_TIMING_DEVICE="${VLM_STAGE_TIMING_DEVICE:-auto}"
export VLM_STAGE_TIMING_SYNC="${VLM_STAGE_TIMING_SYNC:-0}"
export VLM_PRUNE_TIMING="${VLM_PRUNE_TIMING:-1}"
export SEED="${SEED:-42}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-128}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-0}"
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG:-1}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS:-4}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS:-24}"
export QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS="${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS:-32}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-5}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-Mind2Web_test_task}"
# Mind2Web_test_task     -> Cross-Task
# Mind2Web_test_website  -> Cross-Website
# Mind2Web_test_domain   -> Cross-Domain
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TAG="${TAG:-hist4_baseline}"
WORK_DIR="${WORK_DIR:-OUTPUT/mind2web_${TAG}_4B}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="run_output_${TS}_mind2web_${TAG}.log"

SPLIT="${DATASET#Mind2Web_test_}"
SPLIT="${SPLIT#Mind2Web_}"
ANN_FILE="${MIND2WEB_ANN_FILE:-${MIND2WEB_ANN_ROOT}/mind2web_data_test_${SPLIT}.json}"

if [[ ! -f "${ANN_FILE}" ]]; then
  echo "[ERROR] Missing annotation file: ${ANN_FILE}"
  echo "Set MIND2WEB_ANN_ROOT or MIND2WEB_ANN_FILE."
  exit 1
fi
if [[ ! -d "${MIND2WEB_IMAGE_ROOT}" ]]; then
  echo "[ERROR] Missing image root: ${MIND2WEB_IMAGE_ROOT}"
  exit 1
fi

mkdir -p "${WORK_DIR}/${DATASET}"
{
  echo "[Run] tag=${TAG}"
  echo "[Run] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
  echo "[Run] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "[Run] model=${MODEL}"
  echo "[Run] dataset=${DATASET}"
  echo "[Run] work_dir=${WORK_DIR}"
  echo "[Run] mind2web_ann_file=${ANN_FILE}"
  echo "[Run] mind2web_image_root=${MIND2WEB_IMAGE_ROOT}"
  echo "[Run] mind2web_his_num=${MIND2WEB_HIS_NUM}"
  echo "[Run] mind2web_with_no_history=${MIND2WEB_WITH_NO_HISTORY}"
  echo "[Run] mind2web_strict_output_prompt=${MIND2WEB_STRICT_OUTPUT_PROMPT}"
  echo "[Run] state_packet_enable=${MIND2WEB_STATE_PACKET_ENABLE}"
  echo "[Run] structured_fast_decode=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE}"
  echo "[Run] qwen3vl_max_new_tokens=${QWEN3VL_MAX_NEW_TOKENS}"
  echo "[Run] structured_fast_decode_mind2web_action_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS}"
  echo "[Run] structured_fast_decode_mind2web_coord_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS}"
  echo "[Run] structured_fast_decode_mind2web_value_max_tokens=${QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS}"
  echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE}"
  echo "[Run] sample_tasks=${VLM_EVAL_SAMPLE_TASKS}"
  echo "[Run] sample_count=${VLM_EVAL_SAMPLE_COUNT}"
  echo "[Run] sample_seed=${VLM_EVAL_SAMPLE_SEED}"
} | tee -a "${LOG_FILE}"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py \
  --data "${DATASET}" \
  --model "${MODEL}" \
  --work-dir "${WORK_DIR}/${DATASET}" \
  --mode all 2>&1 | tee -a "${LOG_FILE}"
