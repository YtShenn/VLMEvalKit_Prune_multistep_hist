#!/usr/bin/env bash
set -euo pipefail

# Isolated ST-Lite Qwen3-VL entry point. It does not enable state packet,
# structured fast decode, GUI-KV, or existing pruning knobs.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export SEED="${SEED:-42}"
export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"
# GUI action responses are short. This also prevents a malformed/non-EOS
# response from consuming the model config's generic 16384-token budget.
export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-256}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct-STLite}"
DATA="${DATA:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_STLITE/outputs_android_stlite_4B_ratio35_official_v4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# Examples:
# DATA="GUIOdyssey_high_random_split" WORK_DIR="OUTPUT_STLITE/outputs_guiodyssey_stlite_4B_ratio20"
# DATA="Mind2Web_test_task" WORK_DIR="OUTPUT_STLITE/outputs_mind2web_stlite_4B_ratio20"
# DATA="AITW_all" WORK_DIR="OUTPUT_STLITE/outputs_aitw_stlite_4B_ratio20"

export ST_LITE_HISTORY_STEPS="${ST_LITE_HISTORY_STEPS:-4}"
export ST_LITE_KEEP_RATIO="${ST_LITE_KEEP_RATIO:-0.35}"
export ST_LITE_WINDOW_SIZE="${ST_LITE_WINDOW_SIZE:-8}"
export ST_LITE_ALPHA="${ST_LITE_ALPHA:-0.1}"
export ST_LITE_CSS_KERNEL_SIZE="${ST_LITE_CSS_KERNEL_SIZE:-3}"
export ST_LITE_USE_TSG="${ST_LITE_USE_TSG:-1}"
export ST_LITE_USE_CSS="${ST_LITE_USE_CSS:-1}"
export ST_LITE_MIN_TOKENS="${ST_LITE_MIN_TOKENS:-64}"
export ST_LITE_TSG_THRESHOLD="${ST_LITE_TSG_THRESHOLD:-0.95}"

export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-${ST_LITE_HISTORY_STEPS}}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-${ST_LITE_HISTORY_STEPS}}"
export AITW_HIS_NUM="${AITW_HIS_NUM:-${ST_LITE_HISTORY_STEPS}}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-${ST_LITE_HISTORY_STEPS}}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

# Sample complete trajectories by task, matching the GUIKV runner. The
# selected task keeps all of its available steps, which is important for
# constructing ST-Lite history context.
export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-task}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "[Run] MODEL=${MODEL} DATA=${DATA} WORK_DIR=${WORK_DIR}"
echo "[ST_LITE] history_steps=${ST_LITE_HISTORY_STEPS} keep_ratio=${ST_LITE_KEEP_RATIO} window=${ST_LITE_WINDOW_SIZE} alpha=${ST_LITE_ALPHA} css=${ST_LITE_USE_CSS} tsg=${ST_LITE_USE_TSG} max_capacity=${ST_LITE_MAX_CAPACITY:-ratio_mode}"
echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE} sample_tasks=${VLM_EVAL_SAMPLE_TASKS} sample_count=${VLM_EVAL_SAMPLE_COUNT} sample_seed=${VLM_EVAL_SAMPLE_SEED}"
echo "[Run] max_new_tokens=${QWEN3VL_MAX_NEW_TOKENS} runtime_tracking=${QWEN3VL_RUNTIME_TRACKING} flops_profile=${QWEN3VL_PROFILE_FLOPS}"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  "${PYTHON_BIN}" run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    2>&1 | tee -a "run_output_${TIMESTAMP}_qwen3_vl_stlite_${DATA}.log"
else
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    2>&1 | tee -a "run_output_${TIMESTAMP}_qwen3_vl_stlite_${DATA}.log"
fi
