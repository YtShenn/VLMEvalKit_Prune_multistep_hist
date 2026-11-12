#!/usr/bin/env bash
set -euo pipefail

# GUI-KV Qwen3-VL entry point. Existing Qwen3-VL/statepacket/fastdecode/pruning
# scripts are intentionally left untouched.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export SEED="${SEED:-42}"
export QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE="${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE:-1}"
# Use the same SFD execution protocol as PACT-GUI: one full multimodal
# prefill, followed by fixed-template KV-cache continuations and decoding
# only the dynamic slots. GUIKV appends fixed tokens one at a time internally
# to avoid treating a multi-token continuation as a new compressed prefill.
# The native single-generate implementation is retained only for comparison.
export QWEN3VL_STRUCTURED_FAST_DECODE_REBUILD_SLOTS="${QWEN3VL_STRUCTURED_FAST_DECODE_REBUILD_SLOTS:-0}"
export QWEN3VL_STRUCTURED_FAST_DECODE_SINGLE_GENERATE="${QWEN3VL_STRUCTURED_FAST_DECODE_SINGLE_GENERATE:-0}"
# Set to 1 to print each SFD dispatch, template plan, generated slots, and
# static/dynamic forward-step counts. Keep 0 for full evaluations.
export QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG="${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG:-0}"
export ANDROID_CONTROL_CURATED_EVAL_MODE="${ANDROID_CONTROL_CURATED_EVAL_MODE:-official}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/public/ytshen/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/public/ytshen/AndroidControl_Curated/images}"

# export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
# export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
export QWEN3VL_ANDROID_DENORM_ON_INFER="${QWEN3VL_ANDROID_DENORM_ON_INFER:-1}"
export QWEN3VL_ANDROID_DENORM_BASE="${QWEN3VL_ANDROID_DENORM_BASE:-1000}"

MODEL="${MODEL:-Qwen3-VL-4B-Instruct-GUIKV}"
DATA="${DATA:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_GUIKV/outputs_android_guikv_4B_SFD1112}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# DATA="GUIOdyssey_high_random_split"
# WORK_DIR="OUTPUT_GUIKV/outputs_guiodyssey_guikv_4B"
# DATA="Mind2Web_test_task"
# WORK_DIR="OUTPUT_GUIKV/outputs_mind2web_guikv_4B"
# DATA="AITW_general"
# WORK_DIR="OUTPUT_GUIKV/outputs_aitw_guikv_4B"

export GUIKV_HISTORY_STEPS="${GUIKV_HISTORY_STEPS:-4}"
export GUIKV_TOTAL_KEEP_RATIO="${GUIKV_TOTAL_KEEP_RATIO:-0.35}"
export GUIKV_WINDOW_SIZE="${GUIKV_WINDOW_SIZE:-8}"
export GUIKV_ALPHA="${GUIKV_ALPHA:-2.0}"
export GUIKV_TEMPERATURE="${GUIKV_TEMPERATURE:-3.5}"
export GUIKV_POOLING="${GUIKV_POOLING:-avgpool}"

export VLM_TIMING="${VLM_TIMING:-1}"
export VLM_STAGE_TIMING="${VLM_STAGE_TIMING:-1}"
export QWEN3VL_RUNTIME_TRACKING="${QWEN3VL_RUNTIME_TRACKING:-1}"
export QWEN3VL_PROFILE_FLOPS="${QWEN3VL_PROFILE_FLOPS:-1}"

export GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS="${GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS:-1}"
export GUI_ODYSSEY_MAX_HISTORY_IMAGES="${GUI_ODYSSEY_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS="${ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS:-1}"
export ANDROID_CONTROL_MAX_HISTORY_IMAGES="${ANDROID_CONTROL_MAX_HISTORY_IMAGES:-4}"
export ANDROID_CONTROL_SEQUENTIAL_ORDER="${ANDROID_CONTROL_SEQUENTIAL_ORDER:-1}"
export ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT="${ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT:-1}"
export ANDROID_CONTROL_OUTPUT_PROTOCOL="${ANDROID_CONTROL_OUTPUT_PROTOCOL:-auto}"
export ANDROID_CONTROL_STATE_PACKET_ENABLE=0
export QWEN3VL_ENABLE_TEMPLATE_PREFILL=0
export QWEN3VL_ENABLE_ROI_PRUNE=0
export QWEN3VL_ENABLE_ATTN_PRUNE=0
export AITW_HIS_NUM="${AITW_HIS_NUM:-4}"
export MIND2WEB_HIS_NUM="${MIND2WEB_HIS_NUM:-4}"

export VLM_EVAL_SAMPLE_MODE="${VLM_EVAL_SAMPLE_MODE:-off}"
export VLM_EVAL_SAMPLE_TASKS="${VLM_EVAL_SAMPLE_TASKS:-10}"
export VLM_EVAL_SAMPLE_COUNT="${VLM_EVAL_SAMPLE_COUNT:-20}"
export VLM_EVAL_SAMPLE_SEED="${VLM_EVAL_SAMPLE_SEED:-42}"

mkdir -p "${WORK_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "[Run] MODEL=${MODEL} DATA=${DATA} WORK_DIR=${WORK_DIR}"
echo "[Run] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[Run] sample_mode=${VLM_EVAL_SAMPLE_MODE} sample_tasks=${VLM_EVAL_SAMPLE_TASKS} sample_count=${VLM_EVAL_SAMPLE_COUNT} sample_seed=${VLM_EVAL_SAMPLE_SEED}"
echo "[Run] timing=${VLM_TIMING} stage_timing=${VLM_STAGE_TIMING} runtime_tracking=${QWEN3VL_RUNTIME_TRACKING} flops=${QWEN3VL_PROFILE_FLOPS}"
echo "[Run] structured_fast_decode=${QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE} structured_fast_decode_debug=${QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG}"
echo "[Run] sfd_impl=kv_cache_static_continuation single_generate=${QWEN3VL_STRUCTURED_FAST_DECODE_SINGLE_GENERATE} rebuild_slots=${QWEN3VL_STRUCTURED_FAST_DECODE_REBUILD_SLOTS}"
echo "[GUIKV] history_steps=${GUIKV_HISTORY_STEPS} total_keep_ratio=${GUIKV_TOTAL_KEEP_RATIO} window=${GUIKV_WINDOW_SIZE} alpha=${GUIKV_ALPHA} temperature=${GUIKV_TEMPERATURE} max_capacity=${GUIKV_MAX_CAPACITY_PROMPT:-ratio_mode}"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  "${PYTHON_BIN}" run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    2>&1 | tee -a "run_output_${TIMESTAMP}_qwen3_vl_guikv_${DATA}.log"
else
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --work-dir "${WORK_DIR}" \
    --mode all \
    2>&1 | tee -a "run_output_${TIMESTAMP}_qwen3_vl_guikv_${DATA}_SFD.log"
fi
