#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-AndroidControl_Curated_High_Task_Improved}"
export QWEN3VL_PRUMERGE_RETAIN_RATIO="${QWEN3VL_PRUMERGE_RETAIN_RATIO:-0.35}"
export WORK_DIR="${WORK_DIR:-OUTPUT_PRUMERGE/4B_android_control_total_keep${QWEN3VL_PRUMERGE_RETAIN_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_prumerge_4B.sh"
