#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"
export QWEN3VL_PRUMERGE_RETAIN_RATIO="${QWEN3VL_PRUMERGE_RETAIN_RATIO:-0.30}"
export WORK_DIR="${WORK_DIR:-OUTPUT_PRUMERGE/4B_gui_odyssey_total_keep${QWEN3VL_PRUMERGE_RETAIN_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_prumerge_4B.sh"
