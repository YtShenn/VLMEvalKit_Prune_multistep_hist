#!/usr/bin/env bash
set -euo pipefail

# DivPrune runs once before decoder layer 0, so this is total visual retention.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"
export DIVPRUNE_KEEP_RATIO="${DIVPRUNE_KEEP_RATIO:-0.30}"
export WORK_DIR="${WORK_DIR:-OUTPUT_DIVPRUNE/4B_gui_odyssey_total_keep${DIVPRUNE_KEEP_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_divprune_4B.sh"
