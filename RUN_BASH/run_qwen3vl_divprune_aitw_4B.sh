#!/usr/bin/env bash
set -euo pipefail

# DivPrune runs once before decoder layer 0, so this is total visual retention.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-AITW_all}"
export DIVPRUNE_KEEP_RATIO="${DIVPRUNE_KEEP_RATIO:-0.45}"
export WORK_DIR="${WORK_DIR:-OUTPUT_DIVPRUNE/4B_aitw_total_keep${DIVPRUNE_KEEP_RATIO}}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_divprune_4B.sh"
