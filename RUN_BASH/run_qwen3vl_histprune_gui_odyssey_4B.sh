#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper around the complete, timed four-dataset runner. Prepare
# GUI-Odyssey annotations with history length four before running.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASETS="${DATASETS:-GUIOdyssey_high_task_split}"
export WORK_DIR="${WORK_DIR:-OUTPUT_HISTPRUNE/4B_gui_odyssey}"
exec bash "${ROOT_DIR}/RUN_BASH/run_qwen3vl_histprune_4datasets_4B.sh"
