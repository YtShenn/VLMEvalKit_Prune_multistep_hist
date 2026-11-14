#!/usr/bin/env bash
# Deployment-faithful redundancy--utility control:
# physically remove one full history image from the multimodal prompt, so its
# visual tokens are absent.  This wrapper deliberately keeps the legacy
# pixel-mask launcher untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A frame deletion is meaningful only for a complete image, not an arbitrary
# local patch.  Callers may still override HISTORY_STEPS, NUM_TASKS/RUN_ALL,
# GPU_IDS, NPROC_NUM, WORK_DIR, and the usual model/data variables.
export INTERVENTION_MODE=drop_frame
export REGION_SCALES="${REGION_SCALES:-global}"


exec bash "$SCRIPT_DIR/run_history_redundancy_utility.sh"
