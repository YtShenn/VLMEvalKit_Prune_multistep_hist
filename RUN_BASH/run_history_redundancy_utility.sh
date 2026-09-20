#!/usr/bin/env bash
# Opt-in causal redundancy--utility experiment.  It does not modify evaluation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_redundancy_utility_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_redundancy_utility_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
# Full eligible-data study. Each selected region costs one multimodal forward;
# the conservative default region cap can be raised after a pilot.
WORK_DIR="${WORK_DIR:-OUTPUT_history_redundancy_utility/${DATASET}}"
# In `all` mode the manifest is optional and never filters samples; when given,
# it only adds outcome labels where the free-run report contains that step.
SUCCESS_MANIFEST="${SUCCESS_MANIFEST:-${SUCCESS_JSONL:-}}"
SUCCESS_SCOPE="${SUCCESS_SCOPE:-step}"
OUTCOME_SCOPE="${OUTCOME_SCOPE:-all}"
RUN_ALL="${RUN_ALL:-0}"
if [[ -z "$SUCCESS_MANIFEST" && "$OUTCOME_SCOPE" == "success" ]]; then
  AUTO_MANIFEST="OUTPUT_history_attention_freerun_all/${DATASET}/greedy_lastthird_all/workers"
  if [[ -d "$AUTO_MANIFEST" ]]; then
    SUCCESS_MANIFEST="$AUTO_MANIFEST"
    echo "[redundancy-utility] Using discovered free-run workers: $SUCCESS_MANIFEST (SUCCESS_SCOPE=$SUCCESS_SCOPE)" >&2
  else
    echo "Set SUCCESS_MANIFEST (or legacy SUCCESS_JSONL). No default free-run result was found at $AUTO_MANIFEST." >&2
    exit 2
  fi
elif [[ -z "$SUCCESS_MANIFEST" ]]; then
  echo '[redundancy-utility] OUTCOME_SCOPE=all: sampling all complete-history points; outcome labels will be unknown.' >&2
fi
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"

# Comma-separated physical GPU ids, e.g. GPU_IDS=0,1,2,3.  A single id remains
# fully supported.  Do not set both a remapped CUDA_VISIBLE_DEVICES list and
# unrelated physical GPU_IDS values.
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1,2,3,7}}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -eq 0 || -z "${GPUS[0]}" ]]; then echo 'GPU_IDS is empty.' >&2; exit 2; fi
# This is intentionally not torchrun/DDP: every process owns one GPU and one
# task shard.  NPROC_NUM is therefore the number of independent GPU workers.
NPROC_NUM="${NPROC_NUM:-${#GPUS[@]}}"
if ! [[ "$NPROC_NUM" =~ ^[1-9][0-9]*$ ]] || (( NPROC_NUM > ${#GPUS[@]} )); then
  echo "NPROC_NUM must be an integer in [1, ${#GPUS[@]}], got: $NPROC_NUM" >&2
  exit 2
fi
COMMON=(--data "$DATASET" --model "$MODEL_PATH" --success-scope "$SUCCESS_SCOPE" --outcome-scope "$OUTCOME_SCOPE"
  --sample-by task --samples-per-task 1 --history-steps "${HISTORY_STEPS:-4}" --region-scales "${REGION_SCALES:-4,global}"
  --max-regions-per-frame "${MAX_REGIONS_PER_FRAME:-4}" --mask-mode "${MASK_MODE:-mean_rgb}" --action-score "${ACTION_SCORE:-mean_logprob}" --seed "${SEED:-42}" --progress 1)
if [[ -n "$SUCCESS_MANIFEST" ]]; then COMMON+=(--success-manifest "$SUCCESS_MANIFEST"); fi
if [[ "$RUN_ALL" == "1" ]]; then COMMON+=(--all-eligible); else COMMON+=(--num-samples "${NUM_TASKS:-100}"); fi

mkdir -p "$WORK_DIR/workers"
echo "[redundancy-utility] GPUs=$GPU_IDS workers=$NPROC_NUM outcome_scope=$OUTCOME_SCOPE run_all=$RUN_ALL work_dir=$WORK_DIR" >&2
pids=()
cleanup_workers() {
  local exit_status=$?
  trap - INT TERM EXIT
  if (( ${#pids[@]} )); then
    echo '[redundancy-utility] stopping all worker process groups...' >&2
    for pid in "${pids[@]}"; do
      # Every worker is launched through setsid below, so its PID is also its
      # process-group id.  Killing the negative PID reaches Python and tee.
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
  exit "$exit_status"
}
trap cleanup_workers INT TERM EXIT
for ((shard = 0; shard < NPROC_NUM; shard++)); do
  gpu="${GPUS[$shard]//[[:space:]]/}"
  worker_dir="$WORK_DIR/workers/worker_$(printf '%02d' "$shard")"
  mkdir -p "$worker_dir"
  echo "[redundancy-utility] launch worker $shard on GPU $gpu" >&2
  # A separate session makes Ctrl+C cleanup reliable even though stdout is also
  # mirrored through tee for live progress and persistent logs.
  WORKER_LOG="$worker_dir/worker.log" CUDA_VISIBLE_DEVICES="$gpu" setsid bash -c \
    'set -o pipefail; "$@" 2>&1 | tee "$WORKER_LOG"' _ \
    "$PYTHON_BIN" utils/history_redundancy_utility.py "${COMMON[@]}" \
    --work-dir "$worker_dir" --num-shards "$NPROC_NUM" --shard-index "$shard" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
pids=()
if [[ "$status" != 0 ]]; then echo '[redundancy-utility] a worker failed; retaining worker outputs for diagnosis.' >&2; exit "$status"; fi
"$PYTHON_BIN" utils/history_redundancy_utility.py --merge-workers --work-dir "$WORK_DIR" --data "$DATASET" --model "$MODEL_PATH"
echo "[redundancy-utility] complete: $WORK_DIR/figures/redundancy_utility_density.png" >&2
