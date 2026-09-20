#!/usr/bin/env bash
# Independent redundancy-only plots: no causal masking and no action scoring.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_redundancy_only_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_redundancy_only_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_redundancy_only/${DATASET}}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1}}"; IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_TASKS="${NUM_TASKS:-1000}"
NPROC_NUM="${NPROC_NUM:-${#GPUS[@]}}"
if ! [[ "$NPROC_NUM" =~ ^[1-9][0-9]*$ ]] || (( NPROC_NUM > ${#GPUS[@]} )); then echo "Invalid NPROC_NUM=$NPROC_NUM" >&2; exit 2; fi
RUN_ALL="${RUN_ALL:-0}"
COMMON=(--data "$DATASET" --model "$MODEL_PATH" --history-steps "${HISTORY_STEPS:-4}" --local-scale "${LOCAL_SCALE:-4}" --spatial-width "${SPATIAL_WIDTH:-16}" --spatial-height "${SPATIAL_HEIGHT:-9}" --seed "${SEED:-42}" --progress 1)
if [[ "$RUN_ALL" == 1 ]]; then COMMON+=(--all-eligible); else COMMON+=(--num-tasks "${NUM_TASKS:-500}"); fi
mkdir -p "$WORK_DIR/workers"; echo "[redundancy-only] GPUs=$GPU_IDS workers=$NPROC_NUM tasks=${NUM_TASKS:-500} run_all=$RUN_ALL" >&2
pids=()
cleanup() { local s=$?; trap - INT TERM EXIT; for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done; exit "$s"; }
trap cleanup INT TERM EXIT
for ((shard=0; shard<NPROC_NUM; shard++)); do
  gpu="${GPUS[$shard]//[[:space:]]/}"; worker="$WORK_DIR/workers/worker_$(printf '%02d' "$shard")"; mkdir -p "$worker"
  echo "[redundancy-only] launch worker $shard on GPU $gpu" >&2
  WORKER_LOG="$worker/worker.log" CUDA_VISIBLE_DEVICES="$gpu" setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee "$WORKER_LOG"' _ "$PYTHON_BIN" utils/history_redundancy_visualization.py "${COMMON[@]}" --work-dir "$worker" --num-shards "$NPROC_NUM" --shard-index "$shard" &
  pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; pids=()
if (( status )); then echo '[redundancy-only] worker failure; outputs retained.' >&2; exit "$status"; fi
"$PYTHON_BIN" utils/history_redundancy_visualization.py --merge-workers --work-dir "$WORK_DIR" --data "$DATASET" --model "$MODEL_PATH" --spatial-width "${SPATIAL_WIDTH:-16}" --spatial-height "${SPATIAL_HEIGHT:-9}"
echo "[redundancy-only] complete: $WORK_DIR/figures" >&2
