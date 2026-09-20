#!/usr/bin/env bash
# Read-only free-running attention observation.  This never changes benchmark inference.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/ytshen/anaconda3/envs/qwen3_5/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vlmeval_history_attention_freerun_mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/vlmeval_history_attention_freerun_cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL_PATH="${MODEL_PATH:-/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-AndroidControl_Curated_High_Task_Improved}"
## Default to every row with the requested complete history. For a quick
## diagnostic override e.g. RUN_NAME=smoke NUM_SAMPLES=20 SAMPLE_BY=task.
RUN_NAME="${RUN_NAME:-greedy_lastthird_all}"
WORK_DIR="${WORK_DIR:-OUTPUT_history_attention_freerun_all/${DATASET}/${RUN_NAME}}"
## --num-samples=-1 is implemented by the diagnostic as all eligible rows.
## Sequential sampling prevents the task-balanced per-task cap from excluding
## later steps of the same trajectory.
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SAMPLE_BY="${SAMPLE_BY:-sequential}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1}"
HISTORY_STEPS="${HISTORY_STEPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPROC_NUM="${NPROC_NUM:-3}"
export ANDROID_CONTROL_CURATED_ROOT="${ANDROID_CONTROL_CURATED_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated}"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="${ANDROID_CONTROL_CURATED_IMAGE_ROOT:-/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images}"
export GUI_ODYSSEY_STATE_PACKET_ENABLE=0 ANDROID_CONTROL_STATE_PACKET_ENABLE=0
export ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS=1 ANDROID_CONTROL_MAX_HISTORY_IMAGES="$HISTORY_STEPS"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if (( NPROC_NUM < 1 )); then
  echo "NPROC_NUM must be >= 1" >&2; exit 2
fi
if (( NPROC_NUM > ${#GPU_IDS[@]} )); then
  echo "NPROC_NUM=$NPROC_NUM but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes only ${#GPU_IDS[@]} GPU(s)." >&2; exit 2
fi

WORKERS_DIR="$WORK_DIR/workers"
if [[ -d "$WORKERS_DIR" ]] && [[ -n "$(find "$WORKERS_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to mix results with existing worker outputs: $WORKERS_DIR" >&2
  echo "Choose a new RUN_NAME/WORK_DIR, or move the prior run aside before restarting." >&2
  exit 2
fi
mkdir -p "$WORKERS_DIR"
pids=()
for (( rank=0; rank<NPROC_NUM; rank++ )); do
  worker_dir="$WORKERS_DIR/worker_$(printf '%02d' "$rank")"
  worker_mpl="$MPLCONFIGDIR/worker_$(printf '%02d' "$rank")"
  mkdir -p "$worker_mpl"
  echo "[free-running-attn] worker $rank/$NPROC_NUM on GPU ${GPU_IDS[$rank]} -> $worker_dir"
  (
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$rank]}" MPLCONFIGDIR="$worker_mpl" \
    "$PYTHON_BIN" utils/history_attention_freerun_diagnostic.py --data "$DATASET" --model "$MODEL_PATH" --work-dir "$worker_dir" \
      --query-source free_running --num-samples "$NUM_SAMPLES" --history-steps "$HISTORY_STEPS" --sample-by "$SAMPLE_BY" \
      --samples-per-task "$SAMPLES_PER_TASK" --layers last_third --max-new-tokens "$MAX_NEW_TOKENS" --seed "${SEED:-42}" \
      --num-shards "$NPROC_NUM" --shard-index "$rank" --skip-summary-figures
  ) >"$WORKERS_DIR/worker_$(printf '%02d' "$rank").log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON_BIN" utils/merge_history_attention_freerun_results.py \
  --workers-dir "$WORKERS_DIR" --work-dir "$WORK_DIR" --num-workers "$NPROC_NUM" --seed "${SEED:-42}"
