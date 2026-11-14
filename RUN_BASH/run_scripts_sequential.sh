#!/usr/bin/env bash
set -euo pipefail

# Run multiple bash scripts sequentially: one finishes, then the next starts.
# Usage examples:
#   bash RUN_BASH/run_scripts_sequential.sh RUN_BASH/a.sh RUN_BASH/b.sh
#   SEQ_SCRIPT_FILE=RUN_BASH/scripts_to_run.txt bash RUN_BASH/run_scripts_sequential.sh
#   SEQ_STOP_ON_ERROR=0 bash RUN_BASH/run_scripts_sequential.sh RUN_BASH/a.sh RUN_BASH/b.sh
#
# Optional env vars:
#   SEQ_SCRIPT_FILE=...      file containing one script path per line; blank lines and # comments ignored
#   SEQ_STOP_ON_ERROR=1      stop on first failed script; set 0 to continue after failures
#   SEQ_LOG_DIR=...          directory for per-script logs and summary.tsv
#   SEQ_BASH_BIN=bash        shell executable used to launch each script
#   SEQ_RUN_PREFIX=...       optional command prefix, e.g. "conda run -n qwen3_5"

DEFAULT_SCRIPTS=(
    # Edit this list when running without command-line args or SEQ_SCRIPT_FILE.
    # "RUN_BASH/run_gui_odyssey_history_state_packet_qwen3_5.sh"
    # "RUN_BASH/run_android_control_history_state_packet_qwen3_5.sh"
    # "RUN_BASH/run_android_control_history_state_packet_only_hist4.sh"
    # "RUN_BASH/run_android_control_history_fast_decode_only_hist4.sh"
    # "RUN_BASH/run_android_control_history_state_packet_qwen3_5_current_frame_attn_prune.sh"
    # "RUN_BASH/run_android_control_history_baseline_hist4.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_baseline.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_fast_decode_only.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_state_packet_fast_decode_only.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_state_packet_only.sh"
    # "RUN_BASH/run_gui_odyssey_history_state_packet_qwen3_5.sh"
    # "RUN_BASH/run_android_control_history_baseline_hist4.sh"
    # "RUN_BASH/run_android_control_history_state_packet_qwen3_5_current_frame_attn_prune.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_android.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_aitw.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_gui_odyssey.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_aitw.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_android.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_mind2web.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_guiodyssey.sh"
    # "RUN_BASH/run_qwen3vl_histprune_android.sh"
    # "RUN_BASH/run_qwen3vl_histprune_aitw.sh"
    # "RUN_BASH/run_qwen3vl_histprune_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_android.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_aitw.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_histprune_odyssey.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_odyssey.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_android.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_aitw.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_mind2web.sh"
    # "RUN_BASH/run_android_control_history_baseline_hist4_no_hist.sh"
    # "RUN_BASH/run_qwen3vl_sparsevlm_android_4B.sh"
    # "RUN_BASH/run_qwen3vl_sparsevlm_aitw_4B.sh"
    # "RUN_BASH/run_qwen3vl_sparsevlm_mind2web_4B.sh"
    # "RUN_BASH/run_qwen3vl_prumerge_android_4B.sh"
    # "RUN_BASH/run_qwen3vl_prumerge_aitw_4B.sh"
    # "RUN_BASH/run_qwen3vl_prumerge_mind2web_4B.sh"
    # "RUN_BASH/run_qwen3vl_divprune_android_4B.sh"
    # "RUN_BASH/run_qwen3vl_divprune_aitw_4B.sh"
    # "RUN_BASH/run_qwen3vl_divprune_mind2web_4B.sh"

    # 测试一下之前的flops影响大不大
    # "RUN_BASH/run_qwen3vl_guikv_4B_aitw.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_aitw.sh"
    # "RUN_BASH/run_qwen3vl_histprune_aitw.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_aitw.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_aitw.sh"

    # "RUN_BASH/run_android_control_history_modal_ablation_qwen3_5.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_android.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_mind2web.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_android.sh"
    # "RUN_BASH/run_qwen3_vl_stlite_4B_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_histprune_android.sh"
    # "RUN_BASH/run_qwen3vl_histprune_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_android.sh"
    # "RUN_BASH/run_qwen3vl_guipruner_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_android.sh"
    # "RUN_BASH/run_qwen3vl_fastv_4B_mind2web.sh"

    # "RUN_BASH/run_qwen3vl_fastv_4B_odyssey.sh"
    # "RUN_BASH/run_qwen3vl_sparsevlm_odyssey_4B.sh"
    # "RUN_BASH/run_qwen3vl_prumerge_odyssey_4B.sh"
    # "RUN_BASH/run_qwen3vl_divprune_odyssey_4B.sh"

    # "RUN_BASH/run_aitw_history_roi_gt_vs_prediction.sh"
    # "RUN_BASH/run_mind2web_history_roi_gt_vs_prediction.sh"

    # "RUN_BASH/run_android_control_history_modal_ablation_qwen3_5.sh"
    # "RUN_BASH/run_gui_odyssey_history_roi_gt_vs_prediction.sh"
    
    # "RUN_BASH/run_android_control_history_modal_ablation_qwen3_5.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_state_packet_only.sh"
    # "RUN_BASH/run_aitw_history_modal_ablation_qwen3_5.sh"
    # RUN_BASH/run_mind2web_history_modal_ablation_qwen3_5.sh

    # "RUN_BASH/run_qwen3vl_aitw_4B_state_packet_only.sh"
    # "RUN_BASH/run_qwen3vl_aitw_4B_fast_decode_only.sh"

    # "RUN_BASH/run_qwen3vl_guikv_4B_android.sh"
    "RUN_BASH/run_qwen3vl_guikv_4B_aitw.sh"
    "RUN_BASH/run_qwen3vl_guikv_4B_mind2web.sh"
    # "RUN_BASH/run_qwen3vl_guikv_4B_gui_odyssey.sh"

)

STOP_ON_ERROR="${SEQ_STOP_ON_ERROR:-1}"
LOG_DIR="${SEQ_LOG_DIR:-OUTPUT/sequential_run_logs}"
BASH_BIN="${SEQ_BASH_BIN:-bash}"
RUN_PREFIX="${SEQ_RUN_PREFIX:-}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
SUMMARY_FILE="${LOG_DIR}/${TIMESTAMP}_summary.tsv"

on_interrupt() {
  echo "[Sequential] interrupted."
  exit 130
}

trap on_interrupt INT TERM

is_truthy() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_name() {
  local raw="$1"
  raw="$(basename "$raw")"
  raw="${raw%.sh}"
  raw="${raw// /_}"
  raw="${raw//[^A-Za-z0-9_.-]/_}"
  printf '%s' "$raw"
}

format_seconds() {
  local seconds="$1"
  printf '%02d:%02d:%02d' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

SCRIPT_LIST=()
if [[ "$#" -gt 0 ]]; then
  SCRIPT_LIST=("$@")
elif [[ -n "${SEQ_SCRIPT_FILE:-}" ]]; then
  if [[ ! -f "${SEQ_SCRIPT_FILE}" ]]; then
    echo "[ERROR] SEQ_SCRIPT_FILE not found: ${SEQ_SCRIPT_FILE}"
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue
    SCRIPT_LIST+=("$line")
  done < "${SEQ_SCRIPT_FILE}"
else
  SCRIPT_LIST=("${DEFAULT_SCRIPTS[@]}")
fi

if [[ "${#SCRIPT_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] No scripts configured."
  echo "Provide scripts as args, set SEQ_SCRIPT_FILE, or edit DEFAULT_SCRIPTS in $0."
  exit 1
fi

mkdir -p "${LOG_DIR}"
printf 'index\tscript\tstatus\texit_code\tstart_time\tend_time\tduration\tlog_file\n' > "${SUMMARY_FILE}"

echo "[Sequential] start_time=$(date '+%F %T')"
echo "[Sequential] log_dir=${LOG_DIR}"
echo "[Sequential] summary=${SUMMARY_FILE}"
echo "[Sequential] stop_on_error=${STOP_ON_ERROR}"
if [[ -n "${RUN_PREFIX}" ]]; then
  echo "[Sequential] run_prefix=${RUN_PREFIX}"
fi
echo "[Sequential] configured scripts:"
for script_path in "${SCRIPT_LIST[@]}"; do
  echo "  - ${script_path}"
done

failures=0
total="${#SCRIPT_LIST[@]}"

for index in "${!SCRIPT_LIST[@]}"; do
  script_path="${SCRIPT_LIST[$index]}"
  display_index=$((index + 1))

  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Script not found: ${script_path}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${display_index}" "${script_path}" "missing" "127" "" "" "" "" >> "${SUMMARY_FILE}"
    exit 127
  fi

  script_name="$(normalize_name "${script_path}")"
  log_file="${LOG_DIR}/${TIMESTAMP}_$(printf '%02d' "${display_index}")_${script_name}.log"
  start_epoch="$(date +%s)"
  start_time="$(date '+%F %T')"

  echo "[Sequential] (${display_index}/${total}) start ${script_path} at ${start_time}"
  echo "[Sequential] (${display_index}/${total}) log -> ${log_file}"

  set +e
  if [[ -n "${RUN_PREFIX}" ]]; then
    ${RUN_PREFIX} "${BASH_BIN}" "${script_path}" 2>&1 | tee "${log_file}"
  else
    "${BASH_BIN}" "${script_path}" 2>&1 | tee "${log_file}"
  fi
  status=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  end_time="$(date '+%F %T')"
  duration_seconds=$((end_epoch - start_epoch))
  duration="$(format_seconds "${duration_seconds}")"

  if [[ "${status}" -eq 0 ]]; then
    run_status="success"
    echo "[Sequential] (${display_index}/${total}) done ${script_path} duration=${duration}"
  elif [[ "${status}" -eq 130 ]]; then
    run_status="interrupted"
    echo "[Sequential] (${display_index}/${total}) interrupted script=${script_path} duration=${duration}"
  else
    run_status="failed"
    failures=$((failures + 1))
    echo "[Sequential] (${display_index}/${total}) failed status=${status} script=${script_path} duration=${duration}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${display_index}" "${script_path}" "${run_status}" "${status}" \
    "${start_time}" "${end_time}" "${duration}" "${log_file}" >> "${SUMMARY_FILE}"

  if [[ "${status}" -eq 130 ]]; then
    exit 130
  fi
  if [[ "${status}" -ne 0 ]] && is_truthy "${STOP_ON_ERROR}"; then
    echo "[Sequential] stop_on_error=${STOP_ON_ERROR}, aborting."
    exit "${status}"
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "[Sequential] finished with failures=${failures}. summary=${SUMMARY_FILE}"
  exit 1
fi

echo "[Sequential] all scripts finished successfully. summary=${SUMMARY_FILE}"
