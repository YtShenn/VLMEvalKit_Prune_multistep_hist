import argparse
import json
from pathlib import Path
from typing import Any


# Edit this list directly when you want to add or remove compared fields.
DEFAULT_FIELDS = [
    "avg_total_wall_s",
    "avg_infer_wall_s",
    "avg_llm_stage_s",
    "avg_encode_s",
    "avg_prefill_s",
    "avg_decode_s",
    "avg_decode_tokens",
    "avg_decode_steps",
    "avg_prompt_seq_tokens",
    "avg_vision_flops",
    "avg_llm_flops",
    "avg_lm_head_flops",
    "avg_e2e_flops",
    "state_packet_total_packet_estimated_tokens",
    "state_packet_total_original_estimated_tokens",
]


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        abs_value = abs(value)
        if abs_value >= 1e6:
            return f"{value:.6e}"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def compute_speedup(baseline: Any, current: Any) -> str:
    baseline_num = to_number(baseline)
    current_num = to_number(current)
    if baseline_num is None or current_num is None:
        return "N/A"
    if current_num == 0:
        return "Inf" if baseline_num > 0 else "N/A"
    return f"{baseline_num / current_num:.6f}".rstrip("0").rstrip(".")


def default_label(path: str) -> str:
    parent = Path(path).parent.name
    return parent or Path(path).name


def build_speedup_pairs(labels: list[str]) -> list[tuple[int, int, str]]:
    pairs = []
    for current_idx in range(1, len(labels)):
        for baseline_idx in range(current_idx):
            pair_label = f"speedup({labels[current_idx]}/{labels[baseline_idx]})"
            pairs.append((baseline_idx, current_idx, pair_label))
    return pairs


def build_markdown_table(
    labels: list[str],
    summaries: list[dict[str, Any]],
    fields: list[str],
) -> str:
    speedup_pairs = build_speedup_pairs(labels)
    header_cells = ["field", *labels, *[pair_label for _, _, pair_label in speedup_pairs]]
    align_cells = ["---", *(["---:"] * (len(header_cells) - 1))]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(align_cells) + " |",
    ]
    for field in fields:
        raw_values = [summary.get(field) for summary in summaries]
        row_cells = [field, *[format_value(value) for value in raw_values]]
        for baseline_idx, current_idx, _ in speedup_pairs:
            row_cells.append(
                compute_speedup(
                    baseline=raw_values[baseline_idx],
                    current=raw_values[current_idx],
                )
            )
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected fields from multiple summary.json files and output "
            "a Markdown table. Speedup columns are generated for each later file "
            "vs each earlier file, using earlier/later as the numeric formula."
        )
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        help="One or more summary.json paths to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels matching the order of summary_paths",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help=(
            "Fields to compare. If omitted, uses DEFAULT_FIELDS defined in the script."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.summary_paths) < 2:
        raise ValueError("Please provide at least two summary.json paths.")

    summaries = [load_json(path) for path in args.summary_paths]

    if args.labels is not None:
        if len(args.labels) != len(args.summary_paths):
            raise ValueError("--labels count must match the number of summary paths.")
        labels = args.labels
    else:
        labels = [default_label(path) for path in args.summary_paths]

    table = build_markdown_table(
        labels=labels,
        summaries=summaries,
        fields=args.fields,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")
    else:
        print(table)


if __name__ == "__main__":
    main()
'''
python utils/compare_summary_json.py \
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_baseline_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json" \
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_state_packet_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json" \
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_state_packet_structured_fast_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json" \
  --labels baseline state_packet state_packet_structured_decode \
  --output compare_table_structure_decode.md
'''
