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


def build_markdown_table(
    left_label: str,
    right_label: str,
    left_data: dict[str, Any],
    right_data: dict[str, Any],
    fields: list[str],
) -> str:
    ratio_label = f"{left_label}/{right_label}"
    lines = [
        f"| field | {left_label} | {right_label} | speedup({ratio_label}) |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        left_value = left_data.get(field)
        right_value = right_data.get(field)
        lines.append(
            "| {field} | {left} | {right} | {speedup} |".format(
                field=field,
                left=format_value(left_value),
                right=format_value(right_value),
                speedup=compute_speedup(left_value, right_value),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected fields from two summary.json files and output a "
            "Markdown table. The speedup column is computed as file1/file2."
        )
    )
    parser.add_argument("summary_a", help="Baseline summary.json path")
    parser.add_argument("summary_b", help="Compared summary.json path")
    parser.add_argument(
        "--label-a",
        default=None,
        help="Optional display label for summary_a",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help="Optional display label for summary_b",
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

    summary_a = load_json(args.summary_a)
    summary_b = load_json(args.summary_b)
    label_a = args.label_a or default_label(args.summary_a)
    label_b = args.label_b or default_label(args.summary_b)

    table = build_markdown_table(
        left_label=label_a,
        right_label=label_b,
        left_data=summary_a,
        right_data=summary_b,
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
用法：
python utils/compare_summary_json.py \
'OUTPUT/ablation_android_control_hist4_keep_system_prompt_baseline_official_eval_node5_0810\\(only_HTI\\)/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260810_G4e70879f/summary.json' \
'OUTPUT/ablation_android_control_hist4_keep_system_prompt_state_packet_official_eval_node5_0810\\(only_HTI\\)/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260810_G4e70879f/summary.json' \
--label-a baseline \
--label-b state-packet \
--output compare_table_android_ratio1.md
'''
