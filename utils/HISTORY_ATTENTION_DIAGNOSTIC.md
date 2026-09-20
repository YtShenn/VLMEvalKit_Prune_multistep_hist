# History visual-attention diagnostic

Run from the repository root:

```bash
python utils/history_attention_diagnostic.py --data GUIOdyssey_high_random_split --model /path/to/Qwen3-VL --work-dir OUTPUT/history_attn --num-samples 100 --history-steps 3 --layers last_third --action-token-mode all
```

Or edit `MODEL_PATH` and run the reproducible wrapper:

```bash
bash RUN_BASH/run_history_attention_diagnostic.sh
```

The wrapper selects samples round-robin by `task_id` (falling back to
`episode_id`, then task/question text), with one sample per task by default.
Override `NUM_SAMPLES`, `HISTORY_STEPS`, `ROI_JSON`, `MODEL_PATH`, or
`GUI_ODYSSEY_ROOT` as environment variables.

The script uses teacher forcing: it appends the dataset's direct string `answer` to
the normal full-history prompt and reads attention for gold-action prediction
positions. It loads a separate Qwen3-VL process with eager attention solely because
FlashAttention/SDPA frequently cannot materialize attention weights. Existing
benchmark defaults and results are untouched.

`--action-token-mode all` is the safe initial setting: inspect `selected_query_tokens`
in `per_sample.jsonl`, then use `semantic` to omit only JSON punctuation/whitespace.
Datasets lacking a direct `answer` string are recorded as skipped; the script never
uses generated text in place of gold actions. For GUI-Odyssey the annotation is the
command-style string in `answer`. Other datasets need an explicit, audited converter.

Optional `--roi-json` supports the repository's existing `roi_results.json` mapping:
keys are screenshot names and values are xyxy candidate boxes. Its historical ×4
coordinate convention is retained unless the JSON has `coordinate_scale`.

Attention is a diagnostic of a model read path, not causal proof. ROI enrichment > 1
means above-area relative attention, not automatic performance contribution. This
tool contains no occlusion, deletion, pruning, or causal-validation experiment.
