# Free-running history-attention diagnostic

Run `bash RUN_BASH/run_history_attention_freerun_diagnostic.sh` from the repository root. Outputs default to `OUTPUT_history_attention_freerun/`, deliberately separate from the legacy gold teacher-forced diagnostic.

The tool greedily generates a response with the ordinary prompt, then replays its own generated token prefix without KV cache to extract selected causal attention rows. No gold answer is ever added to a model forward. Ground truth is used only afterwards to label outputs by the documented, non-official action equality and click/long-press `gt_min_bbox` IoU >= 0.5 rule.

`absolute` maps use `S_hist * local_patch` with a shared scale across a sample/category's history frames. `conditional` maps use `local_patch / F_i` only to show within-frame location. Attention is read-path evidence, not causal proof; frequent fixed peaks are reported as possible attention sinks/position bias.
