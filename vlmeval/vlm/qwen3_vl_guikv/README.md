# Qwen3-VL GUI-KV

This backend adds GUI-KV cache compression for Qwen3-VL without changing the
existing Qwen3-VL, state-packet, fastdecode, or pruning paths.

It follows the official SalesforceAIResearch/GUI-KV design:

- uniform `max_capacity_prompt` budget for every decoder layer
- recent-window attention pooling
- spatial saliency from visual hidden-state L2 norms
- temporal redundancy scoring across screenshots with QR projection residuals

History screenshots are supplied by the repository's existing dataset prompt
builders. The GUI-KV defaults set those builders to keep up to four history
screenshots, with the current screenshot last, matching the paper's five
screenshot setting. Single-image datasets automatically run without temporal
redundancy scoring.

Qwen3-VL adaptation notes:

- the patch is attached to one model instance instead of globally replacing
  Transformers classes;
- image token ranges are recovered from `mm_token_type_ids` or image token ids
  in the current generated prompt;
- Qwen3-VL `image_grid_thw` is used only as a fallback to split contiguous image
  token runs.

Useful environment variables:

- `GUIKV_MAX_CAPACITY_PROMPT` optional absolute-token override
- `GUIKV_TOTAL_KEEP_RATIO` default `0.40` when absolute capacity is unset
- `GUIKV_WINDOW_SIZE` default `8`
- `GUIKV_ALPHA` default `2.0`
- `GUIKV_TEMPERATURE` default `3.5`
- `GUIKV_HISTORY_STEPS` default `4`
- `QWEN3VL_RUNTIME_TRACKING=1`, `VLM_TIMING=1`, and `VLM_STAGE_TIMING=1`
  enable the `summary.json` latency fields; `QWEN3VL_PROFILE_FLOPS=1` enables
  FLOPS fields. The provided GUI-KV run script turns these on by default.

If `GUIKV_MAX_CAPACITY_PROMPT` is set to a positive integer, GUI-KV uses the
official absolute-token budget style. Otherwise it derives the prompt cache
budget from `GUIKV_TOTAL_KEEP_RATIO`. Previous and current screenshots share
this global budget through GUI-KV's official score-and-top-k selection.
