# HistPrune-GUI reproduction/adaptation for Qwen3-VL

This isolated training-free implementation adapts HistPrune-GUI to this
repository's custom HuggingFace Qwen3-VL backend. It physically prunes only
historical screenshot visual tokens after a configurable decoder layer; the
current screenshot, text, and vision boundary tokens are always retained.

At most four history screenshots are accepted in prompt order `oldest -> newest
-> current`. `HISTPRUNE_HISTORY_KEEP_RATIO` fixes the total history-token
budget; `HISTPRUNE_TEMPORAL_WEIGHTS` distributes that budget recent-first.
Supported modes are `random`, `uniform`, `sobel_foreground`, and
`sobel_background`. The optional `HISTPRUNE_VISUAL_CACHE=1` caches ViT and
DeepStack features by absolute image path plus processed grid; call
`reset_histprune_episode_cache()` at every episode boundary.

Set `HISTPRUNE_LOG_STATS=1` to print one JSON budget audit per pruning step.
It is disabled by default; the latest record remains available at
`model.config.text_config._histprune_last_stats` for programmatic collection.
At the end of an evaluation, `summary.json` additionally reports token-weighted
global retention/pruning rates across all steps containing history: historical
visual tokens only (`HistPrune_history_visual_*`), all visual tokens including
the unpruned current frame (`HistPrune_visual_*`), and full prompts including
text (`HistPrune_prompt_*`).  The `*_global` values are computed from summed
token totals, while `*_avg` is the unweighted per-step mean.

HistPrune requires generation KV caching. Its runner keeps caching on and
defaults FLOPs profiling and forced CUDA synchronization to off; enable
`QWEN3VL_PROFILE_FLOPS=1` or `VLM_TIMING_SYNC=1` only for a measurement run.

Example:

```bash
export HISTPRUNE_MODE=random HISTPRUNE_HISTORY_KEEP_RATIO=.40
export HISTPRUNE_TEMPORAL_WEIGHTS=.4,.3,.2,.1 HISTPRUNE_DROP_LAYER=4
export HISTPRUNE_VISUAL_CACHE=0
python run.py --data AndroidControl_Curated_High_Task_Improved \
  --model Qwen3-VL-4B-Instruct-HistPrune --work-dir OUTPUT/histprune --mode all
```

This is an architecture adaptation, not a claim that the paper's Qwen2/2.5
metrics are reproduced. It intentionally does not implement FastV, PDrop,
DART, DivPrune, or SparseVLM. GUIPruner, GUIKV, STLite, AttnPrune, and ROI
pruning are separate methods and are not invoked by this backend.
