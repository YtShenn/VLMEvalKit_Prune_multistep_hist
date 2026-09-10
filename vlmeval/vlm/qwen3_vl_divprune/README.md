# DivPrune for Qwen3-VL

This isolated backend independently implements DivPrune (CVPR 2025) as a
training-free **Layer-0** baseline: image/video encoder and projector run in
full; only afterwards are redundant visual tokens physically removed before
the first Qwen3 language-decoder layer.

`DIVPRUNE_KEEP_RATIO` is the visual-token retention fraction and defaults to
`0.098`; therefore `prune_ratio = 1 - keep_ratio`.  `DIVPRUNE_SCOPE` defaults
to `global_all_visual`, which forms one candidate pool from all four history
screenshots plus the current screenshot.  `history_only` is an optional local
adaptation and is not the paper's faithful setting.

The selector uses the paper's greedy max-min cosine-distance algorithm on
projected visual embeddings. It is query-free and attention-free.  It keeps
only visual tokens, preserves text and original sequence order, and records
per-image before/after counts.  FLOPs must be read from the actual reduced
Layer-0 prefill length; the vision encoder/projector FLOPs are not removed.

Reference: Alvar et al., *DivPrune: Diversity-based Visual Token Pruning for
Large Multimodal Models*, CVPR 2025. Official repository:
https://github.com/vbdi/divprune
