# GUIPruner-reproduction (unofficial reproduction)

This is an isolated, **unofficial reproduction** of [GUIPruner](https://arxiv.org/html/2602.23235), implemented for this repository's HuggingFace Qwen3-VL backend.  No official implementation was available when this code was written.

It implements TAR at image-source level for up to four history screenshots and SSP after decoder layer 2 (paper numbering, one-based) for the current screenshot only. TAR defaults are `lambda=0.4`, `gamma=0.2`; SSP defaults are `mu=0.75`, `rho=0.3`.

By default, these are the paper's separate history/current budgets. Setting `GUI_PRUNER_OVERALL_KEEP_RATIO=eta` enables the optional global-budget experiment: with history original tokens `H` and current original tokens `C`, it keeps `lambda*H` history tokens and derives current SSP retention as `(eta*(H+C)-lambda*H)/C`. Infeasible settings fail explicitly; `GUI_PRUNER_CURRENT_KEEP_RATIO` is then ignored.

The implementation is intentionally independent of GUIKV, ST-Lite, AttnPrune and ROI pruning.  SSP requires the `eager` attention implementation to obtain the layer-2 attention matrix. It fails closed if the installed Qwen3-VL cache or M-RoPE internals are incompatible, rather than silently substituting another pruning mechanism.

Known migration difference: the paper evaluates Qwen2-VL/Qwen2.5-VL; this reproduction targets Qwen3-VL. It preserves the retained Qwen3-VL M-RoPE indices and trims the first two decoder-layer KV states to the same retained sequence. This is an architecture adaptation, not a claimed reproduction of the paper's reported metrics. The paper assumes a common history resolution; for mixed-resolution repository prompts TAR uses the newest history frame as the paper's `N_orig` reference while rescaling each image from its own aligned grid, and logs both quota and actual tokens.
