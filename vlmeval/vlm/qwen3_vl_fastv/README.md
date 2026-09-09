# FastV for Qwen3-VL with four history screenshots

This isolated inference-only backend implements FastV's official scoring rule:
run layers `0..K-1`, average the layer `K-1` self-attention across heads for
the final sequence query token, globally keep the top
`round(N_visual * (1 - R))` visual tokens, and physically remove the rest
before layer `K`. KV cache, masks, DeepStack features and Qwen3-VL M-RoPE
position subsets are selected with the same indices.

Unlike FastV's original single contiguous LLaVA image span, Qwen3-VL can have
five heterogeneous image blocks: four history screenshots in old-to-new order
followed by the current screenshot, with text between them. The only adaptation
is therefore to dynamically collect the union of injected visual-token
positions. The global FastV ranking covers all five images, including the
current image. This differs intentionally from HistPrune, which only prunes
history screenshots.

Environment variables: `FASTV_ENABLED=1`, `FASTV_K=2`, `FASTV_R=0.5`, and
`FASTV_DEBUG=0`. `FASTV_R=0` is an exact no-pruning control path.
