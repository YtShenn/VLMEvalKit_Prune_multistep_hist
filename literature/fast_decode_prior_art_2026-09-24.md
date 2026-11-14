# Fast decode prior-art audit

Scope: the repository's Android structured fast decode: (i) append deterministic
template segments in a multi-token continuation forward to extend the KV cache,
(ii) generate only action / conditional bbox slots, and (iii) emit an
action-first conditional schema.

## Bottom line

**TeCoD is a near-method-level match for the decoder mechanism.** It compiles a
parameterized output template into static token segments and dynamic literal
slots, then at inference prepends each static token segment to the KV cache and
runs grammar-constrained generation only for the slots. It explicitly calls
this *partitioned constrained decoding*. Thus static-token KV prefill / fast
forward, as an independent claimed contribution, is anticipated.

The closest difference is application and template source: TeCoD retrieves a
verified, input-dependent SQL template from a historical workload, whereas the
current code has a fixed GUI action protocol and executes an action-dependent
branch (bbox only for click / long_press). Neither difference changes the core
static-segment + dynamic-slot cache-advance algorithm.

## Closest work: near-equivalent

### TeCoD — Jivani, Maheshwari, Sarawagi (PACMMOD 2025; arXiv:2604.28028)

- Paper: *Reliable Answers for Recurring Questions: Boosting Text-to-SQL
  Accuracy with Template Constrained Decoding*.
- Link: https://arxiv.org/abs/2604.28028
- DOI: https://doi.org/10.1145/3769822
- Mechanism: offline full grammar-constrained decoding aligns a parameterized
  SQL template to model token IDs and partitions it into literals and slots.
  Online, static token IDs are prepended / KV-cached and GCD runs only over
  string/number slots. The authors identify tokenizer-boundary tokens as an
  important correctness issue.
- Reported rationale and effect: avoids grammar checks for fixed segments;
  1.5--2.3x lower latency than standard GCD and up to 36% execution-accuracy
  gain over ICL on matching SQL templates.
- Similarity to ours: **very high (core algorithm)**. Its static segments are
  our `_append_static`, and its constrained slots are our `_decode_until`.
- Difference: SQL templates are selected from a recurring labelled workload;
  GUI output schema is fixed and its conditional branch follows a predicted
  action. TeCoD uses GCD for slots; current implementation uses greedy
  stop-delimited slots without an explicit action/bbox grammar mask.

## General predecessors: same primitives, not the full specialized algorithm

### Guidance — Lundberg et al. (2023), plus LLGuidance runtime

- Link: https://github.com/guidance-ai/guidance
- A program interleaves literal text, `select`, generated spans, and Python
  branches while retaining KV state. LLGuidance explicitly exposes
  `compute_ff_tokens()` for deterministic grammar tokens, which bypass the
  sampler and are processed as a short prefill.
- Similarity: **high primitive overlap**; an explicit Guidance template can
  express the same GUI protocol.
- Difference: a general authoring/runtime system, rather than a task-specific
  template compiler and evaluation of GUI-agent latency/accuracy.

### LMQL — Beurer-Kellner, Fischer, Vechev (2022/2023)

- Link: https://arxiv.org/abs/2212.06094
- Compiles multi-part prompt programs with constraints/control flow into
  efficient model calls.
- Similarity: **conceptual**, especially template + control-flow slots.
- Difference: not evidence of the exact static-token partition/KV algorithm in
  the audited materials.

### Outlines — Willard & Louf (2023)

- Link: https://arxiv.org/abs/2307.09702
- Efficient regular/CFG-guided generation through vocabulary indexing.
- Similarity: grammar-valid outputs and mask efficiency.
- Difference: generalized mask engine; not template-partitioned KV prefill.

### XGrammar / XGrammar 2 — Dong et al. (2024); Li et al. (2026)

- Links: https://arxiv.org/abs/2411.15100 ; https://arxiv.org/abs/2601.04426
- Optimizes grammar-mask computation and dynamic agent/tool schemas.
- Similarity: conditional structured agent output and latency-oriented GCD.
- Difference: grammar-engine co-design, rather than bypassing static template
  spans by directly cache-prefilling them.

### Geng et al. (EMNLP 2023)

- Link: https://aclanthology.org/2023.emnlp-main.674/
- Shows input-dependent grammar-constrained decoding can improve structured
  task accuracy without fine-tuning.
- Similarity: explains why constrained output can improve measured task
  accuracy.
- Difference: no static-template KV fast-forward mechanism.

## Implications for claims

Do not claim novelty for any of: fixed-token insertion, forced-token
fast-forward, template-constrained decoding, slot-only constrained decoding,
or accuracy improvements caused by shrinking a structured output space.

A defensible remaining claim would need evidence for a GUI-specific problem
that TeCoD/Guidance do not solve, e.g. long-horizon multimodal state handling,
action-conditioned *semantic* validity beyond syntax, cache-safe integration
with Qwen3-VL, or a new matched-budget accuracy/latency trade-off. This must
be experimentally isolated from generic constrained-decoding gains.

## Follow-up: action-conditioned dynamic schemas

This is normally called a **tagged/discriminated union**: emit a discriminator
(`action_type`) and then permit exactly the fields of its selected variant.
It is not a new output-schema idea.

- The AndroidControl source dataset already represents actions exactly this
  way: `action_type` is followed only by fields unique to that action (click
  has `x,y`, scroll has `direction`, open-app has `app_name`, and navigation
  actions have no such parameter). Source:
  https://github.com/google-research/google-research/tree/master/android_control
- XGrammar-2 (Li et al., 2026) is an especially close systems prior: it calls
  this *tag-triggered structure switching* and introduces `TagDispatch` for
  dynamic structural dispatch in agentic tool-calling/response protocols.
  Source: https://arxiv.org/abs/2601.04426

Therefore the action protocol is faithful to the GUI environment and is likely
useful, but neither the conditional schema nor the decoder-side transition
from tag to a sub-schema can be claimed as novel. In the focused search, no
peer-reviewed GUI-agent paper was found that also combines this exact protocol
with TeCoD-style static-segment KV fast-forward; that narrower conjunction is
not established as novel either, because TeCoD supplies the fast-forward half
and XGrammar-2 supplies the tag-dispatch half.

## Sources checked

1. TeCoD project technical description and code repository, accessed
   2026-09-24: https://sslab-cse-iitb.github.io/tecod/ and
   https://github.com/SSLab-CSE-IITB/tecod
2. TeCoD arXiv record: https://arxiv.org/abs/2604.28028
3. Guidance acceleration documentation: https://guidance.readthedocs.io/en/stable/example_notebooks/tutorials/guidance_acceleration.html
4. LLGuidance minimal constrained-decoding loop:
   https://github.com/guidance-ai/llguidance/blob/main/sample_parser/src/minimal.rs
