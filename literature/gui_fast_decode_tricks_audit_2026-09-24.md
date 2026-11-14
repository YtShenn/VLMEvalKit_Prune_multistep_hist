# Audit: accuracy-oriented structured-decoding tricks in GUI fast decode

## Scope and implementation audit

The current fast-decode code contains different mechanisms for different
datasets; they must not be collapsed into one stronger claim.

- AndroidControl: emits an action-first JSON prefix, normalizes a few aliases
  (`tap -> click`, `long press -> long_press`), and conditionally opens a bbox
  slot only for click/long_press. It does **not** token-mask action choices.
- AITW: maps several surface forms to canonical actions, including gesture
  direction versus content-scroll-direction conversion.
- GUI Odyssey: decodes action and direction from closed candidate sets, then
  opens only the direction or coordinate slot implied by the action. TYPE is
  deliberately returned to normal generation.
- All paths: deterministic template text is passed through the model as a
  multi-token continuation to extend the KV cache; a slot/parser/cache failure
  returns `None`, after which the caller uses normal `model.generate()`.

## Prior-art verdict

| Current trick | Existing closest work | Verdict |
|---|---|---|
| Action-conditioned GUI schema | AndroidControl action dictionaries; AndroidWorld tool actions; XGrammar-2 TagDispatch | Fully anticipated |
| Semantic equivalence / action canonicalization | AITW action matching and cross-agent action-space conversions | Standard adapter/evaluation technique |
| Conditional slot decode | Guidance control flow/select, TeCoD slots, XGrammar-2 tag dispatch | Fully anticipated |
| Cache-aware static template injection | TeCoD partitioned constrained decoding; Guidance/LLGuidance fast-forward | Fully anticipated; TeCoD is near identical |
| Safe fallback to regular decoding | TeCoD's no-template-match fallback; standard validation/retry/failover practice | Common robustness pattern; no exact GUI/cache-failure paper identified here |

## Evidence per trick

### 1. Action-conditioned schema

AndroidControl explicitly defines actions as JSON dictionaries with an
`action_type`; remaining fields are action-specific (`x,y`, `direction`,
`app_name`, or no parameter). This is the source task protocol, not a decoder
innovation. XGrammar-2 calls the generic decode-time operation
*tag-triggered structure switching* and implements `TagDispatch` for dynamic
agentic output structures.

Sources:

- https://github.com/google-research/google-research/tree/master/android_control
- https://arxiv.org/abs/2601.04426

### 2. Semantic equivalence-class normalization

The current AITW mapping reconciles different labels/semantics: e.g. tap and
click; press-back and navigate-back; and a finger gesture `swipe up` versus
the resulting content operation `scroll down`. This is a deterministic action
adapter, not semantic constrained decoding. AITW itself represents primitive
dual-point gestures, classifies them as tap or swipe, and evaluates actions by
equivalence classes (tap distance/box match; swipe primary axis). Cross-agent
dataset conversion also needs to reverse scroll directions when conventions
are opposite. Thus normalizing action vocabularies/coordinate conventions is
well-established in GUI benchmarks.

Sources:

- https://github.com/google-research/google-research/tree/master/android_in_the_wild
- https://arxiv.org/abs/2307.10088
- ACL HCI+NLP 2025 proceedings, Table 8: AgentCPM-GUI and UI-TARS use opposite
  scroll-direction conventions: https://aclanthology.org/2025.hcinlp-1.8.pdf

### 3. Conditional slot decoding

Guidance programs can first `select` a closed action set then execute Python
control flow and generate only the selected branch's slots. XGrammar-2 is an
explicit general-purpose constrained-decoding system for the same tag-to-
substructure transition. TeCoD generates only typed literal slots after a
template is chosen. The current GUI code is a hand-written instance with a
closed action head and a direction/coordinate/free-text branch.

Sources:

- https://guidance.readthedocs.io/en/latest/example_notebooks/tutorials/intro_to_guidance.html
- https://arxiv.org/abs/2601.04426
- https://arxiv.org/abs/2604.28028

### 4. Static template KV injection

TeCoD is the direct match: tokenize/compile a template, partition static
segments and dynamic slots, append static segments to KV, and grammar-decode
only slots. LLGuidance's `compute_ff_tokens()` returns deterministic grammar
tokens and documents that they bypass sampling and are processed as a short
prefill. Guidance documents KV-session reuse across template/control-flow
segments.

Sources:

- https://arxiv.org/abs/2604.28028
- https://sslab-cse-iitb.github.io/tecod/
- https://github.com/guidance-ai/llguidance/blob/main/sample_parser/src/minimal.rs

### 5. Fallback

TeCoD falls back to ordinary ICL when its template selector rejects an input.
CRANE combines constrained answer generation with unconstrained reasoning to
avoid the quality loss of rigid constraints. These establish the broad
constraint/flexibility fallback principle. In this repository, however, the
trigger is narrower and engineering-specific: malformed slot suffixes,
length/EOS failures, unsupported free-text branches, exceptions, missing
cache, or unsupported datasets return to ordinary `model.generate()`.

No peer-reviewed source was found in this focused audit for exactly
"template-KV prefill fails mid-GUI action, then resume by re-running plain
generation from the original prompt". This is not enough for a novelty claim:
it is a conventional safe-degradation/fail-open reliability guard. A paper
would need a new *decision rule* (e.g., calibrated risk/cost gate) and evidence
that it improves success-latency tradeoffs, rather than merely documenting
fallback coverage.

Sources:

- https://arxiv.org/abs/2604.28028
- https://arxiv.org/abs/2502.09061

## Claim-safe conclusion

The four techniques are mature primitives. The only defensible research
question is their *joint empirical interaction* in a long-horizon multimodal
GUI policy: whether task-aware canonicalization, branch selection, static
cache injection, and a calibrated safe-degradation policy yield a better
success/latency Pareto frontier than (a) generic structured generation, (b)
TeCoD-like partitioned decoding, and (c) plain generation under matched state
compression. That is an experimental systems claim, not a method-primitive
claim.
