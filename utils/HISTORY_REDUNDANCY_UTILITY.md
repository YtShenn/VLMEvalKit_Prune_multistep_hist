# Full-history redundancy--utility experiment

`history_redundancy_utility.py` is an offline causal analysis.  It does not
modify VLMEval inference, model code, prompts, pruning, or compression.  It
forces the full-history switches on only inside its own process.

The recommended input is an evaluator manifest such as
`{"successful_trajectory_keys": ["/absolute/path/to/task", ...]}`.  The key
must equal the dataset `_trajectory_key`.  Alternatively, run a complete
free-running baseline (all steps of every trajectory that may be analysed),
then point `SUCCESS_MANIFEST` at its `per_sample.jsonl`:

```bash
# Default launcher configuration: 20 task-balanced samples, one decision point
# per task, 2 history frames, and at most 4 regions per frame.
bash RUN_BASH/run_history_redundancy_utility.sh

# Example scale-up after validating the smoke-test outputs.
NUM_SAMPLES=20 HISTORY_STEPS=4 MAX_REGIONS_PER_FRAME=48 \
REGION_SCALES=4,8,global \
bash RUN_BASH/run_history_redundancy_utility.sh
```

The free-running format must contain `sample_id` and
`correctness.overall_success`, as emitted by
`history_attention_freerun_diagnostic.py`.  The experiment maps those ids back
to the dataset's `_trajectory_key`, and retains a trajectory only if every
annotated step is present and correct.  This deliberately fails when the
free-running report sampled only a subset of a trajectory. `SUCCESS_JSONL`
remains accepted as a backward-compatible launcher alias.

In this checkout, omitting the manifest makes the launcher discover
`OUTPUT_history_attention_freerun_all/.../workers` and use `SUCCESS_SCOPE=step`.
That practical default means the current rollout decision is correct; it must
not be reported as whole-trajectory success. For the strict paper setting,
supply evaluator trajectory keys and use `SUCCESS_SCOPE=trajectory`.

## Full-data, multi-GPU run

The launcher now defaults to 500 task-balanced complete-history decision points
(`RUN_ALL=0`, `NUM_TASKS=500`, `OUTCOME_SCOPE=all`, `SEED=42`) and uses one
process per comma-separated GPU id. Each worker gets
a deterministic task/trajectory shard and writes only below
`WORK_DIR/workers/worker_XX`, so it is safe to resume diagnosis after a failed
worker without corrupting another worker's files. After every worker exits
successfully, the launcher merges the rows and creates the final density plot.

```bash
# Four independent model workers. tqdm shows per-worker task and masked-region
# counts, throughput, and ETA in the terminal; full logs are also retained.
GPU_IDS=0,1,2,3 NPROC_NUM=4 RUN_ALL=0 NUM_TASKS=500 OUTCOME_SCOPE=all SEED=42 \
HISTORY_STEPS=4 MAX_REGIONS_PER_FRAME=4 REGION_SCALES=4,global \
bash RUN_BASH/run_history_redundancy_utility.sh
```

`NPROC_NUM` is the number of independent one-GPU workers, not a DDP/`torchrun`
world size. It defaults to the number of IDs in `GPU_IDS` and cannot exceed
that number. `GPU_IDS=7` retains a single-GPU run. `RUN_ALL=1` processes every
complete-history step. In `OUTCOME_SCOPE=all`, a success manifest is optional
and never filters a point; when supplied, it only populates the per-row
`outcome` field (`success`, `failure`, or `unknown`). Increase
`MAX_REGIONS_PER_FRAME` only after measuring the pilot throughput: each region
requires another multimodal forward pass.

For each history image, default scales create 4x4-token and 8x8-token tiles,
plus one full-frame (`global`) region.  `MAX_REGIONS_PER_FRAME` is a uniform,
seeded sampling cap applied after tiling; set it to `0` for exhaustive tiling
(expect one full multimodal forward per region).  `REGION_SCALES=2,4,8,global`
changes the scales without changing normal evaluation.

Outputs are in `WORK_DIR`:

- `regions.jsonl` / `regions.csv`: one masked-history region per row, including
  normalized-coordinate region, pixel box, scale/type, redundancy, and utility.
- `figures/redundancy_utility_density.png`: log-density hexbin with colored
  overlays for scale/type.
- `metadata.json` and `summary.json`: exact configuration and coverage.

Redundancy is cosine similarity between the mean Qwen3-VL post-merge visual
encoder feature for a region and the same normalized rectangle in the preceding
history image.  Utility is the baseline teacher-forced log likelihood of the
entire gold action minus its likelihood after only that source-image rectangle
is replaced with mean RGB.  The first historical image has no predecessor and
is recorded with null redundancy rather than fabricating a comparison.
