# First-sample screening and refill validation

Validated on 2026-11-20 using the production environment: Torch 2.7.1+cu126,
Triton 3.3.1 and shared H200 GPUs. Screening, refill, fused LayerNorm and replay
remain optional. These measurements predate the later promotion of confidence
chunks of five; the screening comparisons explicitly used chunks of one. See the [usage and output contract](inference_instructions.md#optional-first-sample-screening-and-batch-refill).

## Full-checkpoint comparisons

The screening comparison used four seeds (101–104) of the shipped synthetic
128-residue, two-protein-chain example, batch size two, BF16 mixed precision,
ten recycles, 200 steps, three samples, both sample chunks one, Torch LayerNorm,
TMK3 multiplication and reference diffusion attention. Features and seeds were
held fixed across arms. Comparisons included every returned coordinate,
confidence-logit, contact, summary and full-confidence field, excluding only the
new screening metadata.

| Arm | Samples retained per job | Batched sample rounds | Retained outputs versus full baseline |
| --- | --- | --- | --- |
| Unscreened native batching | 3, 3, 3, 3 | 6 | Reference |
| Ordinary screening at 0.05 | 1, 1, 1, 1 | 2 | Bitwise identical |
| Controlled mixed refill, eager | 1, 3, 3, 1 | 4 | Bitwise identical |
| Controlled mixed refill, replay | 1, 3, 3, 1 | 4 | Bitwise identical |

All four native first scores were below 0.05. The **controlled mixed** arms used
a test-only callback override to continue seeds 102 and 103 and stop 101 and 104.
This exercised slot replacement while another job continued; it is not a measured
natural acceptance rate or a production option. Refill rebuilt graphs when slots
changed: the replay arm recorded three captures and 800 replays, with no eager
fallbacks. The final eight CIFs and four completion records had the expected job,
seed and sample ownership. These shared-GPU runs establish correctness on the
fixture, not a clean throughput improvement or the scientific validity of the
cutoff.

A separate full 7r6r native-singleton check at ten recycles, 200 steps and one
sample matched the saved pre-change prediction bitwise across all returned
fields. Both measured repeats also matched. Thus the tested default execution
path was preserved by the sampling refactor.

The corrected fused LayerNorm plus replay matched a saved **fused-eager** 7r6r
prediction bitwise, including confidence, at the same full settings. Each run
captured once and replayed 200 times. This does not imply equality to **Torch**
LayerNorm; the [kernel guide](kernels.md#layernorm) records that separate numerical
comparison.

## Tests and CLI

- Focused screening/refill, scheduling, native batching, config, writer, grouped
  ipSAE, replay, projection-cache, attention, bounded LayerNorm and Triton checks:
  **325 passed**, with the relevant CUDA gates enabled.
- Additional runner reuse, memory-policy, coordination and input-reliability
  checks: **100 passed**. The four CUDA ragged-padding cases initially skipped
  in this CPU invocation were then run with their gate enabled: **4 passed**.
- Additional grouped-interface screening checks, including missing selections
  and full confidence disabled: **4 passed**.
- Fused LayerNorm caller-stream capture/replay across FP32/BF16/FP16 and all
  affine modes: **12 passed**, with the extension required to load successfully.
- Ruff and `git diff --check` passed for the changes.

CPU refill tests include a five-job partial tail, mixed stopping/continuation,
cache enabled/disabled, residue/structural pair outputs, exact-shape and budget
rejection, missing/duplicate results, writer failures, injected OOM and fatal
capture errors. They verify output equality against baselines with fixed batch
size, including tail placeholders. Real CUDA OOM was not induced on the shared
host. Large-input and eight-job refill memory/performance remain unmeasured.

The documented four-job `refill_smoke.json` also passed through the real CLI with
`--cycle 1 --step 4 --sample 3`, both chunks one, cutoff 0.05 and refill enabled.
It produced four CIFs, four summaries and four atomic completion records, each
reporting one completed and two skipped samples. This reduced-step run validates
CLI/output wiring; its structures are not accuracy evidence.
