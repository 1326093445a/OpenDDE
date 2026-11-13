# Mixed-length inference and native ipSAE

## Default policy, 2026-11-13

The supported default configuration is the validated **pre-replay eager
baseline**: BF16 mixed precision with the existing module/length policy,
automatic triangle kernels, reference diffusion attention, existing eager
fusions and bounded conditioning/projection caches. Ordered CUDA aggregation,
native batching, per-job confidence splitting and explicit grouped-ipSAE
selection remain in place.

CUDA replay, efficient attention and TMK3 remain opt-in. Diffusion/Pairformer
compilation and standalone aggregation-plan-cache controls remain benchmark-only;
the reports below do not promote them to default inference. Runtime default
values were already consistent with this policy. CLI help, the README's first
example and tests now make it explicit across configuration and Python entry
points. No precision thresholds, batch limits, sample counts or dependencies
changed. See [the default settings table](inference_instructions.md#default-inference-baseline)
and [the eager example](../README.md#default-eager-batching-example).

## RFD4-inspired whole Pairformer compilation experiment, 2026-11-13

**Keep this benchmark-only: compiling the BF16 trunk changes predictions and
confidence materially.** No production inference flag, backend default,
checkpoint, precision setting or dependency changed. This experiment is
separate from the existing denoiser CUDA replay feature, which does not compile
the trunk.

`scripts/benchmark_inference.py --compile-pairformer-blocks` compiles the 48
main Pairformer blocks individually with `fullgraph=True`, static shapes and
Inductor CUDA graphs disabled. Diffusion and confidence execute eagerly. The
production environment is Torch 2.7.1+cu126 / Triton 3.3.1 / cuEquivariance
0.10.0 with `opendde_v1` (`opendde.pt`), `LAYERNORM_TYPE=torch`, TF32 off
and cuEquivariance triangles.

Unmodified blocks fail Dynamo tracing at an attention `ContextVar` read and
the `ExitStack` used for autocast boundaries. Temporary instance-local wrappers
make those operations explicit without modifying model source or checkpoint
names. Tests confirm the wrappers themselves match eager arithmetic bitwise
in FP32 and BF16, before compilation. CPU, gradients, training, lengths <=16,
unsupported kernel selections and active diffusion attention scopes stay eager;
distributed Fold-CP is rejected. Compiler errors propagate and original methods
are restored on exit, including exceptions. The benchmark requires exclusive
access to the model instance.

### Full-checkpoint findings

The comparisons hold features, seed 101, grouping, precision, 10 recycles,
200 diffusion steps, reference diffusion attention and all model weights fixed.
All inputs use no MSA/templates/RNA MSA. Modes alternate within one loaded
model, with a separate first pass for each mode. Every compiled prediction
executes **480 compiled block calls**, with no eager block fallbacks. All GPUs
were shared; warmed timing comparisons are indicative, not clean speed claims.

For the 245-token 7r6r singleton at BF16, three measured passes per mode gave:

| Measurement | Eager | Compiled |
| --- | ---: | ---: |
| Median model time | 14.70 s | 13.80 s |
| Median summed Pairformer time | 5.25 s | 4.36 s |
| First model pass, including setup | 16.16 s | 59.54 s |
| Warm peak allocated memory | 5.25 GiB | 4.86 GiB |

The median model reduction is about **6.1%**, but the last eager pass dropped to
11.64 s as contention changed. First-use compilation/setup added roughly 45 s
of trunk time relative to the warmed compiled median, with an already populated
on-disk kernel cache. Different static shapes incur additional specialization.

Numerics are the decisive issue: aligned all-atom coordinate RMSD is **9.10 Å**,
mean pLDDT changes **65.513 → 64.240**, pTM **0.7368 → 0.7181**, ipTM
**0.7749 → 0.7679**, and ranking score **0.7673 → 0.7579**. Expected token-pair
PAE differs by up to **8.87 Å**. Eager and compiled each repeat bitwise across
all compared output fields; reproducibility does not establish equivalence.
This comparison changes neither padding nor batch membership.

A separate single-block control with
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` still has BF16 feature differences
(maximum pair-feature difference about 0.0403), so that option is not an exact
fix. This was a block-level control, not a full-prediction benchmark.

The full FP32 7r6r control is much closer: **0.000270 Å** aligned RMSD,
**0.000613 Å** maximum expected-PAE difference, and mean pLDDT difference
**0.000023**. That does not establish equivalence across inputs or justify
changing the default BF16 workload. Two warmed passes per mode gave median
model times of **29.40 s eager / 29.52 s compiled**, so this control shows no
end-to-end gain under the observed shared load.

With three BF16 samples on 7r6r, aligned RMSDs are **8.87 / 2.87 / 2.35 Å** and
the maximum expected-PAE difference is **11.43 Å**. The zero-based sample ranking
order is `[2, 0, 1]` in both modes, but ranking-score values change by up to
**0.0188**. All per-sample confidence axes validate and each mode repeats bitwise.
This single unchanged ranking is not evidence of preserved campaign rankings.
Two warmed passes per mode gave **19.00 s eager / 21.13 s compiled**; the wide
timing variation prevents a useful speed estimate for this case.

The fixed two-job BF16 batch contains 7r6r/7pzb at **245/600 trunk tokens**. Compiled
versus eager aligned RMSDs are **11.45 / 1.83 Å**, with maximum expected-PAE
differences **7.29 / 5.27 Å**. Mean pLDDT differences are **0.410 / 0.414**.
Both jobs retain the correct independent residue/atom and confidence axes.
This comparison holds the padded grouping fixed; it does not compare batched
predictions with singleton predictions. The examples include protein/DNA
complexes and, for 7pzb, ligands. Both modes repeat bitwise across all fields.
Two warmed passes per mode give **103.97 s eager / 102.85 s compiled** for
the batch, only about a 1.1% difference under shared load. Warm peak allocated
memory is **26.00 / 25.60 GiB**. These measurements do not support a useful
general end-to-end speedup claim.

### Validation scope

The focused tests check eager-wrapper equivalence, FP32/BF16 compiled outputs,
static-shape recompilation, repeatability, masked-padding content independence,
batch isolation, state-dict preservation, restored method overrides, eager
fallbacks and compiler failure propagation. Compilation tolerances in the small
tests apply to internal features; full-checkpoint differences above are reported
separately and fail a prediction-equivalence requirement.

The checkpoint probes validate finite prediction tensors, residue/atom axes,
per-job and per-sample confidence splits, and compare all returned fields.
These examples do not specify `ipsae_interface`; their top-level ipSAE is null.
They therefore do **not** establish grouped-ipSAE score or ranking parity.
No eight-job, distributed, training or clean-idle-GPU benchmark was run for
this candidate. Production dependencies and defaults remain unchanged.

Targeted test command (production interpreter, CUDA gates enabled):

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch OPENDDE_RUN_COMPILE_TESTS=1 \
  python -m pytest tests/test_pairformer_compile_experiment.py \
  tests/test_diffusion_compile_experiment.py tests/test_triton_compatibility.py -q
```

The final Pairformer file passed **7 tests**; the adjacent compilation/Triton
files passed **15 tests**. Ruff and `git diff --check` passed, and benchmark help
and conflicting-option rejection were checked. This is targeted validation,
not a claim that the entire repository test suite was run.

### Reproduce the prediction comparison

From the checkout in the production environment, use separate artifact folders:

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r \
  --output /tmp/pairformer-eager --dtype bf16 --trimul-kernel cuequivariance \
  --cycles 10 --steps 200 --samples 1 --repeats 3

CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r \
  --output /tmp/pairformer-compiled --dtype bf16 --trimul-kernel cuequivariance \
  --cycles 10 --steps 200 --samples 1 --repeats 3 \
  --compile-pairformer-blocks --compare /tmp/pairformer-eager/last.pt
```

Reports retain all-field comparisons, repeat checks and cumulative
`pairformer_compilation` counters. These commands run the modes in separate
processes; the timing experiment above alternated them within one process.
Use `--names 7r6r 7pzb` for the mixed-length batch, `--samples 3` for multiple
samples, or `--dtype fp32` for the precision control. Do not combine this flag
with diffusion compilation, CUDA replay or the aggregation-cache experiment.

## RFD4 fused pair-attention compatibility experiment, 2026-11-13

**The proposed direct port is not suitable for the current OpenDDE model and
production environment.** This experiment stops at the compatibility gate;
no prediction backend, precision setting, dependency pin or default changed.
It corrects the earlier suggestion that RFD4's fused pair attention would be
the first straightforward kernel port to try.

The production interpreter has Torch **2.7.1+cu126** and cuEquivariance **0.10.0**.
RFD4's implementation calls the **0.13** API accepting raw single/pair features
and normalization/projection weights. The installed 0.10 API instead takes
already projected Q/K/V; its gate and output projection remain separate Torch
operations. A 0.13 wheel overlay failed to import its attention operation under
Torch 2.7.1 because it requires newer FX APIs; the wheel declares Torch >=2.11.
An isolated **Torch 2.11.0+cu126 / cuEquivariance 0.13.0** environment imported
successfully and ran the GPU checks below. Production packages were untouched.

Three model-level constraints remain even after that version change:

- OpenDDE's main Pairformer uses **384-wide pair features with LayerNorm**.
  The vendor's whole-operation LayerNorm flash kernel accepts pair widths
  **128 or 256**, not 384. Its separate RMSNorm width-384 route does not preserve
  this model's normalization.
- OpenDDE's diffusion uses **adaptive LayerNorm**, with a condition-dependent
  scale and shift. The RFD4 API unconditionally applies an ordinary single-input
  normalization before projecting Q/K/V. Passing OpenDDE's adaptive output into
  it would normalize a second time and change the computation. A synthetic
  check with nonzero conditioning weights confirms this is not an identity.
- The tested OpenDDE lengths use **FP32 diffusion** under the existing BF16
  configuration. The vendor's whole-operation flash path requires BF16/FP16.
  Its fixed mask constant and raw-pair/cache requirements impose further
  restrictions; changing those settings is not a precision-preserving port.

### GPU compatibility checks

The diagnostic uses OpenDDE's own `AttentionPairBias` with nonzero output
weights, fixed seeds, TF32 off and an H200. It instruments actual vendor
dispatch, rather than treating a successful API call as evidence of fusion.

| Case | Batch / tokens | Single / pair width | Precision | Whole-operation flash route |
| --- | --- | --- | --- | --- |
| Trunk | 1 / 245 | 384 / 384 | FP32 | No; native Torch fallback |
| Trunk | 1 / 245 | 384 / 384 | BF16 autocast | No; native Torch fallback |
| Padded trunk batch | 2 / 320 | 384 / 384 | BF16 autocast | No; native Torch fallback |
| Large trunk | 1 / 1218 | 384 / 384 | BF16 autocast | No; native Torch fallback |
| Diffusion-shape control with ordinary LayerNorm | 1 / 475 | 768 / 128 | FP32 | No; native Torch fallback |
| Supported vendor-shape control | 1 / 512 | 256 / 128 | BF16 autocast | Yes, with vendor-default mask constant |

The ordinary-LayerNorm diffusion control isolates shape/dtype support; it is
**not** an adapter for OpenDDE's actual adaptive-normalized diffusion. The
supported control confirms that the installation can execute the flash kernel.
Unmasked tests with the vendor-default mask constant still left all five
OpenDDE-shaped cases on native fallback, independently of the masking penalty.
The supported control's cached-bias call selected the pair-bias/SDPA route,
not the whole-operation flash route. A component API call does not by itself
prove that its internal projection kernel is fused.

All six cases check output shapes through tensor comparison, finite values,
bitwise repeats, projected-bias reuse, key padding, and (for the two-job case)
a fully masked job. Operator differences are checked against FP32 `3e-6` /
BF16 `0.01` maximum absolute tolerances. These are feature-value tolerances,
not coordinate or confidence bounds. The probes use synthetic tensors, not
checkpoint predictions. **No full-prediction, confidence-score/ranking or
end-to-end speedup claim is made for this candidate**: there is no admitted
whole-operation path to validate under the existing model semantics.

### Reproduce

Use a separate environment with Torch >=2.11, cuEquivariance 0.13 CUDA packages
and OpenDDE's remaining dependencies; the repository's production pins remain
2.7.1/0.10.0. With that environment active, run from the checkout:

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python scripts/probe_fused_pair_attention.py \
  --output /tmp/opendde_fused_pair_attention.json
```

The script checks the 0.13 API and SM90 device, asserts dispatch for all six
cases, restores its temporary instrumentation, and records versions, warnings,
comparisons and its source hash. It rejects the production 0.10 API before
running the probes. Vendor route instrumentation uses private implementation
names and intentionally targets 0.13; it is not part of prediction execution.

Further work would need a kernel accepting OpenDDE's pre-normalized adaptive
features and cached arbitrary pair bias, or a separate Pairformer compilation
experiment. Neither is implemented or validated by this compatibility study.

## Complete-denoiser replay and whole-block compilation, 2026-11-10

These are separate experiments. **Raw CUDA replay does not use `torch.compile`**;
the compiler experiment uses full token-transformer blocks and explicitly
disables compiler-managed CUDA graphs. Both retain the existing padding and
precision policy. Results below use cached released weights, no MSA, BF16
configuration (FP32 diffusion for these lengths), and fixed grouping. All GPUs
were shared, so timing magnitudes need an idle-GPU confirmation.

### Request-scoped denoiser replay

The optional `--diffusion_cuda_graph true` prediction flag captures
`_denoise_batch`: diffusion conditioning, atom encoder, token transformer, atom
decoder and coordinate prediction. Random draws, CPU/SciPy rotations, Euler
updates, trunk and confidence stay eager. Three side-stream warmups prepare
request caches before capture. Each replay copies coordinates and noise levels
into persistent buffers and clones outputs so later replay cannot overwrite a
caller's retained tensors. Only one graph is retained: sample-chunk shape
changes replace it, and request completion releases it before confidence.
A capture stream is reused per thread/device. Creating a fresh stream per
request retained an additional cuBLAS workspace (about 32 MiB on this H200);
the final implementation reuses that stream while keeping graph buffers and
conditioning request-local. A repeated-capture test includes GEMM specifically
to check that tensor storage does not accumulate across closed requests.

The initial capture failed at CUDA `bincount` in atom aggregation. Warmup now
prepares stable sorting indices and segment lengths once per immutable topology.
The cached lengths are nonnegative and sum to the atom count by construction;
the existing ordered `segment_reduce` then skips repeated validation, which
otherwise synchronizes. The arithmetic stays sum-then-divide. Plans hold their
source tensor and are cleared with the request. Uncached eager aggregation is
unchanged. No graph/cache survives into another complex's conditioning.

The flag defaults off. It selects native no-MSA batching even for batch size
one; compare against **native eager**, not legacy serial inference. CPU,
gradient and deterministic-algorithm calls stay eager, as do requests lacking
prepared static pair conditioning. Unsupported native batching combinations
are rejected before inference. Capture errors, including capture-time OOM,
raise `DenoiserGraphCaptureError` and stop the worker. Both the batch scheduler
and the CLI's input-file loop propagate this error immediately: they neither
split/retry the batch nor move on to another job or file. The original exception
is retained as its cause. Restart the worker with replay disabled; the observed
Torch 2.7 capture failure invalidated CUDA state, so eager recovery in that
process would be unsafe. OOM before capture (including eager warmup) retains
the ordinary scheduler's split-and-retry behavior. Failure-injection tests cover
capture entry, body and exit, plus warmup, batch scheduling and the CLI file loop.
The caller stream waits for queued warmup work even if warmup raises, before
retry can reuse allocator blocks.

On complete **7r6r, ten recycles, 200 steps, one sample**, each prediction
recorded **one capture and 200 replays** on both Torch 2.7.1 and 2.11.0. For
each version, all returned outputs matched that version's native eager baseline
**bitwise**, including confidence and mappings; repeated graph predictions also
matched bitwise. A **7r6r/7pzb** batch at one recycle, 20 steps, one sample and
efficient attention matched its 2.7.1 eager baseline bitwise across all outputs,
with one capture and 20 replays per batch. Peak allocated memory across those
short measurements was approximately **5.13–5.22 GiB** for the singleton and
**25.27–25.51 GiB** for the pair; these are not large-queue memory bounds.

The capture-failure review added **123 passing tests** under production Torch
2.7.1 with the graph CUDA gate enabled, covering graph execution, injected
capture/warmup failures, scheduling and config/CLI behavior. Capture failures
were injected rather than deliberately exhausting device memory. Ruff lint,
format checks and `git diff --check` passed for the changes.
The CLI shutdown regression also injects an allocator-cleanup error through
the real `InferenceRunner.close()` path: existing best-effort cleanup logs that
secondary error and preserves the original capture exception. Both shutdown
cases passed after the review.

The follow-up large-input check used complete **7wux: 1,218 residue tokens,
2,300 structural tokens and 9,142 atoms**, ten recycles, 200 steps and one
sample. Native eager and two replay predictions ran in one process on a shared
H200, with efficient attention and the default 2,048 MiB cache budget. Both
replay predictions recorded **one capture, 200 replays and zero eager calls**.
All returned fields, including confidence and atom/chain mappings, matched eager
bitwise; replay also repeated bitwise. Peak allocated memory was **52.237 GiB
eager / 52.238–52.271 GiB replay**, with post-request resident allocation stable
at **2.573 GiB** after the first replay. Capture setup was about **1.17 s**.
These are allocated-memory measurements, not bounds on reserved memory or
other processes' allocations. Elapsed model times were **287.05 s eager /
285.10 and 286.30 s replay**. This was a correctness/memory check without a
separate timing warmup, under shared load; it does not demonstrate a large-input
speedup or qualify eight-job capture. The BF16 configuration retained its
existing FP32 diffusion/confidence policy for this input.

A full **7r6r three-sample** check used the same ten recycles, 200 steps and
efficient attention, with `infer_setting.sample_diffusion_chunk_size=2`. The
two-sample chunk followed by a one-sample tail recorded **two captures, 400
replays and no eager calls** per prediction. Both replay predictions matched
the native eager run bitwise across all fields, including per-sample confidence;
the replay repeats also matched bitwise. Peak allocated memory was **5.152 GiB
eager / 5.152–5.156 GiB replay**. This validates sample-chunk shape replacement
in a complete prediction, not just the small graph unit test. Eight-job capture
and clean idle-GPU throughput remain unverified.

### Whole token-transformer blocks

`scripts/benchmark_inference.py --compile-diffusion-blocks` compiles the 24 token
blocks only. Its explicit tensor path uses the original normalization,
projections, gates, residuals and transition, with an already projected bias
and an explicit attention-backend argument. This avoids monkeypatching the
global attention-context getter. Each block receives its own Dynamo code object
without raising process-global recompile limits. The eager first diffusion step
populates the request's pair-bias cache; subsequent steps use compiled blocks.
CPU, training, missing-cache and unsupported attention paths remain eager.
The projected-bias cache must fit: the default 2 GiB budget did **not** fit the
7r6r/7pzb pair, and that diagnostic run reported zero compiled calls. Such a run
is an eager fallback, not evidence of compiled-batch correctness or speed.
`--diffusion-cache-max-mb` in the benchmark exposes the existing cache budget;
use the same value for eager and compiled arms and inspect actual call counts.

This experiment currently supports **FP32 diffusion only**. A BF16-autocast
test exposed another Torch 2.7 tracing failure in the LayerNorm autocast context;
that mode explicitly stays eager and is tested. No precision override is used.
The normal BF16 configuration on the tested examples still qualifies because
its existing diffusion policy is FP32. CUDA graphs are disabled for this arm;
there is no inference CLI flag or production recommendation for compilation.

On complete 7r6r at ten recycles, 200 steps and one sample, 24 modules served
**14,328 compiled calls and 72 eager first-step calls** across warmup and two
measured predictions. Repeated outputs were bitwise identical. Against native
eager, maximum coordinate-component difference was **0.0009804 A**, unaligned
component RMS **0.0001116 A**, maximum expected-PAE difference **0.0003023 A**,
and mean pLDDT difference **0.00001526**. All confidence/mapping keys and shapes
were compared, extending the coordinate-only scratch experiment supplied in
the review. These values do not establish universal numerical equivalence.

The mixed-length compiler check required **8,192 MiB** of projection-cache
budget in both arms. In one loaded model, complete 7r6r/7pzb ran with efficient
attention, one recycle, 20 steps and one sample, one warmup plus two measured
passes per arm. All 24 blocks compiled; three compiled predictions served
**1,368 compiled calls and 72 eager first-step calls**. Each mode repeated
bitwise, and confidence keys/shapes remained correctly split per job.

| Input | Max coordinate-component difference | Max expected-PAE difference | Mean pLDDT difference |
| --- | ---: | ---: | ---: |
| 7r6r | 0.005082 A | 0.15568 A | 0.003624 |
| 7pzb | 0.000872 A | 0.22637 A | 0.001175 |

The larger confidence differences are a reason to retain this as an experiment,
not infer quality equivalence from coordinates alone. This reduced-step batch
does not establish production-setting accuracy. Under shared load, its median
model time was **15.232 s eager / 15.068 s compiled**, and diffusion time was
**2.254 / 2.121 s**. Peak allocated memory was about **25.83 GiB**. An earlier
GPU-0 attempt ran out of memory in the existing trunk after another process
allocated about 105 GiB; it was excluded, and the check was rerun on GPU 2.

### Separate aggregation-cache control

`--cache-atom-aggregation` is a benchmark-only control that reuses the same
sorting and segment plans while leaving the denoiser eager. It lets a replay
comparison distinguish topology reuse from reduced launch overhead. It restores
the sampler and request state on exit, including exceptions. It does not alter
the default prediction path. Compilation, replay and cache-only controls should
first be measured separately on fixed shapes.

```bash
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r --output /tmp/eager
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r --output /tmp/replay \
  --diffusion-cuda-graph --compare /tmp/eager/last.pt
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r --output /tmp/blocks \
  --compile-diffusion-blocks --compare /tmp/eager/last.pt
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r --output /tmp/aggregation \
  --cache-atom-aggregation --compare /tmp/eager/last.pt
```

### Alternating four-way comparison

One loaded model on shared H200 GPU 2 ran 7r6r with ten recycles, 200 steps,
one sample and reference attention. After a warmup per arm, three passes per
arm were interleaved in forward/reverse order. Model time excludes loading,
featurization and writing; diffusion time includes per-request graph setup.

| Arm | Median model time | Median diffusion time |
| --- | ---: | ---: |
| Native eager | 14.690 s | 7.783 s |
| Aggregation plans only | 13.210 s | 6.307 s |
| Whole token blocks, no CUDA graphs | 13.889 s | 6.967 s |
| Aggregation plans plus raw denoiser replay | 12.632 s | 5.697 s |

Every arm repeated bitwise. Aggregation-only and replay matched **all** eager
outputs bitwise; whole blocks had the small coordinate/confidence differences
reported above. This independently reproduces the larger-block improvement and
shows that topology reuse contributes substantially to the raw-replay result.
It does not measure compilation plus replay together. These timings preceded
the capture-stream reuse fix; the trace's growing cuBLAS workspaces prompted
that fix and the GEMM allocation regression test. Shared-GPU timings are
indicative, not a clean speed guarantee or an eight-GPU scaling result.

After the stream-reuse fix, a fresh same-process eager/replay comparison ran
one warmup per arm followed by **eager, replay, replay, eager** at the same
7r6r production settings. Median model time was **14.683 s eager / 12.446 s
replay** (15.2% lower); diffusion time was **7.766 / 5.528 s** (28.8% lower),
including capture/setup on every graph prediction. Each graph prediction used
one capture and 200 replays. All outputs matched eager bitwise, and each mode
repeated bitwise. After the first workspace allocation, measured resident
memory stayed around **2.764 GiB**, rather than growing by 32 MiB per capture;
peak allocation was around **5.254 GiB**. This final check also used a shared
GPU and only two measured passes per arm, so the speed magnitude remains
provisional. No idle H200 was available in this session.

### Isolated Torch 2.11 compatibility check

A temporary venv inheriting the production environment's non-Torch dependencies
was tested with **Torch 2.11.0+cu126 / Triton 3.6.0**. Production remains
**Torch 2.7.1+cu126 / Triton 3.3.1**. No production environment or dependency pin
was changed. The existing cuEquivariance 0.10.0 and opt_core 0.5.228.0 remained
installed. Torch 2.11 imports initially picked an older system `libcudart`; an
explicit library path to the matching CUDA 12.6 runtime fixed the isolated run.

The 2.11 eager and replay runs completed full 7r6r inference and repeated
bitwise within each mode. Eager coordinates and expected PAE also matched
2.7.1 bitwise on this input; some other confidence/contact outputs changed
slightly (maximum raw pLDDT-logit difference **1.67e-6**). This is an input-level
compatibility check, not proof that all version changes are numerically neutral.
The 2.11 GPU suite passed **64 tests plus 102 subtests**, including graph RNG,
ownership, tail shapes and allocation cleanup, ordered aggregation, projection
caches, TMK3 width/mask/dtype/residual tests, and triangle compatibility.
The final stream-reuse tests were repeated on 2.11: **7 passed**. The production
2.7.1 graph/compiler/cache/aggregation suite passed **60 tests**; the broader
CPU config/scheduler/attention run passed **208 tests, 37 opt-in CUDA skips**.
These suites overlap and their counts must not be added as unique tests. Ruff,
format checks, CLI help, and `git diff --check` also passed.

The package still pins Torch 2.7.1, so the experimental venv intentionally has
a dependency-version conflict with installed OpenDDE metadata. A production
migration needs a deliberate packaging change and broader workload validation;
NCCL/Fold-CP, every external extension, and long BF16-diffusion complexes have
not been validated on 2.11. Official CUDA wheel choices are documented in the
[PyTorch 2.11 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0).

## Diffusion compilation experiment, 2026-11-10

The RFD4-Proteina implementation motivated a block-level compilation experiment
in the existing Torch 2.7.1+cu126 environment. Compiling an unmodified complete OpenDDE
diffusion block with `fullgraph=True` fails at the request-local attention
`ContextVar.get()` lookup. The conditioned feed-forward transition compiles
without changing attention, sampling, precision, or padding.

`scripts/benchmark_inference.py --compile-diffusion-transitions` temporarily
compiles **only transitions under `model.diffusion_module`**. It uses Inductor,
`fullgraph=True`, static shapes, and explicitly disables CUDA graphs. The
surrounding Python sampler, CPU/SciPy rotations, projection caches, atom
aggregation, trunk and confidence remain eager. CPU, training and gradient
calls remain eager. The wrapper preserves checkpoint keys, restores methods
on exit, and raises compiler failures rather than silently timing a fallback.
Reports include successful compiled-call counts and the benchmark source hashes.

This option belongs to the **benchmark script**, not the prediction CLI. It is
not a production default or a validated CUDA-graph implementation. Start a new
benchmark process when changing model weights or devices. New shapes can incur
compilation and compiler cache limits; no automatic padding buckets or global
Dynamo cache-limit changes are introduced.

Example with cached assets and the production interpreter:

```bash
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r \
  --output /tmp/opendde_eager --repeats 3
LAYERNORM_TYPE=torch python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r \
  --compile-diffusion-transitions --output /tmp/opendde_compiled \
  --compare /tmp/opendde_eager/last.pt --repeats 3
```

Both commands retain BF16 configuration, ten recycles, 200 steps and one sample
by default. For these input lengths the existing precision policy still runs
diffusion in FP32. Warmup includes any compilation/cache loading and is recorded
separately from measured passes. Use the same grouping and attention backend
for both arms; compare confidence and mapping outputs as well as coordinates.

### Initial correctness checks

On complete 7r6r at the settings above, all **30 diffusion transition modules**
compiled, with **18,000 successful compiled calls** across a warmup and two
measured predictions. Eager and compiled modes each repeated bitwise. Against
eager execution, compilation changed coordinates by at most **0.005432 A per
component** (unaligned component RMS **0.0001424 A**), expected PAE by
**0.0006285 A**, and
the summary mean pLDDT by **0.00000763**. No shape/key mismatch occurred in the
comparison of all returned predictions, including confidence and mappings.
Peak allocated memory was **5.158 GiB** in both modes. This is numerical
agreement on one input, not bitwise equivalence or a general quality result.

An initial broader scratch probe also compiled three transitions outside the
diffusion module and changed a coordinate component by **19.47 A**. That probe
is excluded from the diffusion-only results and is why the saved experiment
selects the diffusion subtree explicitly rather than replacing the transition
class globally.

The initial separate-process measured times were **15.09 / 11.08 s** eager and
**15.20 / 15.17 s** compiled. All GPUs were occupied by other jobs, and the
baseline itself varied substantially. These numbers establish **no speedup**;
they are not a clean performance comparison. The compilation warmup also used
an existing scratch Inductor cache and is not a cold-cache startup benchmark.

A second check batched complete 7r6r and 7pzb (**245 and 600 public tokens**),
using the existing `efficient` attention backend in both arms, **one recycle,
20 steps, one sample**, and the same BF16 precision policy. Each mode repeated
bitwise, and all per-job output keys and shapes agreed, including confidence
splits. Thirty modules served **2,160 compiled calls** over three predictions;
atom transitions run separately for each job while token transitions batch them.

| Input | Maximum coordinate-component difference | Maximum expected-PAE difference |
| --- | ---: | ---: |
| 7r6r | 0.003974 A | 0.0002108 A |
| 7pzb | 0.001070 A | 0.0003912 A |

Peak allocated memory was approximately **25.45 GiB** in both arms. This
reduced-step test validates mixed-length execution and composition with the
efficient attention backend; it is not a production batch-throughput result.

### Alternating comparison and decision

A single loaded model then ran one warmup per mode followed by six measured
7r6r predictions in **eager, compiled, compiled, eager, compiled, eager** order.
The settings remained ten recycles, 200 steps, one sample, BF16 configuration
and reference attention. Synchronization bracketed the sampler in both arms
to measure diffusion separately. The medians of three measured passes were:

| Mode | Model time | Diffusion sampling time |
| --- | ---: | ---: |
| Eager | 14.704 s | 7.759 s |
| Compiled diffusion transitions | 14.871 s | 7.979 s |

Compilation was **1.1% slower overall** and **2.8% slower in diffusion** in this
shared-GPU session. Both modes remained bitwise repeatable, and the
eager/compiled coordinate difference matched the separate-process check.
This is not an isolated-GPU estimate, but it provides no reason to recommend
transition compilation for the tested workload. The experiment remains in the
benchmark script; **no prediction flag, precision change or default change was
made**. Larger compiled regions or CUDA graph replay require a separate
experiment, including resolving attention context outside the compiled region
and checking graph buffer lifetimes. No Torch upgrade was required here.

The production interpreter ran the focused compiler and projection-cache tests
on CUDA: **26 passed**. Coverage includes FP32/BF16, cached/uncached transition
conditioning, repeatability, retained output ownership, eager CPU/gradient
paths, checkpoint keys, and restoration after compiler failure. The compiler
tests require `OPENDDE_RUN_COMPILE_TESTS=1`; cache CUDA checks additionally need
`OPENDDE_RUN_BATCH_KERNEL_TESTS=1`:

```bash
LAYERNORM_TYPE=torch OPENDDE_RUN_COMPILE_TESTS=1 \
  OPENDDE_RUN_BATCH_KERNEL_TESTS=1 python -m pytest \
  tests/test_diffusion_compile_experiment.py \
  tests/test_diffusion_projection_cache.py -q
```

## TMK3 follow-up, 2026-11-05

The optional Mol-1 adapter is now ported as
`--trimul_kernel flashpairformer`. The existing **Torch 2.7.1+cu126 / Triton
3.3.1** environment works with **opt_core 0.5.228.0**; only the optional wheel
was installed, without dependency upgrades. Source-checkout loading through
`OPT_CORE_DIR` is also supported. `auto`, triangle attention and precision
defaults are unchanged. See [kernel setup and fallback rules](kernels.md#optional-tmk3-triangle-multiplication).

OpenDDE's own CUDA tests passed **13 tests and 102 subtests**, covering both
triangle directions, widths including **384**, FP32/BF16 residency under BF16
autocast, residuals, masks, batched inputs, weight reloads, fallback conditions
and error propagation. The subsequent CPU/config/CLI regression run passed
**134 tests with 10 opt-in CUDA checks skipped**, plus 36 subtests. These
suites overlap; do not add their counts as unique tests.

Complete 7r6r with the released checkpoint, no MSA, BF16, **10 recycles, 200
steps and one sample** produced **bitwise-identical predictions to
cuEquivariance**, including confidence and mappings. Repeated TMK3 runs also
matched exactly. Counters recorded **1,208 TMK3 calls and 8 FP32 fallbacks per
prediction**, so this is actual TMK3 execution with the confidence precision
policy preserved. Packed weights are cached without changing checkpoint keys.
Peak allocated memory in this check rose from about **5.16 to 5.47 GiB**.

A second released-checkpoint check batched complete 7r6r and 7pzb (**245 and
600 tokens**) with one recycle, 20 steps and one sample each. TMK3 and
cuEquivariance again matched **bitwise across all compared per-job outputs**,
including confidence splits and mappings, and repeated exactly. Each batch
served 256 TMK3 calls with 16 FP32 fallbacks. This reduced-step check validates
mixed-length execution, not production throughput or eight-input capacity.

GPU 7 was initially free but another job appeared before the first end-to-end
measurements. Those initial timings are excluded from speed claims. The supplied
shared-GPU width-384 kernel ratio is therefore not an OpenDDE end-to-end
speedup. A subsequent clean comparison is reported below. The installed
cuEquivariance Hopper tuning table for its fused dual-projection forward
kernel contains `(N,K)` width pairs `(128,128)` and `(256,128)`, not width
384; this motivates an isolated on-demand-tuning experiment without proving
the cause of the entire observed kernel gap.

The bounded cuEquivariance alternative was also exercised on **245 tokens,
width 384 and BF16 compute with FP32-resident input**. Its two fused-projection
shapes missed the default cache. `CUEQ_TRITON_TUNING=ONDEMAND` populated a
separate cache in about **98 seconds**; a subsequent process found both entries
and matched default cuEquivariance **bitwise in both directions**. Each case
used a separate fixed seed because tuning can consume random state. This
establishes compatibility and parity for the tested shapes, not a speedup:
the GPU was shared, and no production tuning cache or default was changed.

### Matched whole-model kernel comparison

A subsequent session used an otherwise idle H200 with the model code preserved
in commit `2b9c32a`. It ran complete **7r6r, 245 tokens, BF16, no MSA, ten
recycles, 200 steps and one sample**, using a native singleton, reference
diffusion attention, cuEquivariance triangle attention and TF32 disabled.
Torch 2.7.1+cu126, Triton 3.3.1 and the released checkpoint were unchanged.

A fresh isolated cuEquivariance cache was prepared by running the complete
workload with `CUEQ_TRITON_TUNING=ONDEMAND`. Its first forward took **392.33 s**,
including tuning and inference. Tuning was disabled for all measured runs;
the tuned variant read that prepared cache. Default cuEquivariance and TMK3
used a separate untuned cache, including TMK3's eight FP32 confidence fallbacks.

The measured process order was **default, TMK3, tuned, tuned, TMK3, default**.
Each process received one warmup and three measured forwards, giving six
measurements per variant. All model/runner source hashes matched. Continuous
GPU-process sampling produced **660 samples over the session**, with no
competing process detected on the target GPU; other GPUs remained busy.
Model loading, featurization, JIT/tuning setup and output writing are excluded
from measured `predict_batch` time.

| Triangle multiplication | Median | Measured range | Reduction vs default | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| Default cuEquivariance | 8.006 s | 7.974–8.022 s | — | 5.158 GiB |
| TMK3 (`flashpairformer`) | 7.605 s | 7.585–7.660 s | 5.01% | 5.472 GiB |
| Tuned cuEquivariance | 7.636 s | 7.513–7.706 s | 4.62% | 5.158 GiB |

TMK3's median lead over tuned cuEquivariance is only **0.031 s (0.41%)**,
smaller than the observed timing variation. This does **not** establish a
decisive end-to-end winner or the anticipated double-digit gain. The earlier
isolated 640-token kernel ratio cannot be transferred to this 245-token
whole-model workload.

TMK3 matched all default predictions bitwise. Tuned cuEquivariance preserved
coordinates and ranking exactly but changed 11 confidence-related fields at
small FP32 rounding scales: expected PAE differed by at most **0.00000954 A**,
and raw confidence logits by at most **0.00000287**. Each variant repeated
bitwise, including across its two processes. The isolated BF16 kernel parity
check above did not cover every FP32 confidence kernel tuned by this full run.

**Recommendation:** retain `auto` as the general default. On this tested H200
workload, either option provides a modest improvement. TMK3 is an opt-in choice
when retaining default numerics and avoiding on-demand tuning setup matter;
it needs opt_core and about 0.31 GiB more allocated memory here. A warmed
cuEquivariance cache provides similar steady-state speed without opt_core.
Neither result establishes the best choice for larger complexes, unseen
lengths, eight-input batches or multi-GPU throughput.

To measure the tuned alternative, first run the complete workload once with
`CUEQ_TRITON_TUNING=ONDEMAND CUEQ_TRITON_CACHE_DIR=/path/to/separate/cache`,
then unset `CUEQ_TRITON_TUNING` and repeat with the same cache directory.
Measure new shapes separately; cache preparation is not free inference time.

```bash
python scripts/benchmark_inference.py --input examples/cofold/example.json \
  --names 7r6r --trimul-kernel cuequivariance --output /tmp/opendde_cueq
python scripts/benchmark_inference.py --input examples/cofold/example.json \
  --names 7r6r --trimul-kernel flashpairformer --output /tmp/opendde_tmk3 \
  --compare /tmp/opendde_cueq/last.pt
```

Run on a free GPU with cached assets. Reports record dependency versions and
cumulative TMK3/fallback counters in addition to predictions and process
snapshots. First-run JIT compilation belongs to warmup, not steady-state timing.

## Inference optimization follow-up, 2026-11-04

This pass ports stable CUDA atom aggregation, selected multi-chain ipSAE,
native singleton execution, and optional padding/work limits from Mol-1's
approach. The optional efficient-attention implementation also flattens
compatible leading dimensions while keeping shared pair biases as views;
partially shared biases retain separate calls. No model runtime dependencies
were added.

The production optimization target is the repository's default **BF16 mixed
precision**, with the requested **ten recycles, 200 diffusion steps and one
sample per complex**. The initial speed measurements used explicitly requested
**FP32**; a subsequent free-GPU BF16 measurement is reported separately below.
The runner keeps diffusion and confidence outside autocast up to 2,560 tokens;
above that confidence uses autocast, and above 3,840 tokens diffusion does too.
Efficient attention remains opt-in (`reference` is
the default). Confidence chunks remain one: increasing the chunk cannot help
a one-sample request. TMK3 was deferred during the FP32 pass because the available
adapter requires BF16 compute; the subsequent BF16 port and validation are
reported above.
The legacy CLI `--batch_size 1` remains serial; a singleton
produced by the batched scheduler now receives native batching optimizations.

### BF16 correctness follow-up

The released-checkpoint BF16 check used the same complete 7r6r input, ten
recycles, 200 steps, one sample, seed 101 and cuEquivariance, with one warmup
and two measured passes. The optimized native singleton repeated bitwise
across every compared prediction field; all 30 floating output tensors were
finite. A native batch of three synthetic complexes (64, 96 and 128 residue
tokens) also repeated bitwise, with finite outputs and correctly split
coordinates, confidence and atom/residue axes. It used the same sampling
settings and one sample per input. These checks do not establish eight-input
capacity or accuracy across a representative dataset.

The original BF16 singleton differed between its own two measured passes:
maximum coordinate-component difference **3.386 A**, component RMS **1.094 A**,
and expected-PAE maximum difference **1.509 A**. Original versus optimized
BF16 predictions differed by **7.280 A all-atom RMSD after rigid alignment**
(unaligned component maximum **21.922 A**). This is a substantial numerical
compatibility change, despite the much smaller FP32 differences below.
Repeatability does not establish structural accuracy, and these BF16 changes
need representative quality evaluation before treating them as a drop-in
numerical replacement.

An isolated baseline with **only** the stable atom-reduction change also
repeated exactly. It differed from the original baseline by **7.280 A aligned
all-atom RMSD**, but from the optimized native singleton by only **0.000133 A**
(maximum coordinate-component difference **0.001587 A**, expected-PAE maximum
difference **0.000246 A**). Thus, on this input the large shift is attributable
to changing atom aggregation, while switching singleton execution with the
same reduction contributes a much smaller difference. This supports the
reproducibility fix, not a claim of unchanged or improved structural accuracy.

All GPUs were busy during these BF16 checks. Their timings are excluded from
performance claims; the initial clean FP32 speedup must not be extrapolated to
BF16. The benchmark now defaults to BF16 to match the production precision
selection. It preserves the runner's length-dependent FP32 exceptions for
diffusion and confidence rather than forcing every module to BF16.

### Clean BF16 real-input measurement

GPU 7 subsequently became free. The preserved original source and current
code were each measured on complete 7r6r with BF16, ten recycles, 200 steps,
one sample, seed 101 and the same cuEquivariance/reference-attention settings.
Each configuration used one warmup and two measured passes. GPU process
snapshots for every pass contained only that benchmark process on the target
GPU; other GPUs on the host remained busy.

| Configuration | Measured seconds | Median | Peak allocated memory |
| --- | --- | ---: | ---: |
| Original singleton fallback | 8.294, 8.290 | 8.292 s | 4.991 GiB |
| Stable aggregation + native singleton | 8.018, 7.935 | 7.976 s | 5.158 GiB |

This is **3.81% lower model time** on this input, with about 0.17 GiB additional
peak allocation. The optimized fixed-shape runs repeated bitwise; the original
runs again differed. The configurations were sequential, not alternating,
and the small sample does not establish a general BF16 throughput improvement.
Timing boundaries match the FP32 measurement below.

### BF16 singleton versus unequal-length pair

The complete real examples 7r6r and 7pzb contain **245 and 600 residue-level
tokens** after featurization, including DNA and ligands. Using the same BF16
settings as above, the queue ran as separate native singletons and then as one
native pair. Each configuration had one warmup and two measured passes. All
target-GPU process snapshots contained only the benchmark process.

| Execution | Measured queue seconds | Median queue time | Median seconds/input | Peak allocated memory |
| --- | --- | ---: | ---: | ---: |
| Separate native singletons | 34.572, 35.138 | 34.855 s | 17.427 s | 15.027 GiB |
| One native pair | 53.680, 53.790 | 53.735 s | 26.867 s | 25.447 GiB |

Separate singletons used **35.14% less queue time** for this unequal-length
pair. This supports limiting padding for this queue, not a universal claim
that batching is slower. Equal-length pairs, larger queues and eight-GPU
throughput were not measured. Both executions repeated bitwise within their
fixed grouping and produced correctly split per-job outputs.

Grouping also changed the shorter input's prediction: 7r6r differed by
**7.544 A aligned all-atom RMSD** between singleton and pair execution, while
the larger 7pzb input differed by **0.0000362 A**. This is consistent with the
separate BF16 padding control and makes fixed grouping essential when
comparing predictions.

```bash
python scripts/benchmark_inference.py --input examples/cofold/example.json \
  --names 7r6r 7pzb --execution singletons --output /tmp/opendde_bf16_singletons
python scripts/benchmark_inference.py --input examples/cofold/example.json \
  --names 7r6r 7pzb --execution batch --output /tmp/opendde_bf16_pair \
  --compare /tmp/opendde_bf16_singletons/last.pt
```

### FP32 real-input measurement

The complete repository `7r6r` example contains one protein and two DNA chains,
245 residue-level tokens. Tests used the released `opendde.pt`, H200, Torch
2.7.1+cu126, cuEquivariance, PyTorch LayerNorm, no external features and no TF32.
Each configuration received one full warmup and two measured passes. Times
cover `predict_batch`, including its feature transfer, and exclude model loading,
featurization, CPU comparison and CIF/JSON writing. The original source was
preserved before edits. Process snapshots showed no other process on this GPU
during these two configurations; other GPUs on the host were busy.

| Configuration | Measured seconds | Median | Peak allocated memory |
| --- | --- | ---: | ---: |
| Original singleton fallback | 15.333, 15.358 | 15.346 s | 6.356 GiB |
| Stable aggregation + native singleton, reference attention | 14.933, 14.979 | 14.956 s | 6.679 GiB |

The observed model-time reduction is **2.54%**, at about 0.32 GiB additional
peak allocation. These were sequential configurations, not an alternating
sweep; two passes on one input do not establish a universal speedup or an
optimal batch size. No new eight-GPU scaling measurement was made.

The original fixed-seed FP32 runs differed in 24 tensor outputs, including
coordinates (maximum component difference 0.000832 A). The new native singleton
repeated bitwise across all compared tensors, arrays and summary values.
Original-versus-new maximum coordinate-component difference was 0.002270 A
(component RMS 0.000150 A). This is a numerical compatibility change, not an
accuracy benchmark. This example has only one protein chain, so top-level
protein-interface ipSAE is null; selected protein interfaces have separate
end-to-end regression tests.

A Mol-1 process subsequently appeared on the same GPU. Efficient-attention
and padding-run timings are therefore excluded from speed conclusions. The
FP32 efficient run repeated exactly; compared with reference attention, its
maximum coordinate-component difference was 0.004456 A and expected-PAE
maximum difference was 0.000705 A. These results do not justify a default
backend change from that experiment alone: its timing was contaminated, and
one input's numerical agreement is not a general accuracy qualification.

### BF16 padding and reproducibility controls

The missing default-precision control now uses complete 7r6r with ten recycles,
200 steps and one sample. Padding only the residue trunk from **245 to 320
tokens**, with unchanged diffusion shape and seed, changed coordinates by
**7.565 A aligned all-atom RMSD** (maximum unaligned coordinate-component
difference **21.916 A**, component RMS **4.712 A**). The maximum expected-PAE
difference was **6.006 A**. Model/runner source hashes matched the unpadded
reference, and the later clean-GPU unpadded run reproduced that reference
bitwise.

Replacing the masked padding values with independent random values produced
**bitwise-identical predictions to zero padding across all compared fields**.
Both padded configurations also repeated exactly, with finite floating output
tensors. This supports shape-dependent numerical sensitivity rather than
leakage of padding contents for this control; it does not prove correctness
for every input or identify a biological cause for the sensitivity. These
instrumented runs are excluded from speed claims.

Use `--dtype bf16 --pad-trunk-to 320` and then add `--random-padding`, comparing
the second run against the first run's `last.pt`. Fix grouping, precision and
kernels when comparing outputs; FP32's small padding difference below does
not describe the default BF16 behavior.

### FP32 padding and reproducibility controls

Forcing only 7r6r's residue trunk from 245 to 320 tokens, with unchanged
precision, chunk setting, diffusion shape and seed, changed final coordinates
by at most 0.003166 A per component (RMS 0.0000947 A). Replacing every masked
padding value with random values left **all compared predictions bitwise
identical** to zero padding. Both padded configurations repeated exactly.
These controls support shape-dependent numerics for this case; they do not
establish Mol-1's larger reported divergence in OpenDDE or prove masking for
all inputs. Keep actual grouping, shapes, precision and kernels fixed when
comparing changes.

### Validation for this pass

Targeted CPU checks: **412 passed, 34 skipped**. Focused CUDA aggregation,
attention, cache and multi-job tests: **140 passed**. The released-checkpoint
mixed-length test separately passed with three inputs and three samples each,
checking coordinates, confidence splits, rankings and atom/residue mappings.
The initial combined CUDA run reached that smoke test after other tests had
initialized cuBLAS without its deterministic workspace setting; rerunning the
smoke test in a fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8` passed.
These suites overlap and their counts should not be summed as unique tests.
Ruff and `git diff --check` passed.

Those runs used the **OpenDDE Conda interpreter and its Torch 2.7.1+cu126**,
including the CUDA checks. Because that environment lacked pytest and Gemmi,
the temporary pytest launcher appended a second environment's (`bg`)
site-packages after OpenDDE's own packages. This supplied test dependencies;
it did not select the Mol-1 interpreter or its Torch. The original results
therefore were not from a fully self-contained OpenDDE test environment.

A subsequent rerun used the OpenDDE interpreter directly with `-m pytest`,
with pytest 9.1.1, Gemmi 0.7.5 and their test dependencies installed in that
environment, without changing Torch 2.7.1+cu126. It passed **242 tests**, with
**one opt-in ipSAE CUDA check skipped**. This covered atom aggregation, ipSAE,
interface inference, input batching, scheduling, attention, projection caches,
inference reuse and the released-checkpoint smoke test. CUDA kernel tests were
enabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set before process startup.
The separately gated ipSAE test then passed with `TEST_IPSAE_CUDA=1`, covering
all 243 collected tests across the two invocations.
The GPU was shared, so this rerun validates behavior, not performance.

To reproduce that coverage directly in an activated OpenDDE environment:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENDDE_RUN_BATCH_KERNEL_TESTS=1 \
  OPENDDE_RUN_BATCH_SMOKE=1 TEST_IPSAE_CUDA=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  LAYERNORM_TYPE=torch python -m pytest tests/test_atom_aggregation.py \
  tests/test_ipsae.py tests/test_ipsae_interface_inference.py \
  tests/test_input_batching.py tests/test_batched_prediction.py \
  tests/test_diffusion_attention_execution.py tests/test_diffusion_projection_cache.py \
  tests/test_inference_reuse.py tests/smoke/test_input_batching_runtime.py -ra
```

```bash
LAYERNORM_TYPE=torch python -m pytest tests/test_atom_aggregation.py \
  tests/test_ipsae.py tests/test_ipsae_interface_inference.py \
  tests/test_input_batching.py tests/test_batched_prediction.py \
  tests/test_diffusion_attention_execution.py tests/test_diffusion_projection_cache.py -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENDDE_RUN_BATCH_SMOKE=1 \
  LAYERNORM_TYPE=torch python -m pytest tests/smoke/test_input_batching_runtime.py -q
```

### Reproduce

The benchmark saves source hashes, settings, per-pass process snapshots, full
CPU predictions, repeat comparisons, and final CIF/confidence files. Use the
project's interpreter and a free GPU, with cached checkpoint/CCD assets:

```bash
LAYERNORM_TYPE=torch CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --input examples/cofold/example.json --names 7r6r --dtype fp32 \
  --output /tmp/opendde_fp32_reference
```

The benchmark defaults to `--dtype bf16`, matching production. Specify FP32
explicitly to reproduce the measurements above. Use
`--repo /path/to/baseline-checkout` for the original implementation,
`--backend efficient` for the optional attention layout, and
`--compare /path/to/reference/last.pt` to retain full output comparisons.
`--pad-trunk-to 320` changes only the trunk's padded shape;
`--random-padding` fills masked padding with independent random values. These
instrumented runs are correctness controls, not throughput measurements.
`--execution singletons` compares a selected multi-input queue with separate
native forwards. Outputs and large benchmark artifacts should remain outside
the source tree.

Further performance measurements should cover equal-length pairs and more
representative queues on a free GPU. Hold seeds, precision, recycles, steps
and samples fixed, and report total queue time per input rather than time per
forward. The unequal-length comparison above and Mol-1's pairing result do
not establish the best BF16 grouping for every OpenDDE workload.
With one sample, confidence chunking cannot reduce sample-loop work, but the
200 diffusion steps still contain attention and projection costs. Remaining
optimization opportunities are not limited to the trunk. TMK3 compatibility,
numerical validation and the matched whole-model comparison are reported above.

The following sections retain earlier implementation details and measurements;
their five-sample synthetic timings are separate from this one-sample real-input
comparison.

For five structures per complex and ten recycling cycles:

```bash
opendde pred -i inputs.json -o output \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --trimul_kernel auto --triatt_kernel auto \
  --confidence_sample_chunk_size 5 --need_atom_confidence false
```

`batch_size` counts independent complexes; `sample` counts structures for each
complex. More than two inputs are supported, subject to memory. Group similar
lengths to reduce padding. The existing scheduler splits batches after an OOM.

The mixed-length path reuses padded trunk buffers and masks across recycles,
caches noise-independent single/atom conditioning, and reuses projected diffusion
pair biases and atom attention/transition projections. The resident projection
cache budget defaults to 2,048 MiB per prediction. CUDA further caps it at one
eighth of currently available memory (including reusable allocator blocks);
CPU/MPS cap it at 256 MiB. This bounds retained projections, not total model
memory or temporary cache-construction workspace. Projections that do not fit
are recomputed as before. All caches belong to one prediction and are released
before confidence evaluation.

Each denoising step also shares identical noise conditioning among a job's
samples. Coordinates and random streams remain independent, so five requested
samples still produce five structures. The default path preserves diffusion steps, recycles, precision
settings and attention kernels. These optimizations apply to native batches, including singleton groups.
The legacy CLI `--batch_size 1` and distributed Fold-CP paths retain their
existing execution; `predict_batch([job])` now uses native batching.

The CLI and Python runner expose `diffusion_cache_max_mb=2048` and
`share_diffusion_sample_conditioning=True`. Their configuration keys are
`infer_setting.diffusion_cache_max_mb` and
`infer_setting.share_diffusion_sample_conditioning`. To disable the added
projection caches and shared conditioning for comparison:

```bash
opendde pred -i inputs.json -o output --use_msa false --batch_size 2 \
  --sample 5 --cycle 10 --step 200 \
  --diffusion_cache_max_mb 0 --share_diffusion_sample_conditioning false
```

`--enable_cache false` continues to disable all diffusion shared-variable
caches; the sample-conditioning switch is independent.
Successful batches retain CUDA allocator blocks; OOM retry still releases
unused memory.

cuEquivariance preserves independent job axes and padding masks. `auto` selects
it when compatible optional packages are available, otherwise Torch. Explicit
`--trimul_kernel torch --triatt_kernel torch` retains Torch arithmetic.
Distributed Fold-CP retains its existing Torch path.

Confidence chunking defaults to `1`. Setting `5` processes five samples together,
using more memory for the current chunk while still offloading completed pair
logits to CPU. Training, gradient-enabled calls, Fold-CP, and inputs above 2,000
confidence tokens keep serial confidence execution. The Python runner accepts
`confidence_sample_chunk_size=5`; the configuration key is
`infer_setting.confidence_sample_chunk_size`.

## Diffusion attention options

Native batches also reuse local atom-attention padding masks by default. Masks
are built directly at window size, with at most 16 MiB retained per sampling
request, and released before confidence evaluation. This is separate from the
projection-cache budget. Use `--reuse_diffusion_window_masks false` to compare
with the original mask construction. `--enable_cache false` disables this
reuse too. Explicit, data-dependent attention biases keep their original path.

An optional CUDA layout lets PyTorch select its existing memory-efficient FP32
attention kernels for global diffusion attention and chunked local atom
attention:

```bash
opendde pred -i inputs.json -o output \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --confidence_sample_chunk_size 5 --diffusion_attention_backend efficient
```

`diffusion_attention_backend` defaults to `reference`. The Python runner takes
the same argument; configuration uses
`infer_setting.diffusion_attention_backend`. `reuse_diffusion_window_masks`
likewise exists in the Python API and under `infer_setting`.

The optional backend changes the reduction order. It preserves the configured
precision, attention scale, masks, random streams, sample count and cycles, but
coordinates, confidence and native ipSAE can change numerically. Use
`reference` when comparing exact results with previous runs. No external
FlashAttention package is needed. CPU/MPS retain their existing path, and
unsupported CUDA shapes use PyTorch's normal fallback. Both options apply to native batches, including singleton groups; they do
not modify trunk/confidence, legacy serial inference or Fold-CP execution.

### Interpreting GPU contention

CPU overlap does not explain lower sensitivity to GPU contention. If CPU time
`C` stays fixed and GPU time `G` increases by a factor `s`, the serial wall-time
ratio is `(C + s*G) / (C + G)`. CPU work dilutes that ratio; hiding it through
overlap can make the ratio larger. Model-only timings here exclude preparation
and output writing. Mol-1's optional input prefetch and background writer are
off by default, and its published overlap comparison was itself contaminated
by another GPU process, so it establishes neither a speedup nor a universal
slowdown.

The earlier two-input, five-sample profiles show different GPU work:

| Stage | Mol-1, 512/640 tokens | OpenDDE, 256/320 residues, 498/624 structural tokens |
| --- | ---: | ---: |
| Trunk | 27.331 s | 10.767 s |
| Diffusion sampling | 28.446 s | 28.995 s |
| Confidence/heads | 6.249 s | 1.217 s |

Diffusion accounts for approximately 46% and 70% of these respective profiles.
OpenDDE's structural expansion explains why its diffusion token counts can be
nearly twice its residue counts. These historical profiles use different input
lengths, kernels and confidence precision policies; they are not a controlled
cross-model throughput comparison or proof of a particular contention ratio.
They also predate subsequent optimizations.

The saved Torch 2.7.1 attention probe confirms that 5D global attention used
math SDPA, while the supported 4D layout selected memory-efficient SDPA. The
earlier layout prototype reduced warm model time from 37.826 s to 34.843 s
(7.89%, one measured pass per variant). This supports optimizing diffusion,
but does not measure its behavior under contention. Scheduling overhead and
cache interference remain possible mechanisms, not conclusions from these
stage totals; kernel counts alone do not establish context-switch counts or
L2 eviction.

The `empty_cache()` in structural-feature preparation belongs to
`_maybe_expand_to_structural_tokens_foldcp_local`, which returns before that
call when there is no distributed Fold-CP mesh. It is not a per-forward flush
in ordinary single-GPU inference. Current OpenDDE and Mol-1 native batch
runners both retain allocator blocks after successful batches; OOM recovery
and legacy serial cleanup have separate behavior. Removing the Fold-CP flush
would not optimize the native path measured here.

On November 5, a controlled contention comparison was attempted on H200 GPU 7
using complete 7r6r, a native singleton, BF16 trunk with FP32 diffusion, ten
recycles, 200 steps and one sample. Both clean reference processes completed
one warmup and three measured forwards. Instrumentation counted **zero
`empty_cache()` calls in all eight forwards**. Diffusion represented roughly
59% of model time for this one-sample workload, rather than the historical
five-sample profile's 70%.

In each attempt, an unrelated CUDA process appeared during the subsequent
efficient-attention case. Monitoring stopped the benchmark's own process and
marked the case contaminated; neither attempt reached the planned synthetic
contender cases. No shared/idle ratio or new backend speed comparison is
reported. The reference default is unchanged. The missing measurement is a
matched, repeated reference/efficient comparison both alone and alongside the
same contender, with stage timings, output comparisons and process monitoring.
Use `predict_batch([job])` for the singleton: legacy CLI `--batch_size 1` does
not exercise this attention option. Even a completed synthetic-contender test
would characterize that workload, not establish a universal scheduler or L2
cache explanation or a matched Mol-1 comparison.

The production Torch 2.7.1 environment passed all **99** focused attention
tests with CUDA enabled (`OPENDDE_RUN_BATCH_KERNEL_TESTS=1 python -m pytest
tests/test_diffusion_attention_execution.py -q`). These cover layout/mask
parity, precision, request-local state and multi-job sample ownership; they
are correctness checks, not contention measurements.

### Full-checkpoint attention validation on repository examples

A subsequent production-environment check used `examples/cofold/example.json`
with external MSA/templates disabled, ten recycles, 200 steps, one sample and
seed 101. The trunk used BF16; diffusion and confidence used their normal FP32
policy for these lengths. Both triangle kernels used cuEquivariance, TF32 was
off, and the projection-cache budget was 2,048 MiB. The inputs had these public
PAE-axis / structural-token counts: **7r6r 245/475**, **7pzb 600/1122**, and
**7wux 1218/2300**. Counts include the examples' nucleic acids and ligands.

7pzb and 7wux were each evaluated as native singletons, and 7r6r/7pzb as an
unequal-length native batch. Three forwards per backend and grouping completed
**18 forwards**, with all **12 within-setting repeat comparisons bitwise
identical**. Checks covered finite outputs, per-job atom and PAE axes, exact
atom-to-token and chain mappings, and eight written prediction/summary sets.
Every written ipSAE result matched recomputation from that job's PAE. Timings
from this correctness session are excluded from backend performance claims.

The selected interfaces were `[["A"], ["B"]]` for 7pzb and
`[["A", "B"], ["C", "D"]]` for the four-protein-chain 7wux example. The
7pzb scalar was **0.2704062002441992** for both backends, both alone and in the
batch. The 7wux pooled scalar was **0.0**, rather than null, with correct
selected labels and all six individual chain-pair records retained. That zero
verifies output plumbing, not a confident interaction. Five focused pooled
score tests also passed, including a nonzero pooled oracle, exclusion of
within-side/unlisted-chain contributions, and explicit two-chain compatibility.
The targeted command was `python -m pytest tests/test_ipsae.py -q -k
'grouped_interface or internal_interface or explicit_two_chain'`.

Switching attention backends within a fixed grouping changed predictions:

| Input and grouping | Maximum coordinate-component difference | Maximum expected-PAE difference |
| --- | ---: | ---: |
| 7r6r singleton, separate initial check | 0.001413 A | 0.000246 A |
| 7pzb singleton | 0.000207 A | 0.000356 A |
| 7wux singleton | 0.038095 A | 0.004174 A |
| 7r6r in the two-input batch | 0.000367 A | 0.000206 A |
| 7pzb in the two-input batch | 0.000153 A | 0.001802 A |

The largest coordinate-component RMS difference was **0.001272 A** on 7wux.
These are numerical comparisons, not experimental accuracy measurements.
Separately, changing 7r6r from a native singleton to the padded 7r6r/7pzb batch
with reference attention reproduced **7.544334 A aligned all-atom RMSD**.
Keep grouping fixed when comparing backends or reproducing BF16 predictions;
the padding effect must not be attributed to efficient attention. No allocator
flush via `torch.cuda.empty_cache()` occurred inside any of these 18 forwards.

### Attention timing and synthetic contention follow-up

An initial clean 7r6r screen used the same settings and instrumented
`predict_batch` stages. Three measured forwards followed a warmup for each
backend, in reference-then-efficient order. Median model time fell from
**7.871 s to 7.460 s (5.22%)**, and diffusion time from **4.630 s to 4.220 s**.
No unrelated GPU process was detected during those accepted passes. Loading,
featurization, validation and writing were excluded. This short, single-order
screen supports a speed benefit on this input, not a general queue throughput
estimate.

The later contention experiment completed all four conditions in forward and
reverse order: reference/efficient, each with and without an additional
synthetic contender. Each block had one warmup and three measured forwards,
giving **six measurements per condition and 32 forwards in total**. The
contender repeatedly multiplied 8192-by-8192 BF16 matrices in a separate
process, with at most eight GEMMs queued before synchronization and 384 MiB
of tensor storage. Model code hashes matched the correctness session.

**An unrelated `esmfold2` process was present throughout all 32 forwards.**
Shared execution was explicitly allowed, so the experiment continued and
recorded that process. The following are observations under shared GPU load;
the first column is not an idle-GPU baseline, and the ratios do not isolate
the synthetic contender's causal effect:

| Attention | Without synthetic contender, median (range) | With synthetic contender, median (range) | Observed ratio |
| --- | ---: | ---: | ---: |
| Reference | 7.924 s (7.919–10.553) | 16.802 s (16.774–16.903) | 2.12x |
| Efficient | 7.513 s (7.495–7.537) | 16.704 s (16.665–16.790) | 2.22x |

The corresponding diffusion-stage ratios were **1.82x** and **1.97x**. This
screen did not demonstrate reduced contention sensitivity from efficient
attention. It also does not identify context-switch costs or L2 behavior.
All eight retained end-of-block predictions matched their respective
backend's earlier clean prediction bitwise, including confidence outputs.

**Decision:** retain the `reference` default. `--diffusion_attention_backend
efficient` is a validated opt-in for the tested native FP32-diffusion workloads
when its numerical changes are acceptable. The initial clean screen supports
using it for speed; the proposed contention argument does not support a default
change. CPU/MPS, training, legacy serial inference and Fold-CP behavior remain
as described above. Full-model AMP diffusion at larger lengths was not newly
validated by this experiment.

### Eight-GPU independent-worker smoke check

Eight concurrent processes each ran the repository's 7r6r example on one H200,
using `CUDA_VISIBLE_DEVICES`, native singleton inference and efficient
attention. Each process used one warmup and two measured forwards at the same
ten-cycle, 200-step, one-sample BF16/FP32 settings above. All **eight workers
completed successfully**. Each repeated exactly, and every worker's retained
prediction matched the earlier single-GPU efficient prediction bitwise across
all outputs. Model/runner source hashes matched across the eight processes.
The three written files per worker (CIF and confidence outputs) also matched
byte for byte across all eight output directories.

This validates the independent-process execution route; it does not exercise
`torchrun`, NCCL or distributed Fold-CP. Other jobs remained on the devices.
Observed per-worker median model times ranged from **7.559 to 19.364 s**, with
**5.158 GiB peak allocated memory per worker**. The complete smoke run took
**96.881 s including startup, preparation, validation and writing**. These
shared-load observations are not an isolated eight-GPU scaling result or a
throughput estimate for a heterogeneous production queue.

## Choosing a batch size

An H200 sweep used the same eight synthetic protein complexes for each batch
size, five samples per complex, ten cycles, 200 diffusion steps, cuEquivariance,
and a confidence chunk of five. Inputs were sorted by total protein length.
Prediction times exclude model loading, featurization and file writing. Each
reported time followed one complete warmup pass. The projection-cache budget
was **8,192 MiB**, above the default 2,048 MiB.

| Total residues per complex | Batch size | Time for eight complexes | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| 224–336 | 2 | 127.4 s | 12.8 GiB |
| 224–336 | 4 | 132.1 s | 18.6 GiB |
| 224–336 | 8 | 163.6 s | 33.5 GiB |
| 384–640 | 2 | 317.6 s | 38.3 GiB |
| 384–640 | 4 | 392.1 s | 59.3 GiB |

Batch two was fastest among these settings. Larger batches increase padding
and can exceed the projected-pair cache budget. Raising the budget to 16,384 MiB
reduced the long-input batch-four run to 380.6 s, still slower than batch two.
The device-availability cap remains active. These are single measured passes,
so small timing differences should not be treated as precise speedups.

The long-input batch-eight sweep also exposed corrupted rows in very large
Torch 2.7 CUDA layer-normalization calls. The Torch layer-norm path now splits
oversized calls along independent leading axes, before any contiguous copy.
Small calls retain the original path. A `[8,1245,1245,384]` BF16 reproducer,
including transposed input, then matched independent per-job normalization
exactly. The normalization arithmetic, affine parameters and gradients are
preserved. The formerly failing eight-job workload then completed all 40
structures in 560.6 s, with 112.9 GiB peak allocation. That was a single full
validation pass without a preceding warmup, so it is not a steady-state timing.

Full-checkpoint attention comparisons covered mixed-length synthetic dimers
and benign ubiquitin, calmodulin, myoglobin/heme and hemoglobin/heme controls,
with native batches of two, three and four jobs and sample chunks of two and
five. Mask reuse preserved all 1,242 compared tensors and all summary values.
Its measured time reduction ranged from negligible to 3.3%.

Mask reuse plus optional efficient attention reduced prediction time by
4.7–10.0% against the pre-change reference, or about 4–10% against the new
mask-reuse default. Across those
comparisons, the largest coordinate-component difference was 0.4293 Å (largest
per-job coordinate-component RMS difference 0.0081 Å), and the largest expected
PAE difference was 0.0792 Å. Scalar `ipsae_min_10` matched in these controls;
one directional chain-pair score changed by about 0.0000021. These observations
do not guarantee identical future scores or experimental structure accuracy.

## Native scores

Summary JSON includes `ipsae_min_10`, `ipsae_min_10_d0ref`, `ipsae_chain_pairs`,
and `ipsae_parameters`. Scores use the existing in-memory PAE, including when
full confidence JSON is disabled. No extra prediction or PAE-file round trip is
needed. CIF chain labels and sample/ranking ownership are preserved. Raw model
summaries use integer asym IDs and record that in the parameters; the dumper
converts them to the structure's chain labels.

The convention matches the Mol-1 benchmark: round PAE to two decimal places,
select values strictly below 10 Å, use each residue's selected partner count
with a minimum count of 27 for d0, maximize over residues in each direction,
then take the smaller directional score. The `_d0ref` variant clips at 26.
Only protein CA tokens count, including the CA token of an atom-tokenized
modified residue. Ligands and nucleic acids are excluded. With exactly two
protein chains the scalar is defined; otherwise it is null and individually
labelled pair records are available.

Scoring matches postprocessing of the same rounded PAE. Changing kernels or
confidence chunk sizes can change floating-point predictions; identical scores
across different prediction settings are not guaranteed.

## Validation

CPU regressions cover cache budgets and training fallbacks, partial atom caches,
multiple jobs and sample chunks, disabled heads, cache immutability,
residue/structural token projections, and scoring with full-data output both
enabled and disabled. Opt-in GPU tests cover singleton and ragged cuEquivariance
calls, padding, tiny-shape fallback, and Torch comparisons:

```bash
OPENDDE_RUN_BATCH_KERNEL_TESTS=1 LAYERNORM_TYPE=torch \
  python -m pytest tests/test_diffusion_attention_execution.py tests/test_diffusion_projection_cache.py tests/test_inference_reuse.py tests/test_ipsae.py -q
```

Use a free GPU and the project's environment. Synthetic-control timings measure
execution, not structure quality.


## Padding-aware scheduling

`--batch_max_padding_ratio` bounds padded cubic token work relative to the
sum of unpadded work. Zero (the default) disables the cap; values must otherwise
be at least one. `--batch_max_pair_tokens` bounds `B * max_tokens**2`; zero
also disables this limit. These are work/element estimates, not GPU-memory
limits in bytes. Both bounds are checked separately on residue and expanded
structural-token counts after featurization. A singleton is always allowed,
and forward OOM retry remains available.

For a heterogeneous queue with minimal padding, keep eight as the maximum:

```bash
opendde pred -i inputs.json -o output \
  --use_msa false --use_template false --use_rna_msa false \
  --dtype bf16 --batch_size 8 --batch_max_padding_ratio 1.0 \
  --cycle 10 --step 200 --sample 1
```

Jobs equal-sized in **both token spaces** can still share a forward. Equal
protein sequence lengths alone do not guarantee equal structural counts:
glycine contributes one structural token, while most other complete standard
protein residues contribute separate backbone and sidechain tokens. Different
glycine counts can therefore split equal-residue-length designs at a cap of
`1.0`. This distinction is absent from Mol-1's single-token-space limiter.

Jobs exceeding the cap flush the
current group; the scheduler does not search the full queue for an optimal
packing. Job ownership, all requested seeds and sample counts are preserved.
The Python runner accepts the same argument names. Dotted configuration keys
are `infer_setting.batch_max_padding_ratio` and
`infer_setting.batch_max_pair_tokens`.

### Equal-length batch fragmentation audit (2026-11-07)

An external benchmark reported 16 equal-length designs producing eight
singleton forwards, one pair and two triples, with identical grouping at
`batch_size=4` and `8`. Its launch configuration and exact prepared token
counts were not available for this audit, so the cause of that particular run
is not established. Treat this scheduling question separately from GPU
contention and model-kernel throughput.

A CPU regression through the real scheduler reproduces the **exact reported
group-size pattern** with 16 inputs of 320 residue tokens and structural counts
`[620, 619, 618, 617, 616, 615, 614, 613, 612, 612, 611, 611, 611, 610, 610, 610]`.
These counts are consistent with standard protein sequences having different
glycine counts; this is a constructed reproducer, not recovered benchmark data.

| Padding cap | Pair-token cap | `batch_size=4` groups | `batch_size=8` groups |
| --- | --- | --- | --- |
| 1.0 | Disabled | Eight singles, one pair, two triples | Same |
| 1.05 | Disabled | Four groups of four | Two groups of eight |
| Disabled (0) | Disabled | Four groups of four | Two groups of eight |

Sixteen inputs equal-sized in both token spaces also fill two eight-job batches
at a cap of `1.0`. Thus this pattern can result from a configured padding cap;
it does not establish an intrinsic maximum batch size of three. The current
greedy scheduler only combines adjacent prepared jobs and sorts beforehand by
estimated sequence length, so it can also miss nonadjacent structural-size
matches. Changing this packing policy is a separate optimization question.

The scheduler now logs effective limits at startup. Every early flush due to a
limit or precision change logs its reason, pending/max job counts, and the
candidate residue and structural counts. These diagnostics do not change the
grouping policy. OOM retries already emit a distinct memory warning. Use these
logs, the number of input JSON lists and the per-rank job assignment to diagnose
a run before attributing small groups to the model.

For the constructed example, `1.05` permits the small structural-size variation;
it is not a universally optimal cap. Disabling the cap permits more padding.
Neither choice guarantees a speedup or eight-job memory capacity, and changing
grouping can change BF16 predictions. Preserve the strict cap when zero padding
is required; do not silently ignore structural-token sizes to fill a batch.

Validation under the production interpreter: `python -m pytest
tests/test_batched_prediction.py tests/test_structural_tokenizer.py
tests/test_inference_config.py tests/test_diffusion_projection_cache.py -q`
passed **124 tests**, with **two opt-in CUDA tests skipped**. Ruff checks and
the CLI help smoke check passed. No checkpoint forward or external benchmark
rerun was required for the scheduling/diagnostic change; the original workload's
GPU capacity, throughput and precise fragmentation cause remain unverified.

Native singleton groups use the batched sampler and projection caches. They
can differ numerically from the former legacy fallback. Hold grouping, precision,
kernels and sampling settings fixed for comparisons; fixed seeds do not imply
identical predictions across tensor shapes. The ordinary CUDA atom-to-token
reduction now uses a stable atom order for sum/mean inference with a shared
one-dimensional mapping. CPU, gradients and explicitly deterministic execution
retain their previous reduction paths. This reproducibility fix can change
historical predictions.

**Python API compatibility:** `runner.predict_batch([job])` and
`model.forward_batch([features])` now enforce the native batching restrictions
even for one job. In particular, MSA, templates and RNA MSA must be disabled;
a singleton with any of these enabled now raises instead of falling back to
legacy inference. Use `runner.predict(job)` for the legacy single-job API when
those features are needed. The serial CLI `--batch_size 1` retains its existing
behavior.

Selected multi-chain ipSAE interfaces are described in
[the input format](infer_json_format.md#ipsae-interface-selection).
