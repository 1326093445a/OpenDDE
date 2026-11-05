# Mixed-length inference and native ipSAE

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

Equal-sized jobs can still share a forward. Jobs exceeding the cap flush the
current group; the scheduler does not search the full queue for an optimal
packing. Job ownership, all requested seeds and sample counts are preserved.
The Python runner accepts the same argument names. Dotted configuration keys
are `infer_setting.batch_max_padding_ratio` and
`infer_setting.batch_max_pair_tokens`.

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
