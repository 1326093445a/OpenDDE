# Mixed-length inference and native ipSAE

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
settings and attention kernels. These optimizations apply to native
mixed-length batches of two or more jobs; single-job and distributed Fold-CP
paths retain their existing execution.

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
unsupported CUDA shapes use PyTorch's normal fallback. Both options apply only
to native batches of two or more jobs; they do not modify trunk/confidence,
serial inference or Fold-CP execution.

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
