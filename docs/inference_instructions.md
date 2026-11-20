# Inference Instructions

Concise reference for installing OpenDDE, preparing runtime data, and running
`opendde` commands.

## Install

OpenDDE supports CPython `3.11`, `3.12`, and `3.13`. We recommend
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python
installations. Choose one of the following methods.

### Install from PyPI

```bash
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu opendde
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 "opendde[gpu]"
```

### Install from source

```bash
git clone https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu -e .
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

After a PyPI or source installation, verify the environment with:

```bash
uv run --no-project --python .venv opendde doctor
```

### Use Docker

The prebuilt image targets NVIDIA GPU inference:

```bash
docker pull aurekaresearch/opendde:v1
```

See the [Docker guide](./docker_installation.md) for GPU setup, runtime-data
mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU or on the Metal (MPS) backend with `--device mps`; Intel
> macOS is unsupported, and Windows has not been validated. At runtime,
> `--device auto` uses CUDA when available, then MPS, and otherwise falls back
> to CPU.

## Runtime data

Set `OPENDDE_ROOT_DIR` to the directory that stores checkpoints and runtime data:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA search
```

The default checkpoint and managed common files come from a release-pinned
asset revision and are verified against their published size and SHA-256.
Checkpoints passed explicitly with `--load_checkpoint_path` are left untouched.

Prepare data from a source checkout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

For a protein-only prediction that disables MSA, template, and RNA-MSA features,
search databases are not needed:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

If you already have a custom checkpoint, keep a descriptive filename and pass
it directly. Use `--skip-model` when preparing only the remaining runtime data:

```bash
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /path/to/my_checkpoint.pt \
  "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt"
bash scripts/download_opendde_data.sh --skip-model
opendde pred \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt" \
  -i examples/cofold/input.json \
  -o ./output
```

The names `opendde.pt` and `opendde_abag.pt` are reserved for released assets.
Their authoritative links, sizes, and digests are in
[supported_models.md](./supported_models.md).

Use `opendde.pt` with `-n opendde_v1` as the default general-purpose
checkpoint. To use the ABAG-optimized checkpoint, keep it as
`opendde_abag.pt` and pass it with `--load_checkpoint_path`, for example
`opendde pred --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"`.

Install and verify the ABAG checkpoint from the same manifest-backed helper:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh \
  --checkpoint opendde_abag.pt \
  --skip-common \
  --skip-search-database
```

Then run general-purpose inference without an explicit checkpoint path. For ABAG
inference, add:

```bash
--load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENDDE_ROOT_DIR` | Checkpoints, common files, search databases. Defaults to `~/.cache/opendde`. |
| `OPENDDE_DEPENDENCY_URL` | Override checkpoint download root. |
| `OPENDDE_COMMON_URL` | Override common runtime file download root. Falls back to `OPENDDE_DEPENDENCY_URL` when set. |
| `OPENDDE_SEARCH_DATABASE_URL` | Override template/RNA-MSA database download root. |
| `LAYERNORM_TYPE` | LayerNorm backend; defaults to `torch`. Set to `fast_layernorm` to opt into the fused kernel. |

Template/RNA-MSA preprocessing also needs HMMER. Template inference may need
`kalign`:

```bash
apt-get update && apt-get install -y hmmer kalign
```

## Input JSON

OpenDDE input is a top-level list of jobs:

```json
[
  {
    "name": "tiny",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and may be omitted from a job; include it only to
declare explicit covalent links between entities.

Full schema: [infer_json_format.md](./infer_json_format.md).

Convert a structure file to JSON:

```bash
opendde json -i examples/cofold/structures/7pzb.pdb -o ./output --altloc first
opendde json -i examples/cofold/structures/2lwu.cif -o ./output --altloc first --assembly_id 1
```

## Preprocess optional features

```bash
# Protein MSA
opendde msa -i examples/cofold/input.json -o ./output

# Protein MSA + template search
opendde mt -i examples/cofold/input.json -o ./output

# Protein MSA + template search + RNA MSA when RNA is present
opendde prep -i examples/cofold/input.json -o ./output
```

Notes:

- Protein MSA uses the public ColabFold MMseqs2 API unless A3M paths are already
  present in the JSON.
- Template and RNA-MSA search use local databases under
  `$OPENDDE_ROOT_DIR/search_database/`.
- Generated JSON files are written under
  `<out_dir>/.opendde_preprocessed/<input-hash>/` rather than next to the
  input JSON, so read-only input directories work.

Details: [msa_template_pipeline.md](./msa_template_pipeline.md).

## Run prediction

Standard run:

```bash
opendde pred -i examples/cofold/input.json -o ./output -n opendde_v1
```

Compatibility run with the standard step/cycle counts:

```bash
opendde pred \
  -i examples/cofold/input.json \
  -o ./output \
  -n opendde_v1 \
  --trimul_kernel torch --triatt_kernel torch \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Inference defaults to `--device auto`, `bf16` with TF32 off (`--dtype fp32` and
`--enable_tf32 true` opt out), and TMK3 multiplication plus cuEquivariance triangle attention.
Device auto-selection uses NVIDIA CUDA when available, then the Apple Metal
(MPS) backend on Apple Silicon, and otherwise CPU.
The default kernel selections require compatible CUDA dependencies. Select
`--trimul_kernel torch --triatt_kernel torch` for CPU/MPS or
`--trimul_kernel auto --triatt_kernel auto` for portable runtime dispatch.
MPS follows the same BF16 default and dynamic policy as CUDA:
the trunk uses BF16; by default, diffusion and the confidence head stay FP32
through 2560 tokens, confidence uses BF16 above 2560, and diffusion uses BF16
above 3840 to reduce memory.
BF16 performance varies by Apple GPU and workload, so compare it with FP32 for
your inputs. BF16 autocast needs macOS 14 or newer and is downgraded to FP32
below that.

## Default inference baseline

The default generates all five requested samples with Torch LayerNorm, TMK3
multiplication, cuEquivariance attention and diffusion/confidence chunks of five.
Early stopping, refill and CUDA replay remain disabled. Existing correctness
fixes, eager caches and precision policy remain active. An explicitly exported
`LAYERNORM_TYPE` still overrides the Torch LayerNorm default.

| Setting | Default | Optional override |
| --- | --- | --- |
| Precision | `--dtype bf16`, with the existing length-dependent mixed-precision policy | `--dtype fp32` |
| Triangle multiplication / attention | `--trimul_kernel flashpairformer --triatt_kernel cuequivariance` | `auto` or `torch`; multiplication also supports `cuequivariance` |
| LayerNorm | `LAYERNORM_TYPE=torch` | Fused LayerNorm is optional and changes predictions |
| Diffusion / confidence sample chunks | `5` / `5` | Smaller chunks reduce memory but can change numerics and throughput |
| Early stopping / batch refill | Disabled (`None` / `false`) | Explicit screening policy below |
| Diffusion attention | `--diffusion_attention_backend reference` | `efficient`; floating-point predictions can change |
| Diffusion caches and existing eager fusions | `--enable_cache true --enable_fusion true` | Explicitly disable for controls; these switches do not enable compilation or replay |
| Native batch projection cache / shared conditioning / window masks | 2,048 MiB budget / on / on | Existing cache-budget and sharing flags |
| CUDA denoiser replay | `--diffusion_cuda_graph false` | `true`, subject to the restrictions below |
| Compilation and independent aggregation-plan-cache experiments | Off; no prediction CLI flag | Benchmark scripts only |
| TF32 | `--enable_tf32 false` | Explicit override |
| Batch limit / samples | `1` / `5` | Select for the workload; one-sample examples explicitly use `--sample 1` |

The default CUDA path requires cuEquivariance and opt_core; see the
[kernel installation guide](kernels.md#optional-tmk3-triangle-multiplication).
For CPU, MPS or Fold-CP use `--trimul_kernel torch --triatt_kernel torch`.
Explicit `auto` still selects available cuEquivariance or Torch, not TMK3. Ordered
CUDA atom aggregation, native singleton/batch caches, confidence splitting and
the optional per-job `ipsae_interface` selection remain supported. Grouped
ipSAE requires an explicit interface because the intended chain grouping cannot
be inferred generally. Existing jobs retain their scoring behavior.

The optional dependencies must be installed separately; the default may JIT
compile eligible TMK3 kernels, but does not compile the model or capture graphs. Keep batch grouping fixed for reproducible comparisons. The
ordinary `--batch_size 1` CLI route remains legacy serial inference; selecting
`--batch_size 2` permits native eager batching even when only one job is present.
Enabling replay also selects native batching, so comparing it with legacy
serial inference changes more than graph execution.

The [README eager example](../README.md#default-eager-batching-example) uses ten
recycles, 200 steps and five samples per complex. Confidence chunks of five can
raise peak memory; lower the chunk explicitly if necessary. Later experiments remain
available for explicit evaluation; they are not promoted to defaults by an
installation, checkpoint or Torch upgrade. In particular, whole-Pairformer
BF16 compilation changes coordinates and confidence substantially; see the
[measurement report](batched_inference_optimizations.md).

## Multiple complexes per GPU

For independent complexes without MSA, set `--batch_size` to the maximum number
of jobs per GPU forward. Jobs may have different chain lengths and atom counts.
Keep them in one JSON job list so the scheduler can collect them into batches:

```bash
opendde pred -i examples/cofold/no_msa_batch/inputs.json -o ./output/batched \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 3
```

Here a full batch contains two complexes with three diffusion samples each.
`--sample` and `--batch_size` are separate axes. The default `--batch_size 1`
retains the existing single-job path unless CUDA replay is explicitly enabled.
Each job/seed gets its
own structures, confidence summaries, PAE/PDE maps, atom pLDDT and rankings.
The final batch may be smaller. Jobs are grouped by approximate sequence length
and precision policy; a CUDA out-of-memory failure retries smaller batches.

Optional `--batch_max_padding_ratio` and `--batch_max_pair_tokens` limits can
flush smaller groups before `--batch_size` is reached. Both default to `0`
(disabled) and apply separately to residue and expanded structural-token
counts. A padding ratio of `1.0` requires equal counts in **both** spaces:
equal sequence lengths alone are insufficient because structural counts can
vary with residue composition. The startup log records the effective limits;
early-flush logs identify the exceeded limit and both candidate token counts.
See [padding-aware scheduling](batched_inference_optimizations.md#padding-aware-scheduling)
for a concrete equal-length example and the implications of relaxing the cap.

The implementation pads and masks the Pairformer, structural refiner and
diffusion token-transformer inputs. Atom encoding/decoding, reference geometry,
chain mappings and confidence calculations remain per job with their original
dimensions. Seeds and sampling streams are independent of batch membership.
No MSA disables external alignments; the existing query-only MSA features remain.

### Optional first-sample screening and batch refill

Both features are **off by default**. Screening completes the first generated
sample (all diffusion steps and confidence), then skips later samples when its
native `ipsae_min_10` is zero or strictly below the chosen threshold. Equality
with a nonzero threshold continues. An undefined score continues; a complex with
more than two protein chains needs an explicit `ipsae_interface` to produce a
scalar score. This is a screening heuristic: a low first score does not prove
that later samples would also be low.

```bash
LAYERNORM_TYPE=torch opendde pred \
  -i examples/cofold/no_msa_batch/refill_smoke.json \
  -o output/refill_smoke \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --sample_diffusion_chunk_size 1 --confidence_sample_chunk_size 1 \
  --early_stop_ipsae_threshold 0.05 --batch_refill true
```

The fixture contains four synthetic equal-length two-protein-chain jobs with
independent seeds. It exercises execution, not structure accuracy. The command
keeps default TMK3/cuEquivariance kernels and replay off. It explicitly changes
both chunk sizes to one, as required for optional screening.

Without `--batch_refill true`, screening preserves the original batch: it stops
only if **every** input qualifies after sample one. A mixed batch generates all
requested samples for all its inputs. With refill, each job is screened
independently and a stopped or finished slot can accept another job between
complete samples. Surviving jobs retain their slot and Torch/NumPy RNG streams.
Incoming jobs still pay their full trunk/recycling cost; this does not terminate
a diffusion trajectory early. There is no sampling work to save with `--sample 1`,
and refill rejects that setting.

Only consecutive jobs with **both equal residue counts and equal expanded
structural-token counts** share a refill session. Atom counts may differ. Trunk
preparation and diffusion retain the configured batch size through partial tails,
using placeholders that emit no outputs. At most B active and B-1 ready jobs are
retained between trunk preparations. The pair-token budget must fit the full B
in both token spaces. Refills rebuild batch conditioning/projection caches and
any optional replay graph; a surviving job never uses its former neighbour's
conditioning. No GPU overlap of trunk preparation and sampling is implied.

Early confidence and queued conditioning can increase peak memory. A refill OOM
fails unfinished jobs in that exact-shape session without reducing B or retrying
published outputs; later shapes may proceed. A graph-capture failure is fatal to
the worker. Output-writer errors remain attached to their job and fail the CLI.

Screening and refill support CPU and CUDA. Both chunk sizes must be explicitly
one when screening multiple samples. The
chunk setting itself can change predictions relative to the default diffusion
chunk of five; compare screened and unscreened runs with the **same** chunks,
precision, backend, seeds and fixed batch dimensions, including tail placeholders.
Screening selects native batching even at `--batch_size 1`, so its baseline differs
from the legacy serial CLI path. MSA, templates, RNA MSA, TFG, Fold-CP, multiple
model seeds within a forward and nonzero dataloader worker counts are rejected.
Per-job `modelSeeds` and independent processes on different GPUs remain supported.

Summaries include `sample_generation_index` (before confidence ranking) and
`sampling_early_stop` with the first score, threshold, policy, reason, and requested,
completed and skipped counts. `predictions/sampling_status.json` carries the same
metadata and `status: "complete"`. It is published atomically with the entire
prediction directory, replacing any old sample files. Resume tools must honor
this record rather than demand all originally requested samples. External resume
scripts are not changed automatically.

Python callers can set `infer_setting.early_stop_ipsae_threshold` (default `None`)
and `infer_setting.batch_refill` (default `False`). Use `runner.predict_refill` for
a refill stream: it accepts batches of `(key, data, seed)` and an
`on_prediction(key, prediction)` callback. `predict_batch` handles a finite batch
and does not refill it. Dotted configuration also exposes
`infer_setting.sample_diffusion_chunk_size`; the CLI flag is
`--sample_diffusion_chunk_size`.

Validation and remaining measurement limits are recorded in the
[screening/refill validation report](inference_screening_validation.md).

### Optional denoiser CUDA replay

`--diffusion_cuda_graph true` captures the complete denoiser once per sampling
shape and replays it at each diffusion step. It defaults to `false` and requires
the native no-MSA batching mode, including its restrictions on templates, TFG
and Fold-CP. The flag selects that mode even with `--batch_size 1`. CPU and
explicit deterministic-algorithm mode run eagerly; missing static pair caches
also select eager denoising with a warning.

```bash
opendde pred -i examples/cofold/no_msa_batch/inputs.json -o ./output/replay \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 1 --cycle 10 --step 200 --diffusion_cuda_graph true
```

Random draws, rotations, Euler updates, the trunk and confidence stay outside
the graph. Atom aggregation reuses a request-local sorting/segment plan while
preserving the ordered reduction. Captured input/output buffers are released
before confidence; a changed sample-chunk shape replaces the previous graph.
Logs report captures, replays, eager calls and setup time. CUDA replay uses no
automatic length buckets and does not change precision settings.

Benefits depend on input size: shared-GPU measurements showed about 10–15%
less model time on the small 7r6r example, but no measurable speedup on 7wux
at 2,300 structural tokens. Replay reduces kernel-launch overhead; larger
inputs spend more time executing the kernels themselves. These measurements
do not establish a general speedup; idle-GPU timing remains unverified.

Compare against **native eager batching with the same grouping**, for example
`--batch_size 2` for a one-job input, rather than the legacy serial path. Capture
adds setup time and memory. Any capture failure, including capture-time OOM,
stops the worker before another batch or input file runs. Restart that worker
with replay disabled: Torch 2.7 can leave its CUDA capture state invalid.
Python callers receive `DenoiserGraphCaptureError` and must also restart their
worker process; catching it does not make the CUDA context safe to reuse.
OOM before capture retains the ordinary batch split-and-retry behavior.
A singleton that runs out of memory during warmup is reported as failed;
there is currently no automatic retry with replay disabled.
See [validation and compiler experiments](batched_inference_optimizations.md).

To distribute batches across GPUs, launch one worker per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 LAYERNORM_TYPE=torch \
  torchrun --standalone --nproc_per_node 2 -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/inputs.json -o ./output/batched_2gpu \
  --device cuda --use_msa false --use_template false --use_rna_msa false \
  --foldcp_mode single --batch_size 2 --sample 3
```

For padding limits, optimized singleton groups and validation results, see
[the inference optimization report](batched_inference_optimizations.md).

Each worker owns a disjoint subset of jobs and batches locally. This mode
requires no MSA/templates/RNA MSA, no TFG guidance, `foldcp_mode=single`,
`num_workers=0`, and `model.N_model_seed=1`; multiple per-job `modelSeeds` are
supported. Trunk caching through `OPENDDE_TRUNK_CACHE` is unsupported.
Triangle kernels support PyTorch and cuEquivariance for batches; `auto` selects
cuEquivariance when its compatible CUDA packages are available and otherwise
uses PyTorch. Confidence chunks default to `5`, with higher potential peak
memory than `1`; reduce `--confidence_sample_chunk_size` explicitly when needed.
Native `ipsae_min_10` is included in summary JSON even with
`--need_atom_confidence false`. See [batch optimizations and score conventions](batched_inference_optimizations.md).
CPU execution is covered by small-model tests;
batched MPS execution has not been validated.

Even with `foldcp_mode=single`, `torchrun` initializes NCCL for job coordination.
If startup fails in `_create_foldcp_control_groups`, rerun with `NCCL_DEBUG=INFO`
to identify the underlying transport error. An eight-GPU H200 startup probe
with NCCL 2.26.2 failed in `transport/nvls.cc` with CUDA error 401
(`the operation cannot be performed in the present state`), while two ranks
passed. Disabling NVLink SHARP (NVLS) allowed all eight ranks to finish startup:

```bash
export NCCL_NVLS_ENABLE=0
# Run the same torchrun command with the existing inference arguments.
```

This is a workaround for that NVLS initialization failure, not a general NCCL
fix or an inference precision change. See NVIDIA's
[NCCL environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable).
The probe validated startup only; it did not measure folding throughput.
For diagnosis, inspect the Fabric Manager log as well as `nvidia-smi`: a fabric
status of `Completed / Success` does not establish that NVLS multicast setup
works. In the reproduced failure, Fabric Manager reported a missing member GPU
handle and requested a reset of all GPUs in the partition. Its version matched
the driver, but it had restarted since the last host boot. See NVIDIA's
[NVLS troubleshooting guidance](https://docs.nvidia.com/cluster-readiness-engine/operations/troubleshooting#nccl-aborts-on-an-nvls-multicast-bind-failure)
for recovery; a GPU reset or host reboot requires coordinated maintenance.
Already-sharded workloads can also run one independent process per GPU using
`CUDA_VISIBLE_DEVICES`, with distinct input shards and output directories and
without `torchrun` or inherited distributed rank environment variables.

Batching increases peak memory and may not improve speed for every workload.
Use `--dtype fp32` for numerical comparisons, keep seeds and sampling settings
fixed, and compare within tolerance: changing matrix shapes can change floating
point results even with `--deterministic true`. See the
[mixed-length fixtures](../examples/cofold/no_msa_batch/README.md) for validation
commands and tested settings.

## Multi-GPU Fold-CP inference

> [!IMPORTANT]
> Fold-CP inference does not currently support cuEquivariance (`cueq`) triangle
> kernels. Select the distributed PyTorch implementations with
> `--triatt_kernel torch --trimul_kernel torch`. In distributed mode, `auto`
> resolves to these PyTorch kernels and an explicit `cueq` request fails before
> model loading. On CUDA BF16, Fold-CP triangle
> attention also uses Triton 3.3.1 from the GPU install extra to fuse
> attention-bias addition; this Triton helper is separate from cuEquivariance.

Fold-CP distributes token-pair-heavy inference work over a `1 x P` mesh, where
`P` can be any available GPU count greater than one. Launch it with `torchrun`
and expose exactly the GPUs you want to use. For example, four GPUs use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i examples/cofold/protein_200.json \
  -o ./output_foldcp \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --trimul_kernel torch \
  --triatt_kernel torch \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Runtime notes:

- `--nproc_per_node P` must match `--foldcp_size_dp 1` times
  `--foldcp_size_cp P`.
- `--foldcp_size_cp P` creates a `1 x P` context-parallel mesh. The example
  uses `P=4`; other GPU counts require changing both `4` values.
- Keep the input, model, dtype, cycle, step, sample, MSA, and template settings
  identical when comparing single-GPU and Fold-CP outputs. Select
  `--triatt_kernel torch --trimul_kernel torch` for both runs; the distributed
  CUDA BF16 path additionally uses Triton for attention-bias fusion.
- Outputs are written under the requested `-o/--out_dir` just like normal
  inference.
- Optional `--foldcp_metrics_jsonl path/to/metrics.jsonl` records Fold-CP timing
  and memory metrics.

For single-GPU inference, omit the Fold-CP flags or set
`--foldcp_mode single --foldcp_size_cp 1`.

Use prepared features:

```bash
opendde pred -i examples/cofold/template/example_9fm7.json \
  -o ./output -n opendde_v1 \
  --use_msa true --use_template true

opendde pred -i examples/cofold/rna_msa/example_9gmw_2.json \
  -o ./output -n opendde_v1 \
  --use_rna_msa true
```

## Optional TFG Guidance

OpenDDE includes default-off Training-Free Guidance (TFG). TFG refines each
sampled trajectory with geometry potentials while keeping the requested
`--sample` count unchanged.

```bash
opendde pred -i examples/cofold/input.json -o ./output -n opendde_v1 \
  --use_tfg_guidance true
```

Outputs are written to:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

With `--use_tfg_guidance true`, the input JSON `constraint` field (`contact`,
`movable_chains`, `epitope`) additionally steers the sampler towards partial
knowledge of an antibody–antigen interface. See
[TFG constraint guidance](./tfg_constraint_guidance.md), the `constraint` section
of [infer_json_format.md](./infer_json_format.md) and
[`examples/tfg`](../examples/tfg/README.md).

## Common flags

| Flag | Meaning |
| --- | --- |
| `-n`, `--model_name` | Model name. Currently `opendde_v1`. |
| `--load_checkpoint_path` | Explicit checkpoint path. |
| `--seeds` | Comma-separated seeds, e.g. `101,102`. Overrides the job's `modelSeeds`; if unset, `modelSeeds` are used, or a random seed when both are absent. |
| `--batch_size` | Maximum independent complexes per GPU forward; default `1`. Values above `1` require the no-MSA batching mode described above. |
| `--diffusion_cuda_graph` | Experimental request-scoped denoiser replay; default `false`. Selects native no-MSA batching even for singletons. |
| `--sample` | Number of diffusion samples per complex, independent of `--batch_size`. |
| `--sample_diffusion_chunk_size`, `--confidence_sample_chunk_size` | Defaults `5` / `5`. Screening requires both to be `1`. |
| `--early_stop_ipsae_threshold` | Optional first-generated-sample ipSAE cutoff; omitted by default. |
| `--batch_refill` | Optional fixed-slot refill after screening; default `false`. Requires multiple jobs and samples. |
| `--use_msa` | Use/generate protein MSA features. |
| `--use_template` | Use/generate template features. |
| `--use_rna_msa` | Use/generate RNA MSA features; requires `--use_msa true`. |
| `--use_tfg_guidance` | Enable Training-Free Guidance. |
| `--foldcp_mode` | `single` or `distributed`; use `distributed` with `torchrun` for multi-GPU Fold-CP inference. |
| `--foldcp_size_dp` | Compatibility option; only `1` is supported. Runtime `2 x 2` topology is not maintained. |
| `--foldcp_size_cp` | Context-parallel degree `P` in the maintained `1 x P` mesh; must match the launched process count. |
| `--foldcp_devices` | Optional visible-device list recorded in Fold-CP metrics; actual GPU visibility is controlled by `CUDA_VISIBLE_DEVICES`. |
| `--foldcp_metrics_jsonl` | Optional JSONL path for Fold-CP timing and memory metrics. |
| `--dtype` | `bf16` or `fp32`. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`; auto uses CUDA when available, then Apple MPS, and otherwise CPU. |
| `--trimul_kernel`, `--triatt_kernel` | Defaults `flashpairformer` (TMK3) / `cuequivariance`; `auto` and `torch` remain available. |

Run `opendde <command> --help` for the full option list.
