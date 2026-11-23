# OpenDDE-Preview

![OpenDDE banner](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/OpenDDE.png)

![Status](https://img.shields.io/badge/status-preview-orange)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

OpenDDE is an open-source, all-atom biomolecular foundation model that turns co-folding into a scalable engine for structure prediction, design, and optimization in drug discovery.

> [!IMPORTANT]
> OpenDDE is a preview release. CLI flags, input/output JSON fields, and released
> checkpoints may change between versions, and predictions are not guaranteed to
> be reproducible across releases. It is not yet intended for production
> pipelines. Please open an issue for bugs, regressions, or feature requests.

![OpenDDE banner](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/scaling_law.png)
![results](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/results.png)

## News

- **2026-10-09: We release OpenDDE v1.2.0 with constraint-guided (conditional) inference! Read the [technical report](https://github.com/aurekaresearch/OpenDDE/blob/main/assets/OpenDDE_Conditional_Inference.pdf).**
    - Steer the sampler with partial knowledge of an antibody–antigen interface (contacts, movable chains, epitope) via `--use_tfg_guidance true`, without changing the weights. See [docs/tfg_constraint_guidance.md](docs/tfg_constraint_guidance.md)
    - Ready-to-run examples are available in [examples/tfg](examples/tfg), and the new antibody–antigen panel lists are in [benchmarks/OpenBench2026](benchmarks/OpenBench2026)

- **2026-07-03: We release OpenDDE-preview (co-folding)! Read the [technical report](https://arxiv.org/abs/2607.03787) and visit the [website](https://aurekaresearch.github.io/OpenDDE-Website).**
    - Model weights can be downloaded from Hugging Face: [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt) | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt)
    - The Docker image can be pulled with `docker pull aurekaresearch/opendde:v1`
    - The 2026ARK-AB Benchmark is now available

## Installation

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

See the [Docker guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/docker_installation.md)
for GPU setup, runtime-data mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU or on the Metal (MPS) backend with `--device mps`; Intel
> macOS is unsupported, and Windows has not been validated. At runtime,
> `--device auto` uses CUDA when available, then MPS, and otherwise falls back
> to CPU.

For runtime-data setup and additional installation details, see the
[inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md).

## Model and Runtime Data

OpenDDE reads checkpoints and runtime assets from `OPENDDE_ROOT_DIR`, defaulting
to `~/.cache/opendde` when the environment variable is unset:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA preprocessing
```

From a source checkout, use the repository helper script to prepare the runtime
layout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

The helper script lives in the source tree and is not installed with the
`opendde` Python package. If you installed OpenDDE from a wheel or package
index, either run the script from a cloned checkout, or let `opendde pred`
download the default checkpoint and common runtime files when they are missing.
You can also place checkpoint files manually under
`$OPENDDE_ROOT_DIR/checkpoint/`.

Python-managed checkpoint and common-asset downloads use a release-pinned
revision, published size, and SHA-256 before atomic replacement. The source
helper independently verifies released checkpoints; external search databases
are prepared separately. A checkpoint supplied with `--load_checkpoint_path`
is never replaced automatically.

For a prediction that disables protein MSA, template search, and RNA MSA, the
large `search_database/` files are not needed. From a source checkout, use:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

Released checkpoints:

| Checkpoint        | Use case                              | Download                                                                                                                          |
| ----------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `opendde.pt`      | General-purpose checkpoint.           | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt)           |
| `opendde_abag.pt` | Checkpoint tuned on antibody-antigen. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt) |

Use `opendde.pt` with `-n opendde_v1` for the default model. For ABAG runs,
keep the filename as `opendde_abag.pt` and pass it explicitly:

```bash
opendde pred \
  -i input.json \
  -o ./output \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Detailed asset setup, mirrors, and Docker data mounts are documented in
[inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md).

## Running Your First Prediction

Save this minimal OpenDDE input as `tiny.json`:

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

Run a small compatibility-oriented prediction. This disables external feature
searches, so only the checkpoint and common runtime files are required:

```bash
opendde pred \
  -i tiny.json \
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

Inference defaults to `bf16` with TF32 off, TMK3 multiplication, cuEquivariance
attention, and diffusion/confidence chunks of five. Seeds come from the job's
`modelSeeds` unless `--seeds` is provided. The compatibility example above explicitly
uses Torch kernels so it also works without the optional CUDA dependencies.

Outputs are written under:

```text
output/<job_name>/seed_<seed>/predictions/
```

For runs using external features, select `--batch_size 1` and enable the
preprocessing features you need, for example `--use_msa true`, `--use_template true`, or `--use_rna_msa true`. Those paths may
require network access, HMMER/Kalign binaries, and large local search databases;
see the inference guide for details.

## Optimized No-MSA Inference

**The default generates all five samples:** Torch LayerNorm, TMK3 triangle
multiplication, cuEquivariance triangle attention, and diffusion/confidence
chunks of five. Batch size remains one, the padding cap is disabled, and protein
MSA is on. Templates, RNA MSA, early stopping, batch refill and CUDA replay are off.
For multiple designs, explicitly set `--batch_size 2`,
`--batch_max_padding_ratio 1.0` and `--use_msa false`; see
[batching and padding guidance](docs/batching_and_padding.md). BF16 mixed
precision, reference diffusion attention and existing caches/fusions remain in
use; compiler experiments stay benchmark-only. The length-dependent precision
policy is unchanged. Install the [GPU kernel dependencies](docs/kernels.md#optional-tmk3-triangle-multiplication)
for the default CUDA path; no Torch upgrade is needed.

| Improvement | How to use it |
| --- | --- |
| Multiple variable-length complexes per GPU forward, with independent seeds and per-job confidence/output splitting | `--batch_size N` in native no-MSA mode |
| Reuse of diffusion conditioning and projections, including native singleton batches | Enabled by default in the native batched path, within the configured cache budget |
| Ordered atom aggregation for repeatable CUDA inference | Enabled in the ordinary CUDA inference path; historical BF16 predictions can change |
| Grouped ipSAE for selected protein-chain interfaces | Supported through each job's `ipsae_interface`; existing jobs keep their scoring behavior |
| Efficient diffusion attention layout | Opt in with `--diffusion_attention_backend efficient`; floating-point outputs can change |
| TMK3 triangle multiplication | Default `--trimul_kernel flashpairformer`; requires the optional GPU backend dependencies. Explicit `auto`, `cuequivariance` and `torch` overrides remain available. |
| Complete-denoiser CUDA replay with cached atom-aggregation plans | Opt in with `--diffusion_cuda_graph true`; defaults off |
| First-sample ipSAE screening and equal-length batch refill | Optional `--early_stop_ipsae_threshold` and `--batch_refill true`; see the [screening command and constraints](docs/inference_instructions.md#optional-first-sample-screening-and-batch-refill). Requires multiple samples to save work. |
| Diffusion or Pairformer compilation | Benchmark-only; never enabled by prediction defaults |

### Default eager batching example

Run from the repository root after installing the GPU kernel dependencies and
preparing the checkpoint and common runtime data. The supplied
[`flashpairformer_smoke.json`](examples/cofold/no_msa_batch/flashpairformer_smoke.json)
contains two synthetic two-chain complexes with **128 and 192 total residues**,
each with seed 101. These are execution fixtures, not an accuracy benchmark.

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/flashpairformer_smoke.json \
  -o ./output/eager_example \
  --device cuda \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --batch_max_padding_ratio 1.0 --cycle 10 --step 200
```

This uses TMK3 multiplication, cuEquivariance attention and chunks of five without
extra kernel flags. It generates **five structures per complex**, with separate
CIF and confidence outputs under `output/eager_example/<job_name>/seed_101/`.
For a quick execution check, reduce cycles/steps. This command explicitly sets
the batch limit to two and padding cap to 1.0; CLI defaults remain one and 0.0.
These unequal-length inputs run as native singletons with the recommended cap;
equal counts in both token spaces permit a two-job forward.
Screening, refill and replay are disabled.

Different lengths are supported, but padding can change BF16 predictions.
Keep grouping fixed for reproducibility, and use the padding-ratio and pair-token
limits described in the [batching guide](docs/inference_instructions.md#multiple-complexes-per-gpu)
when larger batches waste work or exceed memory. See the
[default/optional settings](docs/inference_instructions.md#default-inference-baseline)
for the complete policy.

<a id="optional-flashpairformer-with-eager-inference"></a>

### Explicit FlashPairformer settings

Use **TMK3 triangle multiplication + cuEquivariance triangle attention**, with
the existing caches/fusions, reference diffusion attention and replay disabled.
This does not compile the model or change LayerNorm or the precision policy.
Install the optional backend using the [kernel guide](docs/kernels.md#optional-tmk3-triangle-multiplication)
if it is not already available; no Torch upgrade is required.

The supplied [FlashPairformer fixture](examples/cofold/no_msa_batch/flashpairformer_smoke.json)
contains the existing 128- and 192-residue synthetic jobs, both above TMK3's
100-token dispatch cutoff:

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/flashpairformer_smoke.json \
  -o ./output/flashpairformer_eager \
  --device cuda --dtype bf16 \
  --use_msa false --use_template false --use_rna_msa false \
  --trimul_kernel flashpairformer --triatt_kernel cuequivariance \
  --enable_cache true --enable_fusion true --enable_tf32 false \
  --diffusion_attention_backend reference --diffusion_cuda_graph false \
  --batch_size 2 --sample 5 --cycle 10 --step 200 \
  --sample_diffusion_chunk_size 5 --confidence_sample_chunk_size 5
```

Replace the input path with your jobs and choose the batch limit for your
workload. Keep grouping fixed for comparisons. FP32 confidence retains its
fallback kernels; TMK3 does not force it to BF16. These explicit settings match
the new defaults. Confidence chunks of five can use more peak memory than one;
lower the chunk size explicitly when needed and keep it fixed for comparisons.

### Optional replay and compiler experiments

To opt into replay, add `--diffusion_cuda_graph true` to the eager command above
and choose a different output directory. Keep the batch size, grouping, seed
and other settings fixed when comparing results. Removing the flag restores
the default eager execution.

Replay reduced model time by about **10–15% on the small 7r6r example** on shared
GPUs, but gave **no measurable speedup on 7wux at 2,300 structural tokens**.
It reduces launch overhead; larger kernels can dominate runtime. Tested replay
outputs matched native eager bitwise, including confidence. These results do
not guarantee speedups or equivalence for every input. The BF16 configuration
retains the existing mixed-precision policy, including FP32 diffusion/confidence
for the tested lengths; replay does not change precision.

Replay requires native no-MSA mode, excludes templates, RNA MSA, TFG and Fold-CP,
and stays eager with explicit
deterministic algorithms. Capture failures stop the worker; restart with replay
disabled. Whole-block compilation remains a benchmark-only experiment: the
BF16 Pairformer experiment changed 7r6r by 9.10 Å aligned RMSD and changed
confidence, despite only about a 6% shared-GPU timing improvement. It is not
part of the default or a recommended equivalent backend. See the
[experiment results](docs/batched_inference_optimizations.md#rfd4-inspired-whole-pairformer-compilation-experiment-2026-11-13).

### Distribute independent jobs across GPUs

The larger [`inputs.json`](examples/cofold/no_msa_batch/inputs.json) contains
eight variable-length jobs. This example assigns them to two GPU workers, each
using local batches of up to two jobs:

```bash
CUDA_VISIBLE_DEVICES=0,1 LAYERNORM_TYPE=torch \
  torchrun --standalone --nproc_per_node 2 -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/inputs.json \
  -o ./output/batched_2gpu \
  --device cuda --dtype bf16 --foldcp_mode single \
  --use_msa false --use_template false --use_rna_msa false \
  --batch_size 2 --sample 1 --cycle 10 --step 200
```

Here each forward runs on one GPU; Fold-CP below instead splits one job across
GPUs. Eight queued jobs do not imply an eight-job forward. Eight-job CUDA capture
and idle-GPU replay throughput remain unverified. See the
[optimization measurements](docs/batched_inference_optimizations.md) and
[replay usage and failure handling](docs/inference_instructions.md#optional-denoiser-cuda-replay)
for tested settings, limitations and benchmark commands.

## Multi-GPU Fold-CP Inference

> [!IMPORTANT]
> Multi-GPU Fold-CP does not support cuEquivariance triangle kernels, so use
> `--trimul_kernel torch --triatt_kernel torch`. Distributed `auto` requests
> resolve to these PyTorch kernels, while an explicit cuEquivariance request is
> rejected before model loading. On CUDA BF16, the distributed
> PyTorch triangle-attention path uses the Triton dependency from the GPU install
> extra to fuse attention-bias addition. See the
> [Fold-CP reproduction guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/foldcp_e2e_baseline.md)
> for controlled launch commands and validation guidance.

OpenDDE supports a `1 x P` Fold-CP inference mode for larger inputs, where `P`
can be any available GPU count greater than one. Launch it with `torchrun` so
that one process runs on each GPU. For example, four GPUs use:

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
  --batch_size 1 --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Only the `1 x P` topology is supported. `--nproc_per_node` must equal
`--foldcp_size_cp P`; `--foldcp_size_dp` is retained only for command-line
compatibility and must remain `1`. The example uses `P=4`; replace both
occurrences of `4` with the desired `P`. For normal single-GPU or CPU inference,
omit the Fold-CP flags or use `--foldcp_mode single --foldcp_size_cp 1`.

## Input JSON

OpenDDE input is a top-level list of jobs. Each job contains `sequences` entries
such as `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or `ion`.
`covalent_bonds` is optional and should be added only when explicit covalent
links are needed.

See the [input JSON format](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/infer_json_format.md) for the full schema,
including covalent bonds, ligands, modifications, MSA paths, and template paths.

## CLI Overview

```bash
opendde pred    # run inference
opendde doctor  # inspect Python/CUDA/kernel setup
opendde json    # convert PDB/CIF structures to OpenDDE JSON
opendde msa     # protein MSA preprocessing
opendde mt      # protein MSA + template preprocessing
opendde prep    # protein MSA + template + RNA MSA preprocessing
```

Use `opendde <command> --help` for command-specific options. Public model names
currently include `opendde_v1`; use `--load_checkpoint_path` for alternate
checkpoint files such as `opendde_abag.pt`.

## Documentation

- [Changelog](https://github.com/aurekaresearch/OpenDDE/blob/main/CHANGELOG.md)
- [Model/checkpoint manifest](https://github.com/aurekaresearch/OpenDDE/blob/main/opendde/config/model_manifest.json)
- [Inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md)
- [Docker installation](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/docker_installation.md)
- [Input JSON format](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/infer_json_format.md)
- [TFG constraint guidance](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/tfg_constraint_guidance.md)
- [MSA/template/RNA-MSA pipeline](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/msa_template_pipeline.md)
- [Kernel options](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/kernels.md)
- [Fold-CP reproduction guide](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/foldcp_e2e_baseline.md)
- [Supported models](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/supported_models.md)
- [Tutorial](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/tutorial.md)

## Citation and Acknowledgements

If you use OpenDDE in your work, please cite this software and the related work.
OpenDDE builds on ideas and components from the AlphaFold 3 ecosystem, including
AlphaFold 3, Protenix, OpenFold, and ColabFold.

## License

OpenDDE is released under the Apache-2.0 license. See [LICENSE](https://github.com/aurekaresearch/OpenDDE/blob/main/LICENSE).

## Partnership and Collaboration

![Hiring](https://raw.githubusercontent.com/aurekaresearch/OpenDDE/main/assets/hiring.png)
