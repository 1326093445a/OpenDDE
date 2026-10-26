# Variable-length complexes without MSA

[`inputs.json`](inputs.json) contains eight independent protein-complex jobs.
Each has two distinct synthetic chains, one copy of each, and inference seed
`101`. [`chains.fasta`](chains.fasta) contains the same sequences for inspection
or use in other tools; its headers identify `<job name>|<chain ID>`. Use the JSON
with `opendde pred`, which does not accept this FASTA as prediction input.

These are execution and memory fixtures, not known interacting proteins or a
structure-accuracy benchmark. Sequences were generated locally with Python's
`random.Random(20261021)`: for each chain in table order, repeat
`ACDEFGHIKLMNPQRSTVWY`, truncate to its length, and shuffle the residues using
that shared generator. No RCSB download is needed.

| Job | Chain A residues | Chain B residues | Total residues |
| --- | ---: | ---: | ---: |
| `synthetic_complex_32_32` | 32 | 32 | 64 |
| `synthetic_complex_32_64` | 32 | 64 | 96 |
| `synthetic_complex_64_64` | 64 | 64 | 128 |
| `synthetic_complex_64_128` | 64 | 128 | 192 |
| `synthetic_complex_128_128` | 128 | 128 | 256 |
| `synthetic_complex_128_256` | 128 | 256 | 384 |
| `synthetic_complex_256_256` | 256 | 256 | 512 |
| `synthetic_complex_256_512` | 256 | 512 | 768 |

Residue totals are not expanded structural-token or atom counts. Record those
dimensions from actual featurization when comparing memory usage.

## Single-GPU baseline

Run from the repository root after installing OpenDDE and preparing its
checkpoint and common molecular data. No MSA or template search databases are
required with these flags. The input JSON contains no external feature paths.

```bash
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=torch python -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/inputs.json \
  -o ./output/no_msa_batch_smoke \
  --device cuda --use_msa false --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch \
  --sample 1 --step 20 --cycle 1 --deterministic true
```

This reduced-step run checks execution, not prediction quality. For a full
sampling baseline, use `--sample 5 --step 200 --cycle 10` and a separate output
directory. Eight jobs with one seed and five samples produce 40 predicted
structures if every job succeeds. Confidence scoring and diffusion chunking
still determine how much of that work runs simultaneously.

## Multiple complexes per GPU

Add `--batch_size 2` to the baseline command and use a separate output directory.
With `--sample 3`, each full batch has two complexes and three structures per
complex. Pairformer, structural refiner and diffusion token-transformer calls
run with a job batch dimension. Atom topology and confidence processing remain
per job. Different lengths are padded only at the token-network boundaries and
are masked before attention; output atom and residue axes retain their original
sizes.

Start comparisons with `--dtype fp32` on both runs. Batch size defaults to `1`.
Large batches can exceed GPU memory; forward OOMs trigger smaller-batch retries.

## Multi-GPU job-distribution check

The following is a baseline to validate on the target machine. Select GPUs
that are available when running it; the device list is an example.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 LAYERNORM_TYPE=torch \
  torchrun --standalone --nproc_per_node 4 -m runner.batch_inference pred \
  -i examples/cofold/no_msa_batch/inputs.json \
  -o ./output/no_msa_batch_4gpu_smoke \
  --device cuda --use_msa false --use_template false --use_rna_msa false \
  --trimul_kernel torch --triatt_kernel torch \
  --foldcp_mode single --foldcp_size_dp 1 --foldcp_size_cp 1 \
  --batch_size 2 --sample 3 --step 20 --cycle 1 --deterministic true
```

Here `single` keeps each model forward on its local GPU. The inference sampler
uses the process world size to distribute whole jobs across ranks. Keep all
jobs in this one JSON: the runner processes a directory's JSON files in order,
so separate one-job files would give this sampler little useful work to shard.
`foldcp_size_dp` remains `1`; it is a Fold-CP compatibility option, not the
worker count for this launch. Fold-CP's `distributed` mode instead makes GPUs
cooperate on the same job.

Each worker batches its assigned jobs locally. Put all jobs in one JSON;
separate files are processed sequentially and cannot share a batch. Multi-input
batching requires no external MSA/templates/RNA MSA, no TFG guidance, no trunk
cache, and the PyTorch triangle kernels. See the complete
[batching constraints](../../../docs/inference_instructions.md#multiple-complexes-per-gpu).

For later comparisons, keep seeds, samples, cycles, steps, dtype, and kernels
fixed. Check that every job produces the expected outputs without duplication,
compare coordinates/confidence with the single-GPU baseline within numerical
tolerance, and measure completed jobs per second plus per-GPU peak memory.
Model loading and feature preparation should be reported separately from model
execution. Check `full_data` residue-pair shapes, atom pLDDT lengths, CIF atom
and chain identities, summary counts and sample rankings for every job.

## Validation performed

The released `opendde_v1` checkpoint was tested on H200 GPUs with PyTorch
triangle kernels, TF32 off, one recycle and 20 diffusion steps:

- Single-GPU batches of two, with a partial final batch, and uneven two-GPU
  sharding completed for the first three fixtures.
- The opt-in checkpoint test below compares three jobs with three samples each
  against independent FP32 forwards. It checks coordinates, confidence logits,
  summary scores and chain/atom mappings, and asserts an actual batched trunk
  call.
- All eight fixtures completed on two GPUs with `--batch_size 2 --sample 3
  --dtype fp32`, producing exactly 24 CIFs and 48 confidence JSONs. File checks
  verified the original sequences, atom counts, residue-pair dimensions, finite
  arrays, atom-pLDDT/CIF association and independent sample rankings. All four
  batches remained batched; peak allocated memory reached 82.22 GiB.
- BF16 completed for the first three fixtures, but changing batch membership
  changed coordinates more than FP32: the padded 64-residue job reached 0.44 Å
  aligned RMSD in that comparison. Fixed seeds do not imply bitwise equality
  across batch shapes. Use FP32 for strict numerical checks.

Run the checkpoint regression with cached runtime assets and an available GPU:

```bash
OPENDDE_RUN_BATCH_SMOKE=1 LAYERNORM_TYPE=torch python -m pytest \
  tests/smoke/test_input_batching_runtime.py -q
```

CPU tests in `tests/test_input_batching.py` and `tests/test_batched_prediction.py`
cover padding masks, equal job/sample/channel sizes, independent random streams,
sample chunking, cache/fusion modes, per-job seeds, output boundaries, rank
ownership, partial batches and OOM retries. A real large-batch OOM also verified
the retry path before the attention allocation fix.

These are execution and numerical-consistency checks. Full 200-step/10-recycle
prediction quality, throughput gains, MPS batching and batches with external
features have not been validated by these runs.
