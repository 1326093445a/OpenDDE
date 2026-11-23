# Batching and padding

**When running multiple designs together, explicitly set a padding cap.**
Use `--batch_max_padding_ratio 1.0` to match the UBC-001 batching setup and avoid
cross-design token padding. This is a recommended setting, not a new default
or a requirement enforced by the CLI.

## Defaults stay unchanged

- `--batch_size 1`: one design per forward; no cross-design padding.
- `--sample 5`: five predictions for that design, not five separate designs.
- `--batch_max_padding_ratio 0.0`: padding cap disabled.
- Protein MSA is on; templates and RNA MSA are off.

Torch LayerNorm, TMK3 multiplication, cuEquivariance attention and diffusion/
confidence chunks of five remain the defaults. Early stopping, refill and CUDA
replay remain off, so all five samples are generated.

## Recommended multi-design command

```bash
opendde pred -i inputs.json -o output/batched \
  --batch_size 2 --batch_max_padding_ratio 1.0 \
  --use_msa false --use_template false --use_rna_msa false
```

Each design still gets five samples. A ratio of `1.0` admits designs to the same
batch only when **both residue counts and expanded structural-token counts
match**. Equal sequence lengths alone may be insufficient. The existing
seed/length ordering determines candidate neighbours; this cap does not globally
regroup all equal-size designs. Batch size is an upper limit, so a job may run as
a native singleton.

OpenDDE supports different lengths using padding and masks when the cap is
relaxed. However, changing tensor shapes can change BF16 predictions even when
masking is correct. Keep batch grouping, sample chunks, precision and backends
fixed for numerical comparisons. Setting the ratio above `1.0` permits some
padding; `0.0` disables the limit. Neither setting guarantees equal predictions
to an unpadded run. A native singleton also differs from the legacy
`--batch_size 1` path.

The cap applies separately to residue and structural-token spaces, using the
ratio of padded to unpadded cubic work. It is not a GPU-memory limit. The optional
`--batch_max_pair_tokens` provides a separate quadratic-work budget. Startup and
flush logs show which limits are active and why a batch was split.

## Checkpoint selection is separate

The default checkpoint remains `opendde.pt`. To reproduce a run that used
`opendde_abag.pt`, explicitly select that file with `--load_checkpoint_path`.
Changing batching flags does not change checkpoints.

See [inference instructions](inference_instructions.md#multiple-complexes-per-gpu)
for supported modes and [the kernel guide](kernels.md) for GPU dependencies.
