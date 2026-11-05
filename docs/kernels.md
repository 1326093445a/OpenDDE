# Kernel Configuration


OpenDDE has a safe PyTorch path and optional GPU kernels.

## LayerNorm

Default:

```bash
export LAYERNORM_TYPE=torch
```

Optional CUDA LayerNorm:

```bash
export LAYERNORM_TYPE=fast_layernorm
```

Use `torch` for CPU, debugging, or environments where CUDA extension compilation
is unavailable.

## Triangle kernels

Both triangle attention and triangle multiplication support:

| Value | Meaning |
| --- | --- |
| `auto` | Use cuEquivariance when available, otherwise PyTorch. |
| `cuequivariance` | Force cuEquivariance GPU kernels. |
| `torch` | Select OpenDDE's PyTorch triangle implementation. Distributed CUDA BF16 triangle attention also uses Triton for attention-bias fusion. |

Triangle multiplication additionally accepts **`flashpairformer`** for the
optional exact-mode TMK3 implementation. `auto` retains its existing behavior;
triangle attention does not accept this value.

CLI flags:

```bash
opendde pred \
  --triatt_kernel auto \
  --trimul_kernel auto
```

### Multi-GPU Fold-CP limitation

The current official cuEquivariance release does not support OpenDDE's
distributed Fold-CP path. Multi-GPU inference must therefore select the
PyTorch triangle implementations:

```bash
--triatt_kernel torch --trimul_kernel torch
```

On CUDA BF16, distributed triangle attention uses Triton 3.3.1 from the GPU
install extra to fuse the two attention-bias additions. Triton is part of this
Fold-CP PyTorch path; it does not enable cuEquivariance. This limitation applies
to multi-GPU Fold-CP only. Single-GPU inference may continue to use `auto` or
`cuequivariance`. TMK3 is also unsupported in distributed Fold-CP and is
rejected before GPU-kernel probing. Independent jobs on separate GPUs may use
TMK3; this is distinct from splitting one model forward across GPUs.

## Optional TMK3 triangle multiplication

Keep the production environment: the adapter is validated with **Torch
2.7.1+cu126, Triton 3.3.1 and opt_core 0.5.228.0** on H200. No Torch or Triton
upgrade is required. This validates the TMK3 path, not every kernel shipped
inside opt_core. Pytest and Gemmi are test dependencies, not prerequisites for
TMK3 inference.

The adapter is ported from Mol-1 with attribution and uses
[Anthropic's opt_core](https://github.com/anthropics/uplifting-biomolecular-modeling)
without applying its model patches. The `flashpairformer` extra declares the
required version. opt_core is not on PyPI: install the pinned source or a wheel
built from it first, alongside OpenDDE's existing GPU dependencies:

```bash
conda activate opendde
python -m pip install --no-deps "opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e#subdirectory=common/opt_core"
opendde pred -i inputs.json -o output \
  --use_msa false --use_template false --use_rna_msa false \
  --dtype bf16 --trimul_kernel flashpairformer --triatt_kernel auto \
  --batch_size 8 --batch_max_padding_ratio 1.0 --cycle 10 --step 200 --sample 1
```

Alternatively, when opt_core is not installed, set
`OPT_CORE_DIR=/path/to/checkout/common/opt_core` (the directory containing the
`opt_core` package). Loading is lazy; ordinary PyTorch/cuEquivariance execution
does not import opt_core. The dependency and JIT caches are not vendored into
OpenDDE.

TMK3 serves no-grad evaluation on CUDA, with BF16 compute and BF16 or FP32
resident pair tensors. Equal input/hidden widths divisible by 64 are required;
width 384 is covered by the OpenDDE tests. It preserves cuEquivariance's
short-sequence dispatch (at least 100 tokens) and supports leading batch axes
by evaluating each pair with shared packed weights. Residual addition remains
in OpenDDE, preserving its dtype and avoiding in-place input mutation.

Unsupported calls use cuEquivariance on CUDA when its shape is supported,
otherwise the existing PyTorch implementation. FP32 confidence calls thus
continue using their original backend and precision. Training/grad-enabled
module calls use the differentiable fallback. Explicit CUDA selection requires
the cuEquivariance dependencies; missing opt_core on an eligible TMK3 call and
unexpected kernel errors raise rather than silently switching implementation.
Only named upstream support refusals fall back.

`flashpairformer_stats(model)` in `opendde.model.triangular.flashpairformer`
reports served and fallback counts. Packed weights are outside the state dict
and are invalidated on normal parameter changes, checkpoint reloads and
device/dtype moves. Call `clear_flashpairformer_cache(model)` after external
`.data` writes that bypass PyTorch version counters, or to release the cache.
Parameters created inside `torch.inference_mode()` have no version counters;
their packed weights are rebuilt on every call so in-place updates stay safe.

See [the optimization report](batched_inference_optimizations.md) for measured
whole-model results. To validate the backend directly:

```bash
TEST_FLASHPAIRFORMER_CUDA=1 LAYERNORM_TYPE=torch \
  python -m pytest tests/test_flashpairformer.py -ra
```

### cuEquivariance on-demand tuning

An alternative experiment is `CUEQ_TRITON_TUNING=ONDEMAND` with
`CUEQ_TRITON_CACHE_DIR` pointing to a dedicated cache directory. First calls
may spend minutes tuning; measure subsequent warmed passes separately. Keep
this opt-in and compare output numerics as well as timing. No tuning variable
is enabled by OpenDDE automatically.
An isolated width-384 probe completed under Torch 2.7.1/Triton 3.3.1 and its
cached tuned outputs matched the default kernels in both directions. The
subsequent full-model H200 comparison found **7.605 s for TMK3 versus 7.636 s
for tuned cuEquivariance**, with an untuned baseline of **8.006 s**. The two
optimized choices were too close to establish a decisive speed winner.
Full-workload tuning also changed confidence values at small FP32 rounding
scales, while TMK3 preserved the default outputs bitwise. Keep `auto` as the
general default; see the report for setup costs, memory and workload limits.

## Compatibility run

```bash
LAYERNORM_TYPE=torch opendde pred \
  -i examples/cofold/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --triatt_kernel torch \
  --trimul_kernel torch \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --dtype fp32
```
