#!/usr/bin/env python3
"""H200 compatibility probe for RFD4-style cuEquivariance 0.13 pair attention.

Run in an isolated Torch >=2.11 / cuEquivariance 0.13 environment. This is a
synthetic operator diagnostic, not a prediction backend or speed benchmark.
It deliberately checks the vendor's internal route counters: a successful
public API call alone does not demonstrate execution of a fused kernel.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opendde.model.modules.transformer import AttentionPairBias
from opendde.utils.seed import seed_everything


def differences(reference, actual):
    assert reference.shape == actual.shape, (reference.shape, actual.shape)
    delta = reference.float() - actual.float()
    return {
        "bitwise_equal": torch.equal(reference, actual),
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "finite": bool(actual.isfinite().all()),
    }


@contextmanager
def record_routes(module):
    names = (
        "_attention_pair_bias_torch",
        "_run_flash_forward",
        "pair_repr_to_attention_bias",
    )
    originals = {name: getattr(module, name) for name in names}
    counts = {}

    def wrap(name, function):
        def call(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return function(*args, **kwargs)

        return call

    try:
        for name, function in originals.items():
            setattr(module, name, wrap(name, function))
        yield counts
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


def vendor_arguments(module, single, pair):
    attention = module.attention
    return dict(
        single_repr=single,
        pair_repr=pair,
        mask=None,
        num_heads=module.n_heads,
        w_ln_a=module.layernorm_a.weight,
        b_ln_a=module.layernorm_a.bias,
        w_proj_q=attention.linear_q.weight,
        b_proj_q=attention.linear_q.bias,
        w_proj_k=attention.linear_k.weight,
        w_proj_v=attention.linear_v.weight,
        w_proj_g=attention.linear_g.weight,
        b_proj_g=attention.linear_g.bias,
        w_proj_o=attention.linear_o.weight,
        b_proj_o=None,
        w_proj_z=module.linear_nobias_z.weight,
        w_ln_z=module.layernorm_z.weight,
        b_ln_z=module.layernorm_z.bias,
        eps=module.layernorm_a.eps,
        inf=1e9,
    )


def check_case(
    vendor, implementation, name, batch, length, width, pair_width, heads, amp
):
    seed_everything(71, deterministic=False)
    module = (
        AttentionPairBias(
            has_s=False,
            create_offset_ln_z=True,
            n_heads=heads,
            c_a=width,
            c_z=pair_width,
        )
        .cuda()
        .eval()
    )
    # The default output projection is zero; avoid a vacuous parity result.
    with torch.no_grad():
        module.attention.linear_o.weight.normal_(std=0.02)
    single = torch.randn(batch, length, width, device="cuda")
    pair = torch.randn(batch, length, length, pair_width, device="cuda")
    params = vendor_arguments(module, single, pair)
    row = {
        "name": name,
        "shape": [batch, length, width, pair_width, heads],
        "bf16_autocast": amp,
    }
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
        warnings.catch_warnings(record=True) as observed,
        record_routes(implementation) as counts,
    ):
        warnings.simplefilter("always")
        reference = module(single, None, pair)
        actual = vendor(**params)[0]
        repeat = vendor(**params)[0]
        row["comparison"] = differences(reference, actual)
        row["repeat"] = differences(actual, repeat)
        row["routes"] = dict(counts)

        # Isolate shape support from the mask constant. There is no mask in
        # this control; changing OpenDDE's mask convention is NOT proposed.
        params["inf"] = 1e6
        counts.clear()
        alternate = vendor(**params)[0]
        row["vendor_default_inf_routes"] = dict(counts)
        row["vendor_default_inf_comparison"] = differences(reference, alternate)

        cached = dict(params)
        cached["pair_repr"] = (
            module.linear_nobias_z(module.layernorm_z(pair))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        cached["is_cached_z_proj"] = True
        counts.clear()
        cached_result = vendor(**cached)[0]
        row["cached_bias_routes"] = dict(counts)
        row["cached_bias_comparison"] = differences(reference, cached_result)

        params["inf"] = 1e9
        mask = torch.ones(batch, length, device="cuda")
        mask[:, -7:] = 0
        if batch > 1:
            mask[1] = 0
        params["mask"] = mask
        bias = (1 - mask[:, None, :]).expand(batch, length, length) * -1e9
        reference = module(single, None, pair, extra_attn_bias=bias)
        actual = vendor(**params)[0]
        row["masked_comparison"] = differences(reference, actual)
        row["warnings"] = [str(w.message) for w in observed]

    assert row["repeat"]["bitwise_equal"], row
    for key in (
        "comparison",
        "vendor_default_inf_comparison",
        "cached_bias_comparison",
        "masked_comparison",
    ):
        assert row[key]["finite"], row
        # Operator-level checks only; these tolerances establish no prediction
        # or confidence-score equivalence for a multi-layer diffusion model.
        assert row[key]["max_abs"] < (0.01 if amp else 3e-6), row
    if name == "supported_shape_control":
        assert row["vendor_default_inf_routes"].get("_run_flash_forward", 0) > 0, row
    else:
        assert row["vendor_default_inf_routes"] == {"_attention_pair_bias_torch": 1}, (
            row
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cuequivariance_torch as cueq

    if (
        not cueq.__version__.startswith("0.13.")
        or "single_repr" not in inspect.signature(cueq.attention_pair_bias).parameters
    ):
        parser.error(
            "Requires isolated cuEquivariance 0.13 and Torch >=2.11; do not upgrade the production environment for this probe."
        )
    implementation = importlib.import_module(
        "cuequivariance_ops_torch.attention_pair_bias"
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("This diagnostic's route assertions target an H200/SM90 GPU.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "torch": torch.__version__,
        "cuequivariance": cueq.__version__,
        "gpu": torch.cuda.get_device_name(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        ("trunk_fp32", 1, 245, 384, 384, 16, False),
        ("trunk_bf16", 1, 245, 384, 384, 16, True),
        ("trunk_pair_bf16", 2, 320, 384, 384, 16, True),
        ("trunk_large_bf16", 1, 1218, 384, 384, 16, True),
        # Ordinary LayerNorm here intentionally isolates shape/dtype support.
        # It is NOT an adapter for OpenDDE's adaptive-normalized diffusion.
        ("diffusion_shape_fp32_control", 1, 475, 768, 128, 16, False),
        ("supported_shape_control", 1, 512, 256, 128, 8, True),
    ]
    for case in cases:
        row = check_case(cueq.attention_pair_bias, implementation, *case)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)

    seed_everything(19, deterministic=False)
    module = (
        AttentionPairBias(has_s=True, n_heads=16, c_a=768, c_s=384, c_z=128)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        module.layernorm_a.linear_s.weight.normal_(std=0.02)
        module.layernorm_a.linear_nobias_s.weight.normal_(std=0.02)
        single = torch.randn(1, 475, 768, device="cuda")
        condition = torch.randn(1, 475, 384, device="cuda")
        adaptive = module.layernorm_a(single, condition)
        normalized_again = torch.nn.functional.layer_norm(adaptive, (768,))
        report["extra_normalization"] = differences(adaptive, normalized_again)
        assert not report["extra_normalization"]["bitwise_equal"]
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
