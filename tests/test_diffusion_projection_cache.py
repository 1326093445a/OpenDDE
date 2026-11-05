"""Request-local projection reuse must preserve topology, noise and fallbacks."""

import math
import os
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from opendde.model.inference_cache import InferenceCacheBudget
from opendde.model.modules.transformer import AtomTransformer, DiffusionTransformer
from opendde.config.schema import InferSettingConfig


def activate(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1 and not parameter.count_nonzero():
                parameter.normal_(std=0.02)


def make_atom(device="cpu"):
    torch.manual_seed(106)
    module = (
        AtomTransformer(
            c_atom=16, c_atompair=8, n_blocks=2, n_heads=2, n_queries=4, n_keys=8
        )
        .to(device)
        .eval()
    )
    activate(module)
    return module


def inputs(n, device="cpu"):
    return (
        torch.randn(1, n, 16, device=device),
        torch.randn(1, math.ceil(n / 4), 4, 8, 8, device=device),
    )


@pytest.mark.parametrize("n", [7, 11, 17])
@pytest.mark.parametrize("budget_blocks", [0, 1, 2])
def test_atom_cache_reuses_static_projections_across_sample_chunks(n, budget_blocks):
    module = make_atom()
    c, p = inputs(n)
    c_before, p_before = c.clone(), p.clone()
    per_block = (8 * n * 16 + math.ceil(n / 4) * 4 * 8 * 2) * 4
    budget = InferenceCacheBudget(per_block * budget_blocks)
    projection = module.diffusion_transformer.blocks[
        0
    ].attention_pair_bias.linear_nobias_z
    keys_before = list(module.state_dict())
    with (
        torch.inference_mode(),
        patch.object(projection, "forward", wraps=projection.forward) as projected,
    ):
        cache = module.prepare_static_cache(c, p, budget)
        assert budget.used_bytes == per_block * budget_blocks
        assert (cache is None) == (budget_blocks == 0)
        if cache is not None:
            assert sum(block is not None for block in cache.blocks) == budget_blocks
            saved = cache.blocks[0].pair_bias.clone()
        for samples in (2, 3, 5, 1):
            q = torch.randn(samples, n, 16)
            want = module(q.clone(), c, p, chunk_size=3, inplace_safe=True)
            before = projected.call_count
            got = module(
                q.clone(), c, p, chunk_size=3, inplace_safe=True, static_cache=cache
            )
            assert projected.call_count - before == int(budget_blocks == 0)
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        if cache is not None:
            assert torch.equal(cache.blocks[0].pair_bias, saved)
    assert torch.equal(c, c_before) and torch.equal(p, p_before)
    assert list(module.state_dict()) == keys_before


def test_cache_rejects_other_jobs_and_training():
    module = make_atom()
    c, p = inputs(11)
    q = torch.randn(5, 11, 16)
    with torch.inference_mode():
        cache = module.prepare_static_cache(c, p, InferenceCacheBudget(1024**2))
        with pytest.raises(ValueError, match="different request"):
            module(q, c.clone(), p, static_cache=cache)
        module.train()
        with pytest.raises(ValueError, match="without gradients"):
            module(q, c, p, static_cache=cache)
        assert module.prepare_static_cache(c, p, InferenceCacheBudget(1024**2)) is None
    module.eval()
    assert module.prepare_static_cache(c, p, InferenceCacheBudget(1024**2)) is None
    with pytest.raises(ValueError, match="without gradients"):
        module(q, c, p, static_cache=cache)
    # No-cache training/gradient behavior remains available.
    module(q.requires_grad_(), c, p).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_memory_budget_limits_and_disabled_cuda_queries():
    with (
        patch("torch.cuda.mem_get_info", return_value=(80 * 1024**2, 100 * 1024**2)),
        patch("torch.cuda.memory_reserved", return_value=16 * 1024**2),
        patch("torch.cuda.memory_allocated", return_value=8 * 1024**2),
    ):
        budget = InferenceCacheBudget.for_device(2048, torch.device("cuda:3"))
        assert budget.max_bytes == 11 * 1024**2
        assert budget.reserve(budget.max_bytes)
        assert not budget.reserve(1)
        assert budget.used_bytes == budget.max_bytes
    with patch("torch.cuda.mem_get_info", side_effect=AssertionError("disabled")):
        assert InferenceCacheBudget.for_device(0, torch.device("cuda")).max_bytes == 0
    assert (
        InferenceCacheBudget.for_device(2048, torch.device("cpu")).max_bytes
        == 256 * 1024**2
    )
    with pytest.raises(ValueError):
        InferenceCacheBudget.for_device(-1, torch.device("cpu"))
    with pytest.raises(ValueError):
        InferSettingConfig(
            chunk_size=1,
            dynamic_chunk_size=False,
            chunk_size_thresholds={},
            sample_diffusion_chunk_size=5,
            diffusion_cache_max_mb=-1,
        )


@pytest.mark.parametrize("samples", [1, 2, 3, 5])
def test_shared_conditioning_preserves_three_ragged_job_members(samples):
    torch.manual_seed(107)
    model = DiffusionTransformer(c_a=16, c_s=16, c_z=8, n_blocks=2, n_heads=2).eval()
    activate(model)
    a = torch.randn(3, samples, 11, 16)
    single = torch.randn(3, 1, 11, 16)
    pair = torch.randn(3, 11, 11, 8)
    bias = torch.randn(3, 11, 11)
    for i, n in enumerate((5, 7, 11)):
        bias[i, :, n:] = -1e9
    with torch.inference_mode():
        for step in range(3):
            # Same noise condition within a job, different condition across jobs/steps.
            s = single + step * 0.1
            want = model(
                a,
                s.expand(-1, samples, -1, -1).contiguous(),
                pair,
                extra_attn_bias=bias,
            )
            got = model(a, s, pair, extra_attn_bias=bias)
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)
            assert not torch.equal(got[0], got[1])


@pytest.mark.skipif(
    os.environ.get("OPENDDE_RUN_BATCH_KERNEL_TESTS") != "1",
    reason="opt-in CUDA projection-cache regression",
)
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_atom_cache_matches_streamed_projection(amp):
    torch.backends.cuda.matmul.allow_tf32 = False
    module = make_atom("cuda")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext(),
    ):
        for n in (31, 65):
            c, p = inputs(n, "cuda")
            cache = module.prepare_static_cache(c, p, InferenceCacheBudget(1024**2))
            for samples in (2, 5):
                q = torch.randn(samples, n, 16, device="cuda")
                want = module(q, c, p, chunk_size=16)
                got = module(q, c, p, chunk_size=16, static_cache=cache)
                torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_cli_projection_controls_reach_inference_without_changing_sample_counts(
    monkeypatch, tmp_path
):
    import json
    from click.testing import CliRunner
    from runner import batch_inference

    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps(
            [{"name": "test", "sequences": [{"proteinChain": {"sequence": "ACDE"}}]}]
        )
    )
    captured = []
    monkeypatch.setattr(
        batch_inference, "inference_jsons", lambda *args, **kw: captured.append(kw)
    )
    result = CliRunner().invoke(
        batch_inference.predict,
        [
            "-i",
            str(path),
            "-o",
            str(tmp_path / "out"),
            "--use_msa",
            "false",
            "--batch_size",
            "3",
            "--sample",
            "5",
            "--cycle",
            "10",
            "--step",
            "200",
            "--diffusion_attention_backend",
            "efficient",
            "--trimul_kernel",
            "flashpairformer",
            "--reuse_diffusion_window_masks",
            "false",
            "--batch_max_padding_ratio",
            "1.4",
            "--batch_max_pair_tokens",
            "500000",
            "--diffusion_cache_max_mb",
            "0",
            "--share_diffusion_sample_conditioning",
            "false",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["diffusion_attention_backend"] == "efficient"
    assert captured[0]["trimul_kernel"] == "flashpairformer"
    assert captured[0]["reuse_diffusion_window_masks"] is False
    assert captured[0]["batch_max_padding_ratio"] == 1.4
    assert captured[0]["batch_max_pair_tokens"] == 500000
    assert captured[0]["diffusion_cache_max_mb"] == 0
    assert captured[0]["share_diffusion_sample_conditioning"] is False
    assert (
        captured[0]["batch_size"],
        captured[0]["n_sample"],
        captured[0]["n_cycle"],
        captured[0]["n_step"],
    ) == (3, 5, 10, 200)
    with pytest.raises(ValueError, match="nonnegative"):
        batch_inference.get_default_runner(diffusion_cache_max_mb=-1)


def test_invalid_attention_backend_rejected_before_model_load():
    from runner.batch_inference import get_default_runner

    with pytest.raises(ValueError, match="reference or efficient"):
        get_default_runner(diffusion_attention_backend="invalid")
