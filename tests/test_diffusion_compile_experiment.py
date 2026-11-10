"""The compiler experiment must preserve module ownership and eager fallbacks."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from opendde.model.modules.transformer import (
    AtomTransformer,
    ConditionedTransitionBlock,
)
from opendde.model.inference_cache import InferenceCacheBudget
from scripts.diffusion_compile_experiment import compiled_diffusion_transitions
from scripts.diffusion_compile_experiment import (
    cached_atom_aggregation,
    compiled_diffusion_blocks,
    token_block_with_bias,
)
from opendde.model.inference_cache import diffusion_attention_scope
from opendde.model.modules.transformer import DiffusionTransformer


def test_cache_only_experiment_restores_request_and_sampler_after_error(monkeypatch):
    from opendde.model import batching
    from opendde.model.inference_cache import get_diffusion_attention_execution

    def fail(*args, **kwargs):
        execution = get_diffusion_attention_execution()
        execution.aggregation_plans["temporary"] = True
        raise RuntimeError("sample failed")

    monkeypatch.setattr(batching, "sample_diffusion_batch", fail)
    with torch.no_grad(), diffusion_attention_scope("reference", True) as state:
        with pytest.raises(RuntimeError, match="sample failed"):
            with cached_atom_aggregation() as stats:
                batching.sample_diffusion_batch()
        assert state.aggregation_plans is None
        assert batching.sample_diffusion_batch is fail
        assert stats == {"requests": 1, "plans": 1}


@pytest.mark.parametrize("samples", [1, 3])
def test_explicit_token_block_matches_original_mask_and_shared_conditioning(samples):
    torch.manual_seed(91)
    transformer = DiffusionTransformer(16, 16, 8, n_blocks=1, n_heads=2).eval()
    with torch.no_grad():
        for p in transformer.parameters():
            if p.ndim > 1 and not p.count_nonzero():
                p.normal_(std=0.02)
    block = transformer.blocks[0]
    a, s, z = (
        torch.randn(2, samples, 7, 16),
        torch.randn(2, 1, 7, 16),
        torch.randn(2, 7, 7, 8),
    )
    mask = torch.zeros(2, 7, 7)
    mask[0, :, 4:] = -1e9
    cache = {}
    with torch.no_grad(), diffusion_attention_scope("reference", True):
        expected = block(
            a, s, z, inplace_safe=True, extra_attn_bias=mask, pair_bias_cache=cache
        )[0]
        bias = cache[block.attention_pair_bias]
        for backend in ("reference", "efficient"):
            actual = token_block_with_bias(block, a, s, bias, backend)
            assert torch.equal(actual, expected)
        with patch("torch.compile", side_effect=AssertionError("CPU fallback")):
            with compiled_diffusion_blocks(transformer) as stats:
                actual = block(a, s, z, inplace_safe=True, pair_bias_cache=cache)[0]
                assert torch.equal(actual, expected)
                assert stats["eager_calls"] == 1
        assert "forward" not in block.__dict__


def make_module(device="cpu"):
    torch.manual_seed(741)
    module = ConditionedTransitionBlock(c_a=16, c_s=16).to(device).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim > 1 and not parameter.count_nonzero():
                parameter.normal_(std=0.02)
    return module


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_and_gradients_remain_eager_and_state_dict_unchanged(enabled):
    module = make_module()
    a, s = torch.randn(1, 7, 16), torch.randn(1, 7, 16)
    expected = module(a, s)
    keys = list(module.state_dict())
    with patch("torch.compile", side_effect=AssertionError("must stay eager")):
        with compiled_diffusion_transitions(module, enabled) as stats:
            with torch.no_grad():
                torch.testing.assert_close(module(a, s), expected, atol=0, rtol=0)
            module.train()
            module(a.requires_grad_(), s).sum().backward()
            assert a.grad is not None and a.grad.isfinite().all()
            assert list(module.state_dict()) == keys
            assert stats["compiled_calls"] == stats["modules"] == 0
    assert "forward" not in module.__dict__


def test_compiler_is_lazy_reused_and_errors_restore_existing_override():
    module = make_module()
    original = Mock(return_value=torch.ones(1))
    module.forward = original
    compiled = Mock(return_value=torch.ones(1))
    with patch("torch.compile", return_value=compiled) as compiler:
        with torch.no_grad(), compiled_diffusion_transitions(module, True) as stats:
            compiler.assert_not_called()
            for _ in range(2):
                module(SimpleNamespace(is_cuda=True), None)
            assert stats == {"enabled": True, "modules": 1, "compiled_calls": 2}
            compiler.assert_called_once_with(
                original,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            original.assert_not_called()
    assert module.forward is original
    with patch("torch.compile", side_effect=RuntimeError("compiler failed")):
        with pytest.raises(RuntimeError, match="compiler failed"):
            with torch.no_grad(), compiled_diffusion_transitions(module, True):
                module(SimpleNamespace(is_cuda=True), None)
    assert module.forward is original


CUDA = os.environ.get("OPENDDE_RUN_COMPILE_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_COMPILE_TESTS=1 on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_whole_blocks_preserve_masks_backends_and_fresh_biases(dtype):
    transformer = DiffusionTransformer(16, 16, 8, n_blocks=1, n_heads=2).cuda().eval()
    with torch.no_grad():
        for p in transformer.parameters():
            if p.ndim > 1 and not p.count_nonzero():
                p.normal_(std=0.02)
    block = transformer.blocks[0]
    original = block.forward
    keys = list(transformer.state_dict())
    a, s, z = (
        torch.randn(*shape, device="cuda")
        for shape in ((2, 1, 7, 16), (2, 1, 7, 16), (2, 7, 7, 8))
    )
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16),
        compiled_diffusion_blocks(transformer) as stats,
    ):
        for backend in ("reference", "efficient"):
            with diffusion_attention_scope(backend, True):
                cache = {}
                mask = torch.zeros(2, 7, 7, device="cuda")
                mask[0, :, 4:] = -1e9
                kwargs = dict(
                    inplace_safe=True, extra_attn_bias=mask, pair_bias_cache=cache
                )
                expected = original(a, s, z, **kwargs)[0]
                result = block(a, s, z, **kwargs)[0]
                retained = result.clone()
                again = block(a, s, z, **kwargs)[0]
                torch.testing.assert_close(
                    result,
                    expected,
                    atol=3e-3 if dtype == torch.bfloat16 else 3e-6,
                    rtol=1e-3,
                )
                assert torch.equal(result, again)
                # Same shapes must read this request's bias values, not prior data.
                cache[block.attention_pair_bias] = torch.zeros_like(
                    cache[block.attention_pair_bias]
                )
                fresh = block(a + 1, s, z, **kwargs)[0]
                torch.testing.assert_close(
                    fresh,
                    original(a + 1, s, z, **kwargs)[0],
                    atol=3e-3 if dtype == torch.bfloat16 else 3e-6,
                    rtol=1e-3,
                )
                assert torch.equal(result, retained)
        if dtype == torch.bfloat16:
            assert stats["modules"] == stats["compiled_calls"] == 0
            assert stats["eager_calls"] == 6
        else:
            assert stats["modules"] == 1 and stats["compiled_calls"] == 6
    assert "forward" not in block.__dict__
    assert list(transformer.state_dict()) == keys


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_COMPILE_TESTS=1 on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
def test_cuda_compilation_values_repeatability_and_condition_ownership(dtype, cached):
    module = make_module("cuda")
    a, s = (torch.randn(1, 7, 16, device="cuda") for _ in range(2))
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16),
    ):
        cache = None
        if cached:
            atom = (
                AtomTransformer(
                    c_atom=16,
                    c_atompair=8,
                    n_blocks=1,
                    n_heads=2,
                    n_queries=4,
                    n_keys=8,
                )
                .cuda()
                .eval()
            )
            pair = torch.randn(1, 2, 4, 8, 8, device="cuda")
            cache = atom.prepare_static_cache(
                s, pair, InferenceCacheBudget(1024**2)
            ).blocks[0]
        a_before, s_before = a.clone(), s.clone()
        expected = module(a, s, atom_cache=cache)
        with compiled_diffusion_transitions(module, True) as stats:
            actual = module(a, s, atom_cache=cache)
            retained = actual.clone()
            again = module(a, s, atom_cache=cache)
            module(a + 1, s, atom_cache=cache)
            assert stats["compiled_calls"] == 3
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.isfinite().all()
        torch.testing.assert_close(
            actual, expected, atol=2e-3 if dtype == torch.bfloat16 else 2e-6, rtol=1e-3
        )
        assert torch.equal(actual, again) and torch.equal(actual, retained)
        assert torch.equal(a, a_before) and torch.equal(s, s_before)
