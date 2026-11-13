"""Compiler wrappers must preserve eager arithmetic and module ownership."""

import os
from unittest.mock import patch

import pytest
import torch

from opendde.model.modules.pairformer import PairformerStack
from opendde.model.modules.primitives import Attention
from opendde.model.triangular.layers import OpenFoldLayerNorm, OpenfoldLinear
from opendde.model.inference_cache import diffusion_attention_scope
from opendde.utils.seed import seed_everything
from scripts.pairformer_compile_experiment import (
    compiled_pairformer_blocks,
    cuda_layer_norm,
    cuda_linear,
    trunk_attention,
)


def make_stack(device="cpu", width=32):
    seed_everything(91, False)
    stack = (
        PairformerStack(
            n_blocks=1, n_heads=4, c_z=width, c_s=width, hidden_scale_up=True
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for parameter in stack.parameters():
            if parameter.ndim > 1 and not parameter.count_nonzero():
                parameter.normal_(std=0.02)
    return stack


def test_cpu_fallback_state_dict_and_restoration_after_exception():
    stack = make_stack()
    original = stack.blocks[0].forward_source
    stack.blocks[0].forward_source = original  # Preserve a pre-existing override.
    state = {key: value.clone() for key, value in stack.state_dict().items()}
    s, z, mask = torch.randn(1, 5, 32), torch.randn(1, 5, 5, 32), torch.ones(1, 5, 5)
    with torch.no_grad():
        expected = stack(s.clone(), z.clone(), mask)
        with patch("torch.compile", side_effect=AssertionError("must remain eager")):
            with pytest.raises(RuntimeError, match="injected"):
                with compiled_pairformer_blocks(stack) as stats:
                    actual = stack(s.clone(), z.clone(), mask)
                    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
                    assert stats["compiled_calls"] == 0 and stats["eager_calls"] == 1
                    raise RuntimeError("injected")
    assert stack.blocks[0].forward_source is original
    assert all(torch.equal(state[k], v) for k, v in stack.state_dict().items())
    assert all("forward" not in m.__dict__ for m in stack.modules())


def test_foldcp_rejected_without_modifying_model(monkeypatch):
    stack = make_stack()
    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    with pytest.raises(ValueError, match="Fold-CP"):
        with compiled_pairformer_blocks(stack):
            pytest.fail("must reject")
    assert "forward_source" not in stack.blocks[0].__dict__


CUDA = os.getenv("OPENDDE_RUN_COMPILE_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_COMPILE_TESTS=1 on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_traceable_wrappers_match_original_bitwise_before_compilation(dtype):
    seed_everything(29, False)
    x = torch.randn(2, 9, 32, device="cuda", dtype=dtype)
    norm = OpenFoldLayerNorm(32).cuda().eval()
    linear = OpenfoldLinear(32, 32).cuda().eval()
    attention = Attention(32, 32, 32, 8, 4, zero_init=False).cuda().eval()
    bias = torch.randn(2, 4, 9, 9, device="cuda")
    bias[0, :, :, -2:] = -1e9
    bias[1] = -1e9
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16),
    ):
        for offset in (True, False):
            n = (
                norm
                if offset
                else OpenFoldLayerNorm(32, create_scale=False, create_offset=False)
                .cuda()
                .eval()
            )
            assert torch.equal(n(x), cuda_layer_norm(n, x))
        for precision in (None, torch.float32):
            linear.precision = precision
            assert torch.equal(linear(x), cuda_linear(linear, x))
        assert torch.equal(
            attention(x, x, attn_bias=bias),
            trunk_attention(attention, x, x, attn_bias=bias),
        )


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_COMPILE_TESTS=1 on CUDA")
@pytest.mark.parametrize("amp", [False, True])
def test_whole_block_compile_masks_repeats_shapes_and_fallbacks(amp):
    stack = make_stack("cuda")
    block = stack.blocks[0]
    kwargs = dict(
        triangle_multiplicative="cuequivariance",
        triangle_attention="cuequivariance",
        inplace_safe=True,
    )
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
        compiled_pairformer_blocks(stack) as stats,
    ):
        # Both warm shapes and multiple leading jobs; masks include a padded job.
        for n in (24, 32):
            s, z = (
                torch.randn(2, n, 32, device="cuda"),
                torch.randn(2, n, n, 32, device="cuda"),
            )
            mask = torch.ones(2, n, n, device="cuda")
            mask[0, -3:, :] = 0
            mask[0, :, -3:] = 0
            bias = torch.zeros_like(mask)
            bias[0, :, -3:] = -1e9
            args = dict(pair_mask=mask, extra_attn_bias=bias, **kwargs)
            # Call the original class method to bypass only the compilation wrapper.
            expected = type(block).forward_source(block, s.clone(), z.clone(), **args)
            actual = block.forward_source(s.clone(), z.clone(), **args)
            repeat = block.forward_source(s.clone(), z.clone(), **args)
            for want, got, again in zip(expected, actual, repeat):
                assert got.shape == want.shape and got.isfinite().all()
                assert torch.equal(got, again)
                torch.testing.assert_close(
                    got, want, atol=0.06 if amp else 1e-5, rtol=0.02 if amp else 1e-5
                )
            noisy_s, noisy_z = s.clone(), z.clone()
            noisy_s[0, -3:] = 10 * torch.randn_like(noisy_s[0, -3:])
            noisy_z[0, -3:] = 10 * torch.randn_like(noisy_z[0, -3:])
            noisy_z[0, :, -3:] = 10 * torch.randn_like(noisy_z[0, :, -3:])
            changed_padding = block.forward_source(noisy_s, noisy_z, **args)
            assert torch.equal(actual[0][0, :-3], changed_padding[0][0, :-3])
            assert torch.equal(actual[1][0, :-3, :-3], changed_padding[1][0, :-3, :-3])
            assert all(torch.equal(a[1], b[1]) for a, b in zip(actual, changed_padding))
        assert stats["modules"] == 1 and stats["compiled_calls"] == 6
        with diffusion_attention_scope("reference", True):
            block.forward_source(s.clone(), z.clone(), **args)
        block.forward_source(
            s[:, :16].clone(), z[:, :16, :16].clone(), pair_mask=None, **kwargs
        )
        stack.train()
        block.forward_source(s.clone(), z.clone(), **args)
        assert stats["eager_calls"] == 3
    assert "forward_source" not in block.__dict__
    assert all("forward" not in module.__dict__ for module in stack.modules())


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_COMPILE_TESTS=1 on CUDA")
def test_compile_failure_propagates_and_gradient_and_kernel_fallbacks():
    stack = make_stack("cuda")
    block = stack.blocks[0]
    s = torch.randn(1, 24, 32, device="cuda")
    z = torch.randn(1, 24, 24, 32, device="cuda")
    kwargs = dict(
        pair_mask=None,
        triangle_multiplicative="cuequivariance",
        triangle_attention="cuequivariance",
    )
    with patch("torch.compile", side_effect=RuntimeError("injected compiler error")):
        with pytest.raises(RuntimeError, match="injected compiler error"):
            with compiled_pairformer_blocks(stack) as stats:
                with torch.enable_grad():
                    result = block.forward_source(s.clone(), z.clone(), **kwargs)
                    assert result[0].requires_grad
                with torch.no_grad():
                    block.forward_source(s.clone(), z.clone(), pair_mask=None)
                    assert stats["eager_calls"] == 2
                    block.forward_source(s.clone(), z.clone(), **kwargs)
    assert "forward_source" not in block.__dict__
    assert all("forward" not in module.__dict__ for module in stack.modules())
