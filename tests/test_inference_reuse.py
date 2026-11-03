"""Caches must preserve masks, sample ownership, and read-only conditioning."""

import os
from unittest.mock import patch

import pytest
import torch

from opendde.model.batching import pad_token_tensors, pairformer_batch
from opendde.model.modules.confidence import ConfidenceHead
from opendde.model.modules.diffusion import DiffusionConditioning
from opendde.model.modules.pairformer import PairformerStack
from opendde.model.modules.transformer import DiffusionTransformer


def activate_residuals(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1 and not parameter.count_nonzero():
                parameter.normal_(std=0.02)


def test_recycled_padding_clears_previous_values():
    buf = pad_token_tensors([torch.ones(3, 3, 2), torch.ones(5, 5, 2)], pair=True)
    actual = pad_token_tensors(
        [torch.full((5, 5, 2), 2.0), torch.full((2, 2, 2), 3.0)], pair=True, out=buf
    )
    assert actual is buf
    assert (actual[0] == 2).all()
    assert (actual[1, :2, :2] == 3).all()
    assert not actual[1, 2:].any() and not actual[1, :, 2:].any()


@pytest.mark.parametrize("fusion", [False, True])
def test_diffusion_projected_bias_is_reused_and_read_only(fusion):
    torch.manual_seed(11)
    model = DiffusionTransformer(c_a=16, c_s=16, c_z=8, n_blocks=2, n_heads=2).eval()
    activate_residuals(model)
    a, s = torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 16)
    z = torch.randn(2, 8, 7, 7) if fusion else torch.randn(2, 7, 7, 8)
    mask = torch.randn(2, 7, 7)
    mask[0, :, 4:] = -1e9
    cache = {}
    with torch.inference_mode():
        want = model(a, s, z, extra_attn_bias=mask, enable_efficient_fusion=fusion)
        first = model(
            a,
            s,
            z,
            extra_attn_bias=mask,
            enable_efficient_fusion=fusion,
            pair_bias_cache=cache,
        )
        assert len(cache) == 2
        saved = {k: v.clone() for k, v in cache.items()}
        again = model(
            a,
            s,
            z,
            extra_attn_bias=mask,
            enable_efficient_fusion=fusion,
            pair_bias_cache=cache,
        )
    assert torch.equal(want, first) and torch.equal(first, again)
    for key in cache:
        assert torch.equal(cache[key], saved[key])


def test_single_conditioning_cache_keeps_noise_dependence():
    torch.manual_seed(12)
    model = DiffusionConditioning(
        c_z=8, c_s=16, c_s_inputs=12, c_noise_embedding=8
    ).eval()
    inputs, trunk, pair = torch.randn(5, 12), torch.randn(5, 16), torch.randn(5, 5, 8)
    with torch.inference_mode():
        static = model.prepare_single_cache(inputs, trunk)
        saved = static.clone()
        outputs = []
        for level in (2.0, 20.0):
            kw = dict(
                t_hat_noise_level=torch.full((3,), level),
                relp_feature=None,
                s_inputs=inputs,
                s_trunk=trunk,
                z_trunk=pair,
                pair_z=pair,
                inplace_safe=False,
            )
            want = model(**kw)
            got = model(**kw, single_s=static)
            assert torch.equal(want[0], got[0]) and got[1] is pair
            outputs.append(got[0])
        assert torch.equal(static, saved)
        assert not torch.equal(*outputs)


@pytest.mark.parametrize("chunk", [2, 3, 5, None])
@pytest.mark.parametrize("inplace", [False, True])
def test_confidence_chunks_preserve_scores_and_disabled_heads(chunk, inplace):
    torch.manual_seed(13)
    head = ConfidenceHead(n_blocks=1, c_s=16, c_z=16, c_s_inputs=32).eval()
    activate_residuals(head)
    n = 5
    features = dict(
        atom_to_token_idx=torch.arange(n),
        atom_to_tokatom_idx=torch.zeros(n, dtype=torch.long),
        distogram_rep_atom_mask=torch.ones(n, dtype=torch.bool),
        structural_pair_attn_bias=torch.randn(n, n),
    )
    kw = dict(
        input_feature_dict=features,
        s_inputs=torch.randn(n, 32),
        s_trunk=torch.randn(n, 16),
        z_trunk=torch.randn(n, n, 16),
        pair_mask=torch.ones(n, n),
        x_pred_coords=torch.randn(5, n, 3),
        inplace_safe=inplace,
        chunk_size=2,
        compute_resolved=False,
    )
    with torch.inference_mode():
        want = head(**kw, sample_chunk_size=1)
        got = head(**kw, sample_chunk_size=chunk)
    assert got[-1] is None and want[-1] is None
    for x, y in zip(got[:-1], want[:-1]):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)
        assert x.shape[0] == 5
    assert got[1].device.type == got[2].device.type == "cpu"


@pytest.mark.skipif(
    os.environ.get("OPENDDE_RUN_BATCH_KERNEL_TESTS") != "1",
    reason="opt-in CUDA regression",
)
@pytest.mark.parametrize("lengths", [(7, 13), (7, 23), (23, 37), (111, 129)])
def test_fused_ragged_padding_matches_unpadded_torch(lengths):
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        PairformerStack(n_blocks=2, c_s=64, c_z=128, n_heads=4, hidden_scale_up=False)
        .cuda()
        .eval()
    )
    activate_residuals(model)
    singles = [torch.randn(n, 64, device="cuda") for n in lengths]
    pairs = [torch.randn(n, n, 128, device="cuda") for n in lengths]
    biases = [torch.randn(n, n, device="cuda") for n in lengths]
    with torch.inference_mode():
        want = [
            model(
                s.clone(),
                z.clone(),
                pair_mask=None,
                extra_attn_bias=bias,
                inplace_safe=True,
                chunk_size=8,
                triangle_multiplicative="torch",
                triangle_attention="torch",
            )
            for s, z, bias in zip(singles, pairs, biases)
        ]
        for single, pair, bias, expected in zip(singles, pairs, biases, want):
            unbatched = model(
                single.clone(),
                pair.clone(),
                pair_mask=None,
                extra_attn_bias=bias,
                inplace_safe=True,
                chunk_size=8,
                triangle_multiplicative="cuequivariance",
                triangle_attention="cuequivariance",
            )
            for value, reference in zip(unbatched, expected):
                torch.testing.assert_close(value, reference, atol=2e-4, rtol=2e-4)
        actual = pairformer_batch(
            model,
            singles,
            pairs,
            biases=biases,
            chunk_size=8,
            triangle_multiplicative="cuequivariance",
            triangle_attention="cuequivariance",
        )
    for s, z, (expected_s, expected_z) in zip(*actual, want):
        torch.testing.assert_close(s, expected_s, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(z, expected_z, atol=2e-4, rtol=2e-4)
