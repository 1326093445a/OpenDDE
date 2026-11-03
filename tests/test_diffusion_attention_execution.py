"""Attention layout choices are opt-in; reused masks preserve reference values."""

from concurrent.futures import ThreadPoolExecutor
import os

import pytest
import torch
from torch.nn import functional as F

from opendde.config.schema import InferSettingConfig
from opendde.model.inference_cache import (
    diffusion_attention_scope,
    get_diffusion_attention_execution,
)
from opendde.model.modules.primitives import (
    _attention,
    _attention_supported_layout,
    rearrange_to_dense_trunk,
)


def test_scope_cleanup_nesting_and_thread_isolation():
    assert get_diffusion_attention_execution() is None
    with torch.no_grad(), diffusion_attention_scope("reference", True) as outer:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(get_diffusion_attention_execution).result() is None
        with pytest.raises(RuntimeError, match="failed"):
            with diffusion_attention_scope("efficient", True) as inner:
                inner.masks[()] = torch.zeros(1)
                inner.mask_bytes = 4
                raise RuntimeError("failed")
        assert not inner.masks and inner.mask_bytes == 0
        assert get_diffusion_attention_execution() is outer
    assert get_diffusion_attention_execution() is None
    with pytest.raises(ValueError, match="without gradients"):
        with diffusion_attention_scope("reference", True):
            pass
    with torch.no_grad(), pytest.raises(ValueError, match="reference or efficient"):
        with diffusion_attention_scope("invalid", True):
            pass


@pytest.mark.parametrize("n", [1, 4, 7, 8, 17, 32])
@pytest.mark.parametrize("windows", [(4, 4), (4, 8), (8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefix", [(), (2, 3)])
def test_direct_window_masks_are_exact_and_reused(n, windows, dtype, prefix):
    q = torch.randn(*prefix, n, 8, dtype=dtype)
    want = rearrange_to_dense_trunk(q, q, q, *windows)
    with torch.no_grad(), diffusion_attention_scope("reference", True) as state:
        got = rearrange_to_dense_trunk(q, q, q, *windows)
        again = rearrange_to_dense_trunk(q + 1, q + 1, q + 1, *windows)
        for x, y in zip(got[:-1], want[:-1], strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        assert got[-1] == want[-1]
        assert got[3] is again[3]
        assert state.mask_bytes == got[3].numel() * got[3].element_size()
        assert len(state.masks) == 1
    assert state.mask_bytes == 0 and not state.masks


def test_mask_budget_disabled_and_explicit_bias_fallback():
    q = torch.randn(2, 11, 8)
    bias = torch.randn(2, 11, 11)
    want = rearrange_to_dense_trunk(q, q, q, 4, 8, bias)
    with torch.no_grad(), diffusion_attention_scope("reference", True) as state:
        state.max_mask_bytes = 0
        got = rearrange_to_dense_trunk(q, q, q, 4, 8)
        assert not state.masks and state.mask_bytes == 0
        assert torch.isfinite(got[3]).all()
        actual = rearrange_to_dense_trunk(q, q, q, 4, 8, bias)
        for x, y in zip(actual[:-1], want[:-1], strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    with torch.no_grad(), diffusion_attention_scope("reference", False) as state:
        rearrange_to_dense_trunk(q, q, q, 4, 8)
        assert not state.masks


@pytest.mark.parametrize("shape", [(3, 7, 16), (2, 3, 4, 7, 16)])
@pytest.mark.parametrize("bias", [False, True])
def test_supported_layout_matches_cpu_reference(shape, bias):
    q = torch.randn(*shape) * 0.1
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.randn(*shape[:-2], shape[-2], shape[-2]) if bias else None
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    got = _attention_supported_layout(q, k, v, mask)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
    # The public primitive retains the CPU path even when efficient is selected.
    with torch.no_grad(), diffusion_attention_scope("efficient", True):
        torch.testing.assert_close(_attention(q, k, v, mask), want, atol=0, rtol=0)


CUDA = (
    os.environ.get("OPENDDE_RUN_BATCH_KERNEL_TESTS") == "1"
    and torch.cuda.is_available()
)


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_BATCH_KERNEL_TESTS=1 on CUDA")
@pytest.mark.parametrize("shape", [(12, 32, 48), (3, 5, 4, 64, 48)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_kind", ["none", "broadcast", "full"])
def test_cuda_attention_layouts_preserve_masks_and_precision(shape, dtype, bias_kind):
    torch.manual_seed(153)
    # Non-contiguous sequence/head layouts occur in the actual transformer.
    q = (
        torch.randn(
            *shape[:-2], shape[-1], shape[-2], device="cuda", dtype=dtype
        ).transpose(-1, -2)
        * 0.1
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = None
    if bias_kind != "none":
        prefix = list(shape[:-2])
        if bias_kind == "broadcast":
            prefix[0] = 1
            if len(shape) == 5:
                prefix[1] = 1
        bias = torch.randn(*prefix, shape[-2], shape[-2], device="cuda")
        bias[..., -3:] = -1e10
    with torch.no_grad():
        want = _attention(q, k, v, bias)
        with diffusion_attention_scope("efficient", True):
            got = _attention(q, k, v, bias)
    assert got.dtype == dtype and torch.isfinite(got).all()
    tolerance = 0.008 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)


def test_config_defaults_keep_reference_backend():
    required = dict(
        chunk_size=None,
        dynamic_chunk_size=False,
        chunk_size_thresholds={},
        sample_diffusion_chunk_size=5,
    )
    config = InferSettingConfig(**required)
    assert config.diffusion_attention_backend == "reference"
    assert config.reuse_diffusion_window_masks is True
    with pytest.raises(ValueError):
        InferSettingConfig(**required, diffusion_attention_backend="invalid")


@pytest.mark.parametrize("q_jobs,kv_jobs,bias_jobs", [(3, 1, 3), (1, 3, 1)])
def test_global_job_broadcast_and_fallback(q_jobs, kv_jobs, bias_jobs):
    q = torch.randn(q_jobs, 2, 4, 7, 16) * 0.1
    k = torch.randn(kv_jobs, 2, 4, 7, 16)
    v = torch.randn_like(k)
    bias = torch.randn(bias_jobs, 1, 4, 7, 7)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
    got = _attention_supported_layout(q, k, v, bias)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def test_invalid_global_mask_does_not_silently_drop_jobs():
    q = torch.randn(1, 2, 4, 7, 16)
    bias = torch.randn(3, 1, 4, 7, 7)
    with pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(q, q, q, attn_mask=bias, scale=1.0)
    with pytest.raises(RuntimeError):
        _attention_supported_layout(q, q, q, bias)


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_BATCH_KERNEL_TESTS=1 on CUDA")
@pytest.mark.parametrize("sample_chunk", [2, 3, 5])
def test_eight_job_cuda_model_keeps_sample_and_atom_ownership(sample_chunk):
    import copy
    from tests.test_input_batching import _toy_config, _toy_features
    from opendde.model.opendde import OpenDDE
    from opendde.utils.torch_utils import to_device

    torch.manual_seed(131)
    cfg = _toy_config()
    cfg.sample_diffusion.N_sample = 5
    cfg.infer_setting.sample_diffusion_chunk_size = sample_chunk
    cfg.enable_efficient_fusion = True
    cfg.enable_diffusion_shared_vars_cache = True
    model = OpenDDE(cfg).cuda().eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim > 1 and not param.count_nonzero():
                param.normal_(std=0.02)
        features = [
            to_device(_toy_features(n, 101 + i), torch.device("cuda"))
            for i, n in enumerate(range(3, 11))
        ]
        expected = model.forward_batch(copy.deepcopy(features))
        cfg.infer_setting.diffusion_attention_backend = "efficient"
        actual = model.forward_batch(copy.deepcopy(features))
    assert get_diffusion_attention_execution() is None
    assert len(actual) == len(expected) == 8
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed["coordinate"].shape == wanted["coordinate"].shape
        assert observed["coordinate"].shape[0] == 5
        for key in ("coordinate", "plddt", "pae", "pde", "resolved"):
            torch.testing.assert_close(observed[key], wanted[key], atol=5e-4, rtol=2e-4)
        assert len(observed["summary_confidence"]) == 5
