"""Bound normalization work without changing per-row arithmetic or gradients."""

import pytest
import torch
from torch.nn import functional as F

from opendde.model.triangular import layers


@pytest.mark.parametrize("shape", [(3, 5, 7, 8), (1, 1, 30, 8)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_chunked_layer_norm_matches_rows_and_gradients(
    monkeypatch, shape, transpose, affine
):
    torch.manual_seed(81)
    source = torch.randn(shape, dtype=torch.float64)
    if transpose:
        source = source.transpose(1, 2)
    x = source.detach().requires_grad_()
    y = source.detach().clone().requires_grad_()
    layer = layers.OpenFoldLayerNorm(
        8, create_scale=affine, create_offset=affine
    ).double()
    weight = layer.weight.detach().clone().requires_grad_() if affine else None
    bias = layer.bias.detach().clone().requires_grad_() if affine else None
    reference = F.layer_norm(y, (8,), weight, bias, layer.eps)
    monkeypatch.setattr(layers, "_LAYER_NORM_MAX_ELEMENTS", 80)
    actual = layer(x)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    coefficient = torch.randn_like(actual)
    (actual * coefficient).sum().backward()
    (reference * coefficient).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
    if affine:
        torch.testing.assert_close(
            layer.weight.grad, weight.grad, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(layer.bias.grad, bias.grad, atol=1e-12, rtol=1e-12)


def test_large_single_normalization_axis_keeps_native_semantics(monkeypatch):
    x = torch.randn(1, 1, 8)
    monkeypatch.setattr(layers, "_LAYER_NORM_MAX_ELEMENTS", 4)
    layer = layers.OpenFoldLayerNorm(8)
    torch.testing.assert_close(
        layer(x),
        F.layer_norm(x, (8,), layer.weight, layer.bias, layer.eps),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_layer_norm_preserves_dtype_and_empty_batches(monkeypatch, dtype):
    layer = layers.OpenFoldLayerNorm(8)
    x = torch.randn(5, 7, 8, dtype=dtype)
    with torch.inference_mode():
        want = layer(x)
        monkeypatch.setattr(layers, "_LAYER_NORM_MAX_ELEMENTS", 80)
        got = layer(x)
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        assert got.dtype == dtype
        assert layer(x[:0]).shape == (0, 7, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_chunked_normalization_keeps_transposed_rows(monkeypatch, dtype):
    layer = layers.OpenFoldLayerNorm(32).cuda()
    x = torch.randn(3, 19, 23, 32, device="cuda", dtype=dtype).transpose(1, 2)
    with torch.inference_mode():
        want = layer(x)
        monkeypatch.setattr(layers, "_LAYER_NORM_MAX_ELEMENTS", 2000)
        actual = layer(x)
        torch.testing.assert_close(actual, want, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_chunked_normalization_preserves_autocast_promotion(monkeypatch):
    layer = layers.OpenFoldLayerNorm(32).cuda()
    x = torch.randn(3, 19, 23, 32, device="cuda", dtype=torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        want = layer(x)
        monkeypatch.setattr(layers, "_LAYER_NORM_MAX_ELEMENTS", 2000)
        actual = layer(x)
    assert actual.dtype == want.dtype
    torch.testing.assert_close(actual, want, atol=0, rtol=0)
