"""The optional CUDA extension must honor caller streams and graph capture."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_FAST_LAYERNORM_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set TEST_FAST_LAYERNORM_CUDA=1 on a CUDA host with a compiler",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "scale,offset", [(False, False), (True, False), (False, True), (True, True)]
)
def test_side_stream_and_replay_match_fused_eager(dtype, scale, offset):
    from opendde.model.layer_norm.layer_norm import (
        FusedLayerNorm,
        _load_fast_layer_norm_cuda_v2,
    )

    assert _load_fast_layer_norm_cuda_v2() is not None, (
        "Test must exercise the extension, not fallback"
    )
    layer = FusedLayerNorm(384, create_scale=scale, create_offset=offset).cuda().eval()
    torch.manual_seed(101)
    x = torch.randn(2, 32, 384, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        expected = layer(x)
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        static = x.clone()
        with torch.cuda.graph(graph, stream=stream):
            output = layer(static)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        static.copy_(torch.randn_like(static))
        expected = layer(static)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
