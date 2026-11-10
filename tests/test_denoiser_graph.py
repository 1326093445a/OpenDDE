"""Replay must preserve RNG, topology, ragged ownership and eager fallbacks."""

import os
from contextlib import nullcontext
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from opendde.model.denoiser_graph import DenoiserGraph, DenoiserGraphCaptureError
from opendde.model.inference_cache import diffusion_attention_scope
from opendde.model.utils import _aggregate_atom_to_token_grouped


def test_cpu_is_eager_and_does_not_allocate_cuda_state():
    fn = Mock(side_effect=lambda x, t: [x[0] * t[0]])
    with patch("torch.cuda.CUDAGraph", side_effect=AssertionError("CPU")):
        with DenoiserGraph(fn, enabled=True) as graph:
            x, t = torch.ones(3), torch.tensor(2.0)
            assert torch.equal(graph([x], [t])[0], x * t)
            assert graph.stats["eager_calls"] == 1
            assert graph.stats["captures"] == graph.stats["replays"] == 0
    assert graph.inputs is graph.outputs is graph.graph is None


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
@pytest.mark.parametrize("phase", ["enter", "body", "exit", "warmup"])
def test_capture_failure_is_fatal_but_warmup_oom_is_retryable(error_type, phase):
    # Inject failures without poisoning a real CUDA context. Include context
    # entry/exit: CUDA may report an asynchronous capture failure at either.
    error = error_type("injected failure")
    capture = MagicMock()
    capture.__exit__.return_value = False
    fn = Mock(return_value=[torch.ones(1)])
    if phase == "body":
        fn.side_effect = [[torch.ones(1)]] * 3 + [error]
    elif phase == "warmup":
        fn.side_effect = error
    else:
        getattr(capture, f"__{phase}__").side_effect = error
    with (
        patch("torch.cuda.device", return_value=nullcontext()),
        patch("torch.cuda.current_stream") as current_stream,
        patch("opendde.model.denoiser_graph._capture_stream"),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.CUDAGraph"),
        patch("torch.cuda.graph", return_value=capture) as capture_factory,
        patch.object(DenoiserGraph, "close") as close,
    ):
        expected = error_type if phase == "warmup" else DenoiserGraphCaptureError
        with pytest.raises(expected) as caught:
            with DenoiserGraph(fn, enabled=True) as graph:
                graph._capture([torch.ones(1)], [torch.ones(1)])
        close.assert_not_called()
        current_stream.return_value.wait_stream.assert_called_once()
        if phase == "warmup":
            assert caught.value is error
            capture_factory.assert_not_called()
        else:
            assert caught.value.__cause__ is error
            assert not isinstance(caught.value, torch.cuda.OutOfMemoryError)
            assert "Restart this worker" in str(caught.value)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_planned_aggregation_preserves_empty_tokens_and_releases_plans(reduce):
    indices = torch.tensor([2, 0, 2, 4, 0])
    x = torch.randn(2, 5, 7)
    expected = _aggregate_atom_to_token_grouped(x, indices, 6, reduce)
    with (
        torch.no_grad(),
        diffusion_attention_scope(
            "reference", True, reuse_aggregation_plans=True
        ) as execution,
    ):
        first = _aggregate_atom_to_token_grouped(x, indices, 6, reduce)
        with patch("torch.bincount", side_effect=AssertionError("must reuse")):
            second = _aggregate_atom_to_token_grouped(x, indices, 6, reduce)
        assert torch.equal(first, expected) and torch.equal(first, second)
        assert len(execution.aggregation_plans) == 1
        with pytest.raises(RuntimeError, match="exceeds n_token"):
            _aggregate_atom_to_token_grouped(x, indices, 3, reduce)
    assert not execution.aggregation_plans


CUDA = os.environ.get("OPENDDE_RUN_GRAPH_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_GRAPH_TESTS=1 on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_replay_ragged_outputs_rng_tail_shape_and_request_isolation(dtype):
    indices = [
        torch.tensor([0, 1, 0, 2], device="cuda"),
        torch.tensor([1, 0, 1], device="cuda"),
    ]
    generator = torch.Generator(device="cuda").manual_seed(71)
    x = [
        torch.randn(2, len(idx), 4, device="cuda", dtype=dtype, generator=generator)
        for idx in indices
    ]
    times = [
        torch.tensor([1.5, 2.0], device="cuda", dtype=dtype).view(2, 1, 1) for _ in x
    ]
    calls = []

    def denoise(values, levels):
        calls.append(True)
        return [
            _aggregate_atom_to_token_grouped(v, idx, 3, "mean") * t
            for v, idx, t in zip(values, indices, levels, strict=True)
        ]

    with torch.inference_mode():
        expected = denoise(x, times)
        rng = torch.cuda.get_rng_state()
        job_rng = generator.get_state()
        with (
            diffusion_attention_scope("reference", True, reuse_aggregation_plans=True),
            DenoiserGraph(denoise, enabled=True) as graph,
        ):
            first = graph(x, times)
            saved = [v.clone() for v in first]
            repeat = graph(x, times)
            changed = graph([v + 3 for v in x], times)
            for want, actual, again, kept in zip(
                expected, first, repeat, saved, strict=True
            ):
                assert torch.equal(want, actual) and torch.equal(actual, again)
                assert torch.equal(actual, kept)
            assert not torch.equal(changed[0], first[0])
            assert len(calls) == 5  # eager reference + 3 warmups + capture
            tail_x, tail_t = [v[:1] for v in x], [v[:1] for v in times]
            tail = graph(tail_x, tail_t)
            assert tail[0].shape == (1, 3, 4)
            assert graph.stats["captures"] == 2
            assert graph.stats["replays"] == 4
        assert graph.graph is graph.inputs is graph.outputs is None
        assert torch.equal(torch.cuda.get_rng_state(), rng)
        assert torch.equal(generator.get_state(), job_rng)
        # Same shapes, different conditioning belong to a fresh request.
        with DenoiserGraph(
            lambda values, levels: [v * 7 for v in values], enabled=True
        ) as other:
            assert torch.equal(other(x, times)[0], x[0] * 7)


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_GRAPH_TESTS=1 on CUDA")
def test_gradients_and_explicit_deterministic_mode_stay_eager():
    x = torch.randn(3, device="cuda", requires_grad=True)
    with DenoiserGraph(lambda values, levels: [values[0] * 2], enabled=True) as graph:
        graph([x], [x])[0].sum().backward()
        assert torch.equal(x.grad, torch.full_like(x, 2))
        with (
            torch.no_grad(),
            patch("torch.are_deterministic_algorithms_enabled", return_value=True),
        ):
            graph([x], [x])
        assert graph.stats["eager_calls"] == 2 and graph.stats["captures"] == 0


@pytest.mark.skipif(not CUDA, reason="Set OPENDDE_RUN_GRAPH_TESTS=1 on CUDA")
def test_closed_graphs_do_not_accumulate_tensor_storage():
    # Include GEMM: a fresh warmup stream per request retains a cuBLAS workspace
    # even when all graph tensors have been released.
    x = torch.ones(32, 32, device="cuda")
    used = []
    with torch.inference_mode():
        for _ in range(8):
            with DenoiserGraph(
                lambda values, levels: [values[0] @ values[0]], enabled=True
            ) as graph:
                result = graph([x], [x])
            del result, graph
            torch.cuda.synchronize()
            used.append(torch.cuda.memory_allocated())
    assert max(used[2:]) - min(used[2:]) < 1024**2
