"""Request-owned CUDA replay of a deterministic denoiser, with eager sampling."""

import time
import threading

import torch

from opendde.utils.logger import get_logger

logger = get_logger(__name__)
_capture_streams = threading.local()


class DenoiserGraphCaptureError(RuntimeError):
    """Fatal capture failure: stop this worker; its CUDA state may be invalid.

    This deliberately is not an OutOfMemoryError, even when capture failed
    from OOM. Retrying or splitting work in this process would be unsafe.
    """


def _capture_stream(device):
    # cuBLAS retains a workspace for each stream. Reuse the stream, not graph
    # buffers/conditioning, so a long queue does not accumulate workspaces.
    streams = getattr(_capture_streams, "streams", None)
    if streams is None:
        streams = _capture_streams.streams = {}
    if device not in streams:
        streams[device] = torch.cuda.Stream(device=device)
    return streams[device]


class DenoiserGraph:
    """Capture one shape at a time; never retain topology across requests.

    The callable must be deterministic and its conditioning immutable. Warmup
    must populate request caches before capture. Returned tensors own their
    storage and survive the next replay. Random draws belong to the caller.
    """

    def __init__(self, denoise, *, enabled=False):
        self.denoise = denoise
        self.enabled = enabled
        self.graph = None
        self.signature = None
        self.inputs = None
        self.outputs = None
        self.stats = {
            "enabled": enabled,
            "captures": 0,
            "replays": 0,
            "eager_calls": 0,
            "setup_seconds": 0.0,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # A failed capture can invalidate the CUDA context in Torch 2.7.
        # Do not hide that error with another CUDA call during cleanup.
        if exc_type is None:
            self.close()
        if self.enabled:
            logger.info("Diffusion CUDA graph: %s", self.stats)

    def close(self):
        if self.graph is not None:
            torch.cuda.current_stream(self.inputs[0][0].device).synchronize()
            self.graph.reset()
        self.graph = self.inputs = self.outputs = self.signature = None

    def __call__(self, noisy, levels):
        tensors = [*noisy, *levels]
        if (
            not self.enabled
            or torch.is_grad_enabled()
            or torch.are_deterministic_algorithms_enabled()
            or not tensors
            or not all(x.is_cuda and x.device == tensors[0].device for x in tensors)
        ):
            self.stats["eager_calls"] += 1
            return self.denoise(noisy, levels)
        signature = (
            tuple((tuple(x.shape), x.dtype, x.device) for x in tensors),
            len(noisy),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        if signature != self.signature:
            self.close()
            self._capture(noisy, levels)
            self.signature = signature
        for buffers, values in zip(self.inputs, (noisy, levels), strict=True):
            for target, value in zip(buffers, values, strict=True):
                target.copy_(value)
        self.graph.replay()
        self.stats["replays"] += 1
        return [value.clone() for value in self.outputs]

    def _capture(self, noisy, levels):
        start = time.perf_counter()
        self.inputs = ([x.clone() for x in noisy], [x.clone() for x in levels])
        device = noisy[0].device
        with torch.cuda.device(device):
            caller = torch.cuda.current_stream()
            stream = _capture_stream(device)
            stream.wait_stream(caller)
            try:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.denoise(*self.inputs)
            finally:
                # A warmup OOM is retryable, but already queued side-stream
                # work must finish before the caller reuses allocator blocks.
                caller.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    self.outputs = self.denoise(*self.inputs)
            except Exception as error:
                raise DenoiserGraphCaptureError(
                    "Diffusion CUDA graph capture failed. Restart this worker with "
                    "diffusion_cuda_graph=false; eager retry in the same CUDA "
                    "context is unsafe after a capture failure."
                ) from error
            self.graph = graph
        self.stats["captures"] += 1
        self.stats["setup_seconds"] += time.perf_counter() - start
