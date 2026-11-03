# SPDX-License-Identifier: Apache-2.0
"""Resident-memory accounting for request-local inference projections."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch


@dataclass
class InferenceCacheBudget:
    """Bound retained projections, not model activations or build workspace."""

    max_bytes: int
    used_bytes: int = 0

    @classmethod
    def for_device(cls, max_mb: int, device: torch.device):
        if max_mb < 0:
            raise ValueError("diffusion_cache_max_mb must be nonnegative")
        limit = int(max_mb) * 1024**2
        if device.type == "cuda" and limit:
            free, _ = torch.cuda.mem_get_info(device)
            # Cached allocator blocks can be reused by this request. Retain
            # at most one eighth of currently available memory for speed caches.
            available = (
                free
                + torch.cuda.memory_reserved(device)
                - torch.cuda.memory_allocated(device)
            )
            limit = min(limit, max(0, available // 8))
        elif device.type != "cuda":
            limit = min(limit, 256 * 1024**2)
        return cls(limit)

    def reserve(self, size: int) -> bool:
        if size < 0:
            raise ValueError("Cache reservation must be nonnegative")
        if size > self.max_bytes - self.used_bytes:
            return False
        self.used_bytes += size
        return True


@dataclass
class DiffusionAttentionExecution:
    """Execution choices and bounded masks belonging to one sampling request."""

    backend: str
    reuse_window_masks: bool
    masks: dict[tuple, torch.Tensor] = field(default_factory=dict)
    mask_bytes: int = 0
    max_mask_bytes: int = 16 * 1024**2


_DIFFUSION_ATTENTION: ContextVar[DiffusionAttentionExecution | None] = ContextVar(
    "opendde_diffusion_attention", default=None
)


def get_diffusion_attention_execution() -> DiffusionAttentionExecution | None:
    return _DIFFUSION_ATTENTION.get()


@contextmanager
def diffusion_attention_scope(backend: str, reuse_window_masks: bool):
    """Keep attention choices local to a thread/task and reset even on failure.

    Called only around native batched diffusion. Trunk, confidence, training,
    serial and Fold-CP calls retain their normal execution. Nothing is stored
    on model modules, and no cache survives this sampling request.
    """
    if backend not in ("reference", "efficient"):
        raise ValueError("diffusion_attention_backend must be reference or efficient")
    if torch.is_grad_enabled():
        raise ValueError(
            "Diffusion attention reuse requires inference without gradients"
        )
    execution = DiffusionAttentionExecution(backend, reuse_window_masks)
    token = _DIFFUSION_ATTENTION.set(execution)
    try:
        yield execution
    finally:
        execution.masks.clear()
        execution.mask_bytes = 0
        _DIFFUSION_ATTENTION.reset(token)
