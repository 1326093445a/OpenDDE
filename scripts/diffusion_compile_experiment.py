"""Benchmark-only transition compilation; prediction defaults are unaffected.

Keep attention, RNG and request-local caches eager. This experiment deliberately
raises compiler errors instead of silently measuring an eager fallback. CUDA
graphs are disabled so output storage has the same lifetime as eager tensors.
"""

from contextlib import contextmanager
from types import FunctionType

import torch
from torch.nn import functional as F

from opendde.model.modules.transformer import ConditionedTransitionBlock
from opendde.model.modules.primitives import _attention_supported_layout
from opendde.model.inference_cache import get_diffusion_attention_execution


@contextmanager
def cached_atom_aggregation():
    """Benchmark topology reuse alone; exclusive access to this process is required."""
    from opendde.model import batching

    original = batching.sample_diffusion_batch
    stats = {"requests": 0, "plans": 0}

    def sample(*args, **kwargs):
        execution = get_diffusion_attention_execution()
        previous = execution.aggregation_plans
        execution.aggregation_plans = {}
        try:
            stats["requests"] += 1
            return original(*args, **kwargs)
        finally:
            stats["plans"] += len(execution.aggregation_plans)
            execution.aggregation_plans.clear()
            execution.aggregation_plans = previous

    batching.sample_diffusion_batch = sample
    try:
        yield stats
    finally:
        batching.sample_diffusion_batch = original


def token_block_with_bias(block, a, s, bias, backend):
    """Standard token block with explicit request inputs, without ContextVar reads.

    Uses the original projections, normalization, gates and transition. Pair
    projection is already cached by the eager first step. Only the SDPA dispatch
    is explicit here; tests pin this function to the original block's arithmetic.
    """
    apb = block.attention_pair_bias
    normalized = apb.layernorm_a(a, s)
    attention = apb.attention
    q, k, v = attention._prep_qkv(normalized, normalized)
    bias = apb._align_bias_to_query(bias, normalized, n_pair_dims=2)
    if len(bias.shape) != len(q.shape):
        bias = bias.unsqueeze(-3)
    dtype = q.dtype
    q, k, v, bias = q.float(), k.float(), v.float(), bias.float()
    if backend == "efficient" and q.is_cuda:
        out = _attention_supported_layout(q, k, v, bias)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
    out = attention._wrap_up(out.to(dtype).transpose(-2, -3), normalized)
    out *= torch.sigmoid(apb.linear_a_last(s))
    out = block.residual_path(out)
    out += a
    return block.residual_path(block.conditioned_transition_block(out, s)) + out


@contextmanager
def compiled_diffusion_blocks(transformer):
    """Benchmark whole token blocks with explicit bias/backend and no CUDA graphs.

    Each block has an independent Dynamo code object, avoiding a process-global
    cache-limit increase. The eager first step prepares the request's pair cache.
    No attention getter or process-global execution state is monkeypatched.
    """
    stats = {"enabled": True, "modules": 0, "compiled_calls": 0, "eager_calls": 0}
    originals = []

    def wrap(block, original, index):
        compiled = None

        def run(a, s, bias, backend):
            return token_block_with_bias(block, a, s, bias, backend)

        run = FunctionType(
            run.__code__.replace(co_name=f"diffusion_token_block_{index}"),
            run.__globals__,
            closure=run.__closure__,
        )

        def forward(a, s, z, **kwargs):
            nonlocal compiled
            cache = kwargs.get("pair_bias_cache")
            apb = block.attention_pair_bias
            execution = get_diffusion_attention_execution()
            if (
                block.training
                or torch.is_grad_enabled()
                or not a.is_cuda
                # Torch 2.7 cannot trace the BF16 LayerNorm autocast context.
                # Preserve the precision policy instead of forcing FP32.
                or torch.is_autocast_enabled("cuda")
                or a.dtype != torch.float32
                or s.dtype != torch.float32
                or execution is None
                or cache is None
                or apb not in cache
                or kwargs.get("n_queries")
                or kwargs.get("atom_cache") is not None
                or apb.cross_attention_mode
                or not apb.attention.use_efficient_implementation
            ):
                stats["eager_calls"] += 1
                return original(a, s, z, **kwargs)
            if compiled is None:
                compiled = torch.compile(
                    run,
                    fullgraph=True,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                stats["modules"] += 1
            result = compiled(a, s, cache[apb], execution.backend)
            stats["compiled_calls"] += 1
            return result, s, z

        return forward

    try:
        for index, block in enumerate(transformer.blocks):
            originals.append((block, "forward" in block.__dict__, block.forward))
            block.forward = wrap(block, block.forward, index)
        yield stats
    finally:
        for block, had_override, original in reversed(originals):
            if had_override:
                block.forward = original
            else:
                del block.forward


@contextmanager
def compiled_diffusion_transitions(diffusion_module, enabled=False):
    """Temporarily compile CUDA inference transitions without replacing modules.

    Enter after checkpoint loading/device placement, with exclusive access to
    the model. Forward overrides are removed on context exit;
    checkpoint names and CPU/training behavior are preserved.
    """
    stats = {"enabled": enabled, "modules": 0, "compiled_calls": 0}
    restorations = []

    def wrap(module, original):
        compiled = None

        def forward(a, s, atom_cache=None):
            nonlocal compiled
            if module.training or torch.is_grad_enabled() or not a.is_cuda:
                return original(a, s, atom_cache=atom_cache)
            if compiled is None:
                compiled = torch.compile(
                    original,
                    fullgraph=True,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                stats["modules"] += 1
            result = compiled(a, s, atom_cache=atom_cache)
            stats["compiled_calls"] += 1
            return result

        return forward

    try:
        if enabled:
            for module in diffusion_module.modules():
                if isinstance(module, ConditionedTransitionBlock):
                    restorations.append(
                        (module, "forward" in module.__dict__, module.forward)
                    )
                    module.forward = wrap(module, module.forward)
        yield stats
    finally:
        for module, had_override, original in reversed(restorations):
            if had_override:
                module.forward = original
            else:
                del module.forward
