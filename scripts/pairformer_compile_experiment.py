"""Benchmark-only whole Pairformer block compilation on CUDA with cuEquivariance.

Temporary, instance-local wrappers make the CUDA autocast boundary and trunk
attention explicit to Dynamo. Eager calls retain their original methods. This
context requires exclusive access to the model and changes no prediction API.
"""

from contextlib import contextmanager
import os
from types import FunctionType

import torch
from torch.nn import functional as F

from opendde.model.inference_cache import get_diffusion_attention_execution
from opendde.model.triangular.layers import (
    OpenFoldLayerNorm,
    OpenfoldLinear,
    _bounded_layer_norm,
)


def cuda_layer_norm(module, x):
    """Same dtype/large-tensor policy as OpenFoldLayerNorm, without ExitStack."""
    if x.dtype == torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            return _bounded_layer_norm(
                x,
                module.c_in,
                None if module.weight is None else module.weight.to(x.dtype),
                None if module.bias is None else module.bias.to(x.dtype),
                module.eps,
            )
    return _bounded_layer_norm(x, module.c_in, module.weight, module.bias, module.eps)


def cuda_linear(module, x):
    """Same explicit precision and BF16 parameter rounding as OpenfoldLinear."""
    dtype = module.precision or x.dtype
    if module.precision is not None or x.dtype == torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            return F.linear(
                x.to(dtype),
                module.weight.to(dtype),
                None if module.bias is None else module.bias.to(dtype),
            ).to(x.dtype)
    return F.linear(x, module.weight, module.bias)


def trunk_attention(module, q_x, kv_x, attn_bias=None, **kwargs):
    """Original global SDPA arithmetic; trunk has no diffusion execution scope."""
    if (
        kwargs.get("n_queries")
        or kwargs.get("n_keys")
        or kwargs.get("trunked_attn_bias") is not None
    ):
        raise ValueError("Pairformer compilation supports global attention only")
    q, k, v = module._prep_qkv(q_x, kv_x, apply_scale=True)
    if attn_bias is not None and attn_bias.ndim != q.ndim:
        attn_bias = attn_bias.unsqueeze(-3)
    output = F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None if attn_bias is None else attn_bias.float(),
        scale=1.0,
    ).to(q.dtype)
    return module._wrap_up(output.transpose(-2, -3), q_x)


@contextmanager
def compiled_pairformer_blocks(stack):
    """Compile only this stack, fullgraph/static shapes, without CUDA replay.

    CPU, training, lengths <=16, unsupported kernels and an active diffusion
    attention scope remain eager. Compilation failures propagate; they are never
    timed as an eager fallback. Checkpoint names and original method overrides
    are restored. This is not numerically equivalent to eager BF16 inference.
    """
    if os.getenv("OPENDDE_FOLDCP_MODE") == "distributed":
        raise ValueError("Pairformer compilation experiment does not support Fold-CP")
    stats = {"enabled": True, "modules": 0, "compiled_calls": 0, "eager_calls": 0}
    originals = []

    def replace(module, name, forward):
        originals.append((module, name, name in module.__dict__, getattr(module, name)))
        setattr(module, name, forward)

    def explicit_when_compiling(module, original, implementation):
        def forward(*args, **kwargs):
            if torch.compiler.is_compiling():
                return implementation(module, *args, **kwargs)
            return original(*args, **kwargs)

        return forward

    def wrap(block, original, index):
        compiled = None

        def run(s, z, **kwargs):
            return original(s, z, **kwargs)

        run = FunctionType(
            run.__code__.replace(co_name=f"pairformer_block_{index}"),
            run.__globals__,
            closure=run.__closure__,
        )

        def forward(s, z, **kwargs):
            nonlocal compiled
            if (
                block.training
                or torch.is_grad_enabled()
                or not z.is_cuda
                or z.shape[-2] <= 16
                or kwargs.get("triangle_multiplicative", "torch") != "cuequivariance"
                or kwargs.get("triangle_attention", "torch") != "cuequivariance"
                or get_diffusion_attention_execution() is not None
                or (
                    block.c_s > 0
                    and not block.attention_pair_bias.attention.use_efficient_implementation
                )
            ):
                stats["eager_calls"] += 1
                return original(s, z, **kwargs)
            if compiled is None:
                compiled = torch.compile(
                    run,
                    fullgraph=True,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                stats["modules"] += 1
            result = compiled(s, z, **kwargs)
            stats["compiled_calls"] += 1
            return result

        return forward

    try:
        for module in stack.modules():
            implementation = (
                cuda_layer_norm
                if isinstance(module, OpenFoldLayerNorm)
                else cuda_linear
                if isinstance(module, OpenfoldLinear)
                else None
            )
            if implementation is not None:
                replace(
                    module,
                    "forward",
                    explicit_when_compiling(module, module.forward, implementation),
                )
        for index, block in enumerate(stack.blocks):
            if block.c_s > 0:
                attention = block.attention_pair_bias.attention
                replace(
                    attention,
                    "forward",
                    explicit_when_compiling(
                        attention, attention.forward, trunk_attention
                    ),
                )
            replace(block, "forward_source", wrap(block, block.forward_source, index))
        yield stats
    finally:
        for module, name, had_override, original in reversed(originals):
            if had_override:
                setattr(module, name, original)
            else:
                delattr(module, name)
