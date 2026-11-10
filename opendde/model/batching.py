# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Ragged inference: batch token networks, retain per-job atom topology.

Only named token axes are padded. Atom arrays, reference conformers, chain
identities, RNG streams and confidence summaries never acquire a job axis.
This avoids interpreting padding as atoms or a job dimension as samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from opendde.model.denoiser_graph import DenoiserGraph
from opendde.model.inference_cache import (
    InferenceCacheBudget,
    diffusion_attention_scope,
)
from opendde.model.utils import centre_random_augmentation, permute_final_dims
from opendde.utils.torch_utils import autocasting_disable_decorator
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


def pad_token_tensors(
    values: list[torch.Tensor], *, pair: bool = False, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Stack [*, T, C] or [*, T, T, C], padding only their token axes."""
    if not values:
        raise ValueError("Cannot pad an empty inference batch.")
    spatial_dims = 3 if pair else 2
    prefix = values[0].shape[:-spatial_dims]
    width = values[0].shape[-1]
    for value in values:
        if value.shape[:-spatial_dims] != prefix or value.shape[-1] != width:
            raise ValueError("Batch members must share sample and channel dimensions.")
        if value.shape[-2] < 1 or (pair and value.shape[-3] != value.shape[-2]):
            raise ValueError(
                "Token dimensions must be nonempty; pair tensors must be square."
            )
    size = max(value.shape[-2] for value in values)
    shape = (len(values), *prefix, *((size, size) if pair else (size,)), width)
    if out is None:
        out = values[0].new_zeros(shape)
    else:
        if (
            out.shape != shape
            or out.dtype != values[0].dtype
            or out.device != values[0].device
        ):
            raise ValueError("Padding buffer has incompatible shape, dtype or device")
        # Callers reusing storage must supply values that do not alias it.
        out.zero_()
    for i, value in enumerate(values):
        n = value.shape[-2]
        if pair:
            out[i, ..., :n, :n, :].copy_(value)
        else:
            out[i, ..., :n, :].copy_(value)
    return out


def token_masks(
    lengths: list[int],
    reference: torch.Tensor,
    biases: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair validity and finite attention bias, including per-job structural bias."""
    size = max(lengths)
    valid = (
        torch.arange(size, device=reference.device)[None, :]
        < torch.tensor(lengths, device=reference.device)[:, None]
    )
    pair = valid[:, :, None] & valid[:, None, :]
    bias = reference.new_zeros(len(lengths), size, size)
    if biases is not None:
        for i, (length, item) in enumerate(zip(lengths, biases, strict=True)):
            if item is not None:
                if item.shape != (length, length):
                    raise ValueError(
                        "Structural attention bias has the wrong token axes."
                    )
                bias[i, :length, :length] = item
    # Finite masks also keep completely padded query rows finite. Those rows
    # are discarded; valid queries cannot attend to padded keys.
    bias.masked_fill_(~pair, -1e9)
    return pair.to(reference.dtype), bias


def pairformer_batch(stack, singles, pairs, *, biases=None, chunk_size=None, **kernels):
    lengths = [value.shape[-2] for value in singles]
    s = pad_token_tensors(singles)
    z = pad_token_tensors(pairs, pair=True)
    mask, bias = token_masks(lengths, z, biases)
    s, z = stack(
        s=s,
        z=z,
        pair_mask=mask,
        extra_attn_bias=bias,
        inplace_safe=True,
        chunk_size=chunk_size,
        **kernels,
    )
    # Clone: cropped views must not retain the complete padded batch, and the
    # next per-job in-place stage must own its storage.
    return (
        [s[i, :n].clone() for i, n in enumerate(lengths)],
        [z[i, :n, :n].clone() for i, n in enumerate(lengths)],
    )


def _trunk_batch(model, features, chunk_size, kernels):
    inputs, initial_s, initial_z = [], [], []
    for feat in features:
        s_input = model.input_embedder(feat, inplace_safe=False, chunk_size=chunk_size)
        s_init = model.linear_no_bias_sinit(s_input)
        z_init = (
            model.linear_no_bias_zinit1(s_init)[:, None, :]
            + model.linear_no_bias_zinit2(s_init)[None, :, :]
            + model.relative_position_encoding(feat["relp"])
            + model.linear_no_bias_token_bond(feat["token_bonds"].unsqueeze(-1))
        )
        inputs.append(s_input)
        initial_s.append(s_init)
        initial_z.append(z_init)
    singles = [torch.zeros_like(x) for x in initial_s]
    pairs = [torch.zeros_like(x) for x in initial_z]
    lengths = [s.shape[-2] for s in singles]
    mask, bias = token_masks(lengths, initial_z[0])
    s_batch, z_batch = None, None
    for _ in range(model.N_cycle):
        for i, feat in enumerate(features):
            z = initial_z[i] + model.linear_no_bias_z_cycle(
                model.layernorm_z_cycle(pairs[i])
            )
            # Keep the query-only MSA and empty-template semantics of B=1.
            # These per-job calls never broadcast one job into another's MSA.
            if model.template_embedder.n_blocks > 0:
                z = z + model.template_embedder(
                    feat, z, inplace_safe=True, chunk_size=chunk_size, **kernels
                )
            pairs[i] = model.msa_module(
                feat,
                z,
                inputs[i],
                pair_mask=None,
                inplace_safe=True,
                chunk_size=chunk_size,
                **kernels,
            )
            singles[i] = initial_s[i] + model.linear_no_bias_s(
                model.layernorm_s(singles[i])
            )
        # All per-job updates above allocate before the previous recycle's
        # padded buffers are overwritten. Only final outputs need owned copies.
        s_batch = pad_token_tensors(singles, out=s_batch)
        z_batch = pad_token_tensors(pairs, pair=True, out=z_batch)
        s_batch, z_batch = model.pairformer_stack(
            s=s_batch,
            z=z_batch,
            pair_mask=mask,
            extra_attn_bias=bias,
            inplace_safe=True,
            chunk_size=chunk_size,
            **kernels,
        )
        singles = [s_batch[i, :n] for i, n in enumerate(lengths)]
        pairs = [z_batch[i, :n, :n] for i, n in enumerate(lengths)]
    return [
        (x, s.clone(), z.clone())
        for x, s, z in zip(inputs, singles, pairs, strict=True)
    ]


def _denoise_batch(
    module,
    states,
    noisy,
    noise_levels,
    *,
    chunk_size,
    fusion,
    static_pair=None,
    attention_bias=None,
    pair_bias_cache=None,
    share_sample_conditioning=False,
):
    """Unpadded atom encoders/decoders surrounding one batched token transformer."""
    atoms, conditions, pair_conditions, skips = [], [], [], []
    for state, x, level in zip(states, noisy, noise_levels, strict=True):
        feat, s_input, s, z, cache = state
        r = x / torch.sqrt(module.sigma_data**2 + level**2)[..., None, None]
        single, pair = module.diffusion_conditioning(
            level,
            feat["relp"],
            s_inputs=s_input,
            s_trunk=s,
            z_trunk=z,
            pair_z=cache["pair_z"],
            inplace_safe=False,
            use_conditioning=True,
            single_s=cache.get("single_s"),
        )
        a, q, c, p = module.atom_attention_encoder(
            feat["atom_to_token_idx"],
            feat["ref_pos"],
            feat["ref_charge"],
            feat["ref_mask"],
            feat["ref_atom_name_chars"],
            feat["ref_element"],
            feat["d_lm"],
            feat["v_lm"],
            feat["pad_info"],
            r_l=r,
            s=s.unsqueeze(-3),
            z=pair,
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            conditioned_cache=cache.get("conditioned_atom"),
            transformer_cache=cache.get("atom_encoder"),
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        a = a.float() + module.linear_no_bias_s(module.layernorm_s(single))
        atoms.append(a)
        # sample_diffusion_batch expands one scalar noise level across this
        # job's samples. Only conditioning is shared; atom coordinates remain
        # independent. General denoiser callers keep the default False.
        conditions.append(
            single[:1].float() if share_sample_conditioning else single.float()
        )
        if static_pair is None:
            pair_conditions.append(
                module.normalize(pair.float()) if fusion else pair.float()
            )
        skips.append((q, c, p))
    lengths = [a.shape[-2] for a in atoms]
    a_batch = pad_token_tensors(atoms)
    s_batch = pad_token_tensors(conditions)
    z_batch, bias = static_pair, attention_bias
    if z_batch is None:
        z_batch = pad_token_tensors(pair_conditions, pair=True)
        _, bias = token_masks(
            lengths,
            z_batch,
            [state[0].get("structural_pair_attn_bias") for state in states],
        )
        if fusion:
            z_batch = permute_final_dims(z_batch, [2, 0, 1]).contiguous()
    a_batch = module.diffusion_transformer(
        a=a_batch,
        s=s_batch,
        z=z_batch,
        inplace_safe=True,
        chunk_size=chunk_size,
        enable_efficient_fusion=fusion,
        extra_attn_bias=bias,
        pair_bias_cache=pair_bias_cache,
    )
    del s_batch, z_batch, atoms, conditions, pair_conditions
    result = []
    for i, (state, x, level, skip, length) in enumerate(
        zip(states, noisy, noise_levels, skips, lengths, strict=True)
    ):
        update = module.atom_attention_decoder(
            atom_to_token_idx=state[0]["atom_to_token_idx"],
            a=module.layernorm_a(a_batch[i, :, :length]),
            q_skip=skip[0],
            c_skip=skip[1],
            p_skip=skip[2],
            transformer_cache=state[4].get("atom_decoder"),
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        ratio = (level / module.sigma_data)[..., None, None].to(update.dtype)
        result.append(
            (
                1 / (1 + ratio**2) * x
                + level[..., None, None] / torch.sqrt(1 + ratio**2) * update
            ).to(update.dtype)
        )
    return result


def sample_diffusion_batch(
    states,
    denoise,
    *,
    schedules,
    seeds,
    n_sample,
    chunk_size,
    gamma0,
    gamma_min,
    noise_scale_lambda,
    step_scale_eta,
):
    """Same unguided Euler sampler as B=1, with independent unpadded RNG streams."""
    generators = [
        torch.Generator(device=state[1].device).manual_seed(seed)
        for state, seed in zip(states, seeds, strict=True)
    ]
    numpy_rngs = [np.random.default_rng(seed) for seed in seeds]
    outputs: list[list[torch.Tensor]] = [[] for _ in states]
    chunk_size = chunk_size or n_sample
    # Avoid a scalar CUDA synchronization for every input at every step.
    gammas = [
        [float(gamma0) if value > gamma_min else 0.0 for value in schedule.tolist()[1:]]
        for schedule in schedules
    ]
    for start in range(0, n_sample, chunk_size):
        count = min(chunk_size, n_sample - start)
        coords = [
            schedule[0]
            * torch.randn(
                count,
                state[0]["atom_to_token_idx"].numel(),
                3,
                device=state[1].device,
                dtype=state[1].dtype,
                generator=generator,
            )
            for state, schedule, generator in zip(
                states, schedules, generators, strict=True
            )
        ]
        for step in range(len(schedules[0]) - 1):
            noisy, levels = [], []
            for i, (x, state, schedule, generator, rng) in enumerate(
                zip(coords, states, schedules, generators, numpy_rngs, strict=True)
            ):
                dtype = state[1].dtype
                x = (
                    centre_random_augmentation(
                        x, N_sample=1, torch_generator=generator, numpy_rng=rng
                    )
                    .squeeze(-3)
                    .to(dtype)
                )
                last, current = schedule[step], schedule[step + 1]
                gamma = gammas[i][step]
                level = last * (gamma + 1)
                noisy.append(
                    x
                    + noise_scale_lambda
                    * torch.sqrt(level**2 - last**2)
                    * torch.randn(
                        x.shape, device=x.device, dtype=dtype, generator=generator
                    )
                )
                levels.append(level.reshape(1).expand(count).to(dtype))
            denoised = denoise(noisy, levels)
            if len(denoised) != len(states):
                raise ValueError("Denoiser changed the number of jobs.")
            coords = [
                x
                + step_scale_eta
                * (schedule[step + 1] - level)[..., None, None]
                * ((x - predicted) / level[..., None, None])
                for x, predicted, level, schedule in zip(
                    noisy, denoised, levels, schedules, strict=True
                )
            ]
        for output, x in zip(outputs, coords, strict=True):
            output.append(x)
    return [torch.cat(output, dim=0) for output in outputs]


def forward_batch(model, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate independent complexes with shared batched token-network calls."""
    from opendde.model.opendde import update_input_feature_dict

    cfg = model.configs
    features = [
        update_input_feature_dict(
            model.relative_position_encoding.generate_relp(dict(feat), lazy=True)
        )
        for feat in features
    ]
    n_max = max(
        max(
            feat["token_index"].numel(),
            feat.get("parent_residue_idx", feat["token_index"]).numel(),
        )
        for feat in features
    )
    chunk = model._resolve_pairformer_chunk_size(
        n_max,
        cfg.infer_setting.chunk_size,
        dynamic_chunk_size=cfg.infer_setting.dynamic_chunk_size,
    )
    # Bound score work across jobs as well as token rows.
    budget_chunk = max(1, 450_000_000 // (len(features) * n_max * n_max))
    chunk = min(chunk or budget_chunk, budget_chunk)
    # Runner resolves auto against the actual CUDA runtime. Direct library
    # callers retain a portable Torch fallback for unresolved auto values.
    kernels = {
        key: "torch" if getattr(cfg, key) == "auto" else getattr(cfg, key)
        for key in ("triangle_multiplicative", "triangle_attention")
    }
    residue = _trunk_batch(model, features, chunk, kernels)
    structural = [
        model.expand_to_structural_tokens(
            input_feature_dict=feat,
            s_inputs=trunk[0],
            s=trunk[1],
            z=trunk[2],
            inplace_safe=True,
            chunk_size=chunk,
            lazy_relp=True,
            run_refiner=False,
        )
        for feat, trunk in zip(features, residue, strict=True)
    ]
    if model.enable_structural_token_refiner:
        singles, pairs = pairformer_batch(
            model.structural_token_refiner,
            [x[2] for x in structural],
            [x[3] for x in structural],
            biases=[x[0].get("structural_pair_attn_bias") for x in structural],
            chunk_size=chunk,
            **kernels,
        )
        structural = [
            (x[0], x[1], s, z)
            for x, s, z in zip(structural, singles, pairs, strict=True)
        ]
        del singles, pairs
    pair_branches = [
        model.select_pair_output_branch(
            residue_feature_dict=feat,
            residue_s_inputs=res[0],
            residue_s=res[1],
            residue_z=res[2],
            structural_feature_dict=st[0],
            structural_s_inputs=st[1],
            structural_s=st[2],
            structural_z=st[3],
        )
        for feat, res, st in zip(features, residue, structural, strict=True)
    ]
    del residue
    states, schedules, seeds = [], [], []
    for feat, s_input, s, z in structural:
        cache = model.prepare_diffusion_cache_for_sampling(
            input_feature_dict=feat, z=z, foldcp_mesh=None, diffusion_z_spec=None
        )
        states.append((feat, s_input, s, z, cache))
        schedules.append(
            model.inference_noise_scheduler(
                N_step=cfg.sample_diffusion.N_step,
                device=s_input.device,
                dtype=s_input.dtype,
            )
        )
        seeds.append(int(feat["inference_seed"].item()))
    del structural

    def sample(states, schedules):
        # The serial sampler's AMP wrapper casts its tensor arguments (not
        # tensors inside the feature dictionary). Preserve that same boundary
        # for each ragged member before entering the batched denoiser.
        sample_states = states
        sample_schedules = schedules
        if cfg.skip_amp.sample_diffusion:
            sample_states = []
            for feat, s_input, s, z, cache in states:
                local_cache = dict(cache)
                local_cache["pair_z"] = (
                    None if cache["pair_z"] is None else cache["pair_z"].float()
                )
                local_cache["p_lm/c_l"] = [
                    None if x is None else x.float() for x in cache["p_lm/c_l"]
                ]
                sample_states.append(
                    (feat, s_input.float(), s.float(), z.float(), local_cache)
                )
            sample_schedules = [x.float() for x in schedules]
        module = model.diffusion_module
        static_pair, attention_bias, pair_bias_cache = None, None, None
        if cfg.enable_diffusion_shared_vars_cache:
            budget = InferenceCacheBudget.for_device(
                cfg.infer_setting.diffusion_cache_max_mb,
                sample_states[0][1].device,
            )
            for feat, s_input, s, z, cache in sample_states:
                cache["single_s"] = module.diffusion_conditioning.prepare_single_cache(
                    s_input, s
                )
                if all(value is not None for value in cache["p_lm/c_l"]):
                    cache["conditioned_atom"] = (
                        module.atom_attention_encoder.prepare_conditioned_cache(
                            feat["atom_to_token_idx"],
                            s.unsqueeze(-3),
                            *cache["p_lm/c_l"],
                        )
                    )
            if all(state[4]["pair_z"] is not None for state in sample_states):
                pairs = [state[4]["pair_z"].float() for state in sample_states]
                if cfg.enable_efficient_fusion:
                    pairs = [module.normalize(pair) for pair in pairs]
                static_pair = pad_token_tensors(pairs, pair=True)
                _, attention_bias = token_masks(
                    [pair.shape[-2] for pair in pairs],
                    static_pair,
                    [
                        state[0].get("structural_pair_attn_bias")
                        for state in sample_states
                    ],
                )
                if cfg.enable_efficient_fusion:
                    static_pair = permute_final_dims(
                        static_pair, [2, 0, 1]
                    ).contiguous()
                # Bound resident projected biases; large inputs keep the
                # existing streamed projection path instead of risking OOM.
                transformer = module.diffusion_transformer
                cache_bytes = (
                    len(sample_states)
                    * max(pair.shape[-2] for pair in pairs) ** 2
                    * transformer.n_heads
                    * transformer.n_blocks
                    * 4
                )
                if budget.reserve(cache_bytes):
                    pair_bias_cache = {}
                del pairs
            for *_, cache in sample_states:
                conditioned = cache.get("conditioned_atom")
                if conditioned is not None:
                    p_lm, c_l = conditioned
                    cache["atom_encoder"] = (
                        module.atom_attention_encoder.atom_transformer.prepare_static_cache(
                            c_l, p_lm, budget
                        )
                    )
                    cache["atom_decoder"] = (
                        module.atom_attention_decoder.atom_transformer.prepare_static_cache(
                            c_l, p_lm, budget
                        )
                    )
        graph_enabled = (
            getattr(cfg.infer_setting, "diffusion_cuda_graph", False)
            and not module.training
        )
        if graph_enabled and static_pair is None:
            logger.warning(
                "Diffusion CUDA graph needs prepared static pair conditioning; "
                "using eager denoising for this request."
            )
            graph_enabled = False
        denoise = lambda x, t: _denoise_batch(
            module,
            sample_states,
            x,
            t,
            chunk_size=chunk,
            fusion=cfg.enable_efficient_fusion,
            static_pair=static_pair,
            attention_bias=attention_bias,
            pair_bias_cache=pair_bias_cache,
            share_sample_conditioning=cfg.infer_setting.share_diffusion_sample_conditioning,
        )
        with (
            diffusion_attention_scope(
                cfg.infer_setting.diffusion_attention_backend,
                cfg.enable_diffusion_shared_vars_cache
                and cfg.infer_setting.reuse_diffusion_window_masks,
                reuse_aggregation_plans=graph_enabled,
            ),
            DenoiserGraph(denoise, enabled=graph_enabled) as graph,
        ):
            model.last_diffusion_graph_stats = graph.stats
            return sample_diffusion_batch(
                sample_states,
                graph if graph_enabled else denoise,
                schedules=sample_schedules,
                seeds=seeds,
                n_sample=cfg.sample_diffusion.N_sample,
                chunk_size=cfg.infer_setting.sample_diffusion_chunk_size,
                **{
                    key: getattr(cfg.sample_diffusion, key)
                    for key in (
                        "gamma0",
                        "gamma_min",
                        "noise_scale_lambda",
                        "step_scale_eta",
                    )
                },
            )

    coordinates = autocasting_disable_decorator(cfg.skip_amp.sample_diffusion)(sample)(
        states, schedules
    )
    # Release diffusion caches before confidence, which can retain quadratic
    # per-sample maps. Every subsequent operation owns just one job's metadata.
    structural_features = [state[0] for state in states]
    del states, schedules, cache, s_input, s, z
    predictions = []
    for feat, pair, coordinate in zip(
        structural_features, pair_branches, coordinates, strict=True
    ):
        pair_feat, s_input, s, z = pair
        pred = {"coordinate": coordinate}
        model.run_distogram_contact_stage(pred_dict=pred, pair_z=z, pair_z_spec=None)
        model.run_confidence_head_stage(
            pred_dict=pred,
            input_feature_dict=pair_feat,
            s_inputs=s_input,
            s_trunk=s,
            z_trunk=z,
            z_trunk_spec=None,
            pair_mask=None,
            x_pred_coords=coordinate,
            inplace_safe=False,
            chunk_size=chunk,
            **kernels,
        )
        model.run_post_confidence_outputs_stage(
            pred_dict=pred,
            input_feature_dict=feat,
            pair_input_feature_dict=pair_feat,
            N_cycle=model.N_cycle,
        )
        predictions.append(pred)
    return predictions
