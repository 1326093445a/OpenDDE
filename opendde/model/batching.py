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
import torch.nn.functional as F

from opendde.model.utils import centre_random_augmentation, permute_final_dims
from opendde.utils.torch_utils import autocasting_disable_decorator


def pad_token_tensors(
    values: list[torch.Tensor], *, pair: bool = False
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
    return torch.stack(
        [
            F.pad(
                value,
                (0, 0, 0, size - value.shape[-2])
                + ((0, size - value.shape[-3]) if pair else ()),
            )
            for value in values
        ]
    )


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
        singles, pairs = pairformer_batch(
            model.pairformer_stack, singles, pairs, chunk_size=chunk_size, **kernels
        )
    return list(zip(inputs, singles, pairs, strict=True))


def _denoise_batch(module, states, noisy, noise_levels, *, chunk_size, fusion):
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
            inplace_safe=True,
            use_conditioning=True,
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
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        a = a.float() + module.linear_no_bias_s(module.layernorm_s(single))
        atoms.append(a)
        conditions.append(single.float())
        pair_conditions.append(
            module.normalize(pair.float()) if fusion else pair.float()
        )
        skips.append((q, c, p))
    lengths = [a.shape[-2] for a in atoms]
    a_batch = pad_token_tensors(atoms)
    s_batch = pad_token_tensors(conditions)
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
                gamma = float(gamma0) if current > gamma_min else 0
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
    # The library entry point may not have passed through Runner's automatic
    # kernel resolution. Input batching uses the validated PyTorch path.
    kernels = dict(triangle_multiplicative="torch", triangle_attention="torch")
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
        return sample_diffusion_batch(
            sample_states,
            lambda x, t: _denoise_batch(
                model.diffusion_module,
                sample_states,
                x,
                t,
                chunk_size=chunk,
                fusion=cfg.enable_efficient_fusion,
            ),
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
