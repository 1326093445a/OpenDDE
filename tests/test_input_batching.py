# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Input batches must never turn jobs into samples or change atom topology."""

import copy

import pytest
import torch

from opendde.config.inference import build_inference_config, validate_input_batching
from opendde.model.batching import (
    pad_token_tensors,
    pairformer_batch,
    sample_diffusion_batch,
)
from opendde.model.generator import InferenceNoiseScheduler, sample_diffusion
from opendde.model.modules.pairformer import PairformerStack


def test_padding_preserves_sample_and_channel_axes_when_sizes_coincide():
    values = [torch.arange(3 * n * n * 3).reshape(3, n, n, 3) for n in (2, 3)]
    padded = pad_token_tensors(values, pair=True)
    assert padded.shape == (2, 3, 3, 3, 3)
    for i, value in enumerate(values):
        n = value.shape[-2]
        assert torch.equal(padded[i, :, :n, :n], value)
    assert not padded[0, :, 2:].any()
    assert not padded[0, :, :, 2:].any()


@pytest.mark.parametrize("lengths", [(2, 5), (3, 3), (1, 7)])
def test_pairformer_batch_matches_independent_jobs(lengths):
    torch.manual_seed(7)
    model = PairformerStack(
        n_blocks=2, c_s=8, c_z=8, n_heads=2, hidden_scale_up=False
    ).eval()
    with torch.no_grad():
        # Activate normally zero-initialized residual outputs: zero updates
        # would make a padding test pass without exercising attention.
        for param in model.parameters():
            if param.ndim > 1 and not param.count_nonzero():
                param.normal_(std=0.05)
        singles = [torch.randn(n, 8) for n in lengths]
        pairs = [torch.randn(n, n, 8) for n in lengths]
        biases = [torch.randn(n, n) for n in lengths]
        expected = [
            model(
                s=s.clone(),
                z=z.clone(),
                pair_mask=None,
                extra_attn_bias=bias,
                inplace_safe=True,
                triangle_multiplicative="torch",
                triangle_attention="torch",
            )
            for s, z, bias in zip(singles, pairs, biases, strict=True)
        ]
        s_batch, z_batch = pairformer_batch(
            model,
            singles,
            pairs,
            biases=biases,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            chunk_size=1,
        )
    for s, z, (want_s, want_z) in zip(s_batch, z_batch, expected, strict=True):
        torch.testing.assert_close(s, want_s, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(z, want_z, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("chunk", [None, 1, 2, 5])
def test_ragged_sampler_preserves_each_jobs_random_stream(chunk):
    states = [
        ({"atom_to_token_idx": torch.zeros(n, dtype=torch.long)}, torch.zeros(2, 4))
        for n in (3, 7, 3)
    ]
    seeds = [101, 102, 103]
    schedules = [InferenceNoiseScheduler()(N_step=3) for _ in states]
    options = dict(
        gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5
    )
    expected = []
    for state, seed, schedule in zip(states, seeds, schedules, strict=True):
        expected.append(
            sample_diffusion(
                denoise_net=lambda *, x_noisy, **kw: x_noisy * 0.9,
                input_feature_dict=state[0],
                s_inputs=state[1],
                s_trunk=state[1],
                z_trunk=torch.zeros(2, 2, 4),
                pair_z=None,
                p_lm=None,
                c_l=None,
                noise_schedule=schedule,
                N_sample=3,
                diffusion_chunk_size=chunk,
                rollout_seed=seed,
                **options,
            )
        )
    actual = sample_diffusion_batch(
        states,
        lambda xs, ts: [x * 0.9 for x in xs],
        schedules=schedules,
        seeds=seeds,
        n_sample=3,
        chunk_size=chunk,
        **options,
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert torch.equal(observed, wanted)
    reordered = sample_diffusion_batch(
        states[::-1],
        lambda xs, ts: [x * 0.9 for x in xs],
        schedules=schedules[::-1],
        seeds=seeds[::-1],
        n_sample=3,
        chunk_size=chunk,
        **options,
    )
    for observed, wanted in zip(reordered, expected[::-1], strict=True):
        assert torch.equal(observed, wanted)


def test_batch_validation_keeps_single_job_compatible_and_rejects_unsupported_modes():
    cfg = build_inference_config()
    validate_input_batching(cfg)
    cfg.batch_size = 2
    with pytest.raises(ValueError, match="use_msa"):
        validate_input_batching(cfg)
    cfg.use_msa = False
    validate_input_batching(cfg)
    for key, value in (
        ("use_template", True),
        ("num_workers", 1),
        ("foldcp_mode", "distributed"),
    ):
        changed = copy.deepcopy(cfg)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            validate_input_batching(changed)


def test_prediction_boundary_rejects_sample_and_job_axis_confusion():
    from runner.batched_prediction import validate_prediction

    pred = {
        "coordinate": torch.zeros(3, 5, 3),
        "plddt": torch.zeros(3, 5, 4),
        "resolved": torch.zeros(3, 5, 2),
        "pae": torch.zeros(3, 2, 2, 4),
        "pde": torch.zeros(3, 2, 2, 4),
        "contact_probs": torch.zeros(2, 2),
        "summary_confidence": [{}, {}, {}],
        "full_data": [{}, {}, {}],
    }
    validate_prediction(pred, n_atom=5, n_token=2, n_sample=3)
    for field, value in (
        ("coordinate", torch.zeros(2, 3, 5, 3)),
        ("pae", torch.zeros(3, 3, 3, 4)),
        ("full_data", [{}, {}]),
    ):
        with pytest.raises(ValueError):
            validate_prediction(pred | {field: value}, n_atom=5, n_token=2, n_sample=3)


def _toy_features(n_token, seed):
    """Two chains, BB/SC structural tokens, four atoms/residue; no CCD assets."""
    from opendde.data.utils import make_dummy_feature

    g = torch.Generator().manual_seed(seed)
    residue = torch.arange(n_token)
    atom_parent = residue.repeat_interleave(4)
    parent = residue.repeat_interleave(2)
    atom_slot = torch.arange(4).repeat(n_token)
    asym = (residue >= n_token // 2).long()
    rep = atom_slot == 1
    frame = residue[:, None] * 4 + torch.arange(3)[None, :]
    feat = {
        "token_index": residue,
        "residue_index": residue,
        "asym_id": asym,
        "entity_id": asym,
        "sym_id": torch.zeros_like(residue),
        "restype": torch.nn.functional.one_hot(residue % 20, 32).float(),
        "token_bonds": torch.zeros(n_token, n_token),
        "ref_pos": torch.randn(n_token * 4, 3, generator=g),
        "ref_charge": torch.zeros(n_token * 4),
        "ref_mask": torch.ones(n_token * 4),
        "ref_space_uid": atom_parent,
        "ref_element": torch.nn.functional.one_hot(torch.full((n_token * 4,), 6), 128),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.zeros(n_token * 4, 4, dtype=torch.long), 64
        ),
        "atom_to_token_idx": atom_parent,
        "atom_to_tokatom_idx": atom_slot,
        "has_frame": torch.ones(n_token, dtype=torch.long),
        "frame_atom_index": frame,
        "distogram_rep_atom_mask": rep,
        "pae_rep_atom_mask": rep,
        "is_protein": torch.ones(n_token * 4, dtype=torch.long),
        "is_ligand": torch.zeros(n_token * 4, dtype=torch.long),
        "is_dna": torch.zeros(n_token * 4, dtype=torch.long),
        "is_rna": torch.zeros(n_token * 4, dtype=torch.long),
        "parent_residue_idx": parent,
        "structural_token_index": torch.arange(n_token * 2),
        "subtoken_role_id": torch.tensor([1, 2]).repeat(n_token),
        "atom_to_structural_token_idx": atom_parent * 2 + (atom_slot == 3).long(),
        "atom_to_structural_tokatom_idx": atom_slot % 3,
        "structural_distogram_rep_atom_mask": rep | (atom_slot == 3),
        "structural_pae_rep_atom_mask": rep | (atom_slot == 3),
        "structural_has_frame": torch.ones(n_token * 2, dtype=torch.long),
        "structural_frame_atom_index": frame.repeat_interleave(2, dim=0),
        "structural_polymer_type": torch.ones(n_token * 2, dtype=torch.long),
        "prev_parent_residue_idx": torch.where(
            (parent > 0) & (parent != n_token // 2), parent - 1, -1
        ),
        "next_parent_residue_idx": torch.where(
            (parent < n_token - 1) & (parent != n_token // 2 - 1), parent + 1, -1
        ),
        "inference_seed": torch.tensor(seed),
    }
    return make_dummy_feature(feat, dummy_feats=("msa", "template"))


def _toy_config():
    return build_inference_config(
        arg_str=" ".join(
            [
                "--use_msa false --use_template false --use_rna_msa false --dtype fp32",
                "--triangle_multiplicative torch --triangle_attention torch",
                "--c_s 16 --c_z 16 --c_token 16 --c_atom 16 --c_atompair 8 --c_s_inputs 81",
                "--n_blocks 1 --hidden_scale_up false --model.N_cycle 2",
                "--model.msa_module.c_m 16 --model.msa_module.n_blocks 1",
                "--model.template_embedder.n_blocks 1 --model.template_embedder.c 16",
                "--model.structural_token_expansion.structural_refiner.n_blocks 1",
                "--model.structural_token_expansion.structural_refiner.hidden_scale_up false",
                "--model.diffusion_module.c_token 32 --model.diffusion_module.c_z_pair_diffusion 16",
                "--model.diffusion_module.atom_encoder.n_blocks 1",
                "--model.diffusion_module.transformer.n_blocks 1",
                "--model.diffusion_module.transformer.n_heads 4",
                "--model.diffusion_module.atom_decoder.n_blocks 1 --model.confidence_head.n_blocks 1",
                "--sample_diffusion.N_sample 3 --sample_diffusion.N_step 4",
            ]
        )
    )


@pytest.mark.parametrize(
    "fusion,cache,pair_space",
    [(False, False, "residue"), (True, True, "residue"), (True, True, "structural")],
)
def test_full_model_batch_keeps_predictions_and_confidence_per_job(
    fusion, cache, pair_space
):
    from opendde.model.opendde import OpenDDE
    from runner.batched_prediction import validate_prediction

    torch.manual_seed(42)
    cfg = _toy_config()
    cfg.enable_efficient_fusion = fusion
    cfg.enable_diffusion_shared_vars_cache = cache
    cfg.model.structural_token_expansion.pair_output_space = pair_space
    cfg.infer_setting.sample_diffusion_chunk_size = 2
    model = OpenDDE(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim > 1 and not param.count_nonzero():
                param.normal_(std=0.02)
        feats = [_toy_features(3, 101), _toy_features(5, 102), _toy_features(5, 103)]
        expected = [model(input_feature_dict=copy.deepcopy(f))[0] for f in feats]
        # Prove this is a batched neural forward, not a loop over whole jobs.
        trunk_batch_sizes, diffusion_batch_sizes = [], []
        trunk_hook = model.pairformer_stack.register_forward_pre_hook(
            lambda m, a, kw: trunk_batch_sizes.append(kw["s"].shape[0]),
            with_kwargs=True,
        )
        diffusion_hook = (
            model.diffusion_module.diffusion_transformer.register_forward_pre_hook(
                lambda m, a, kw: diffusion_batch_sizes.append(kw["a"].shape[0]),
                with_kwargs=True,
            )
        )
        actual = model.forward_batch(copy.deepcopy(feats))
        trunk_hook.remove()
        diffusion_hook.remove()
    assert trunk_batch_sizes == [3, 3]
    assert diffusion_batch_sizes and set(diffusion_batch_sizes) == {3}
    for observed, wanted, feat in zip(actual, expected, feats, strict=True):
        validate_prediction(
            observed,
            n_atom=feat["atom_to_token_idx"].numel(),
            n_token=feat["token_index"].numel(),
            n_sample=3,
        )
        for key in ("coordinate", "plddt", "pae", "pde", "resolved", "contact_probs"):
            torch.testing.assert_close(observed[key], wanted[key], atol=2e-4, rtol=2e-4)
        for obs, want in zip(
            observed["summary_confidence"], wanted["summary_confidence"], strict=True
        ):
            for key in ("plddt", "ptm", "iptm", "ranking_score"):
                torch.testing.assert_close(obs[key], want[key], atol=2e-4, rtol=2e-4)


def test_triangle_chunking_keeps_broadcast_bias_unexpanded(monkeypatch):
    from opendde.model.triangular import triangular

    torch.manual_seed(51)
    layer = triangular.TriangleAttention(8, 4, 2).eval()
    with torch.no_grad():
        for param in layer.parameters():
            if param.ndim > 1 and not param.count_nonzero():
                param.normal_(std=0.05)
        x = torch.randn(2, 7, 7, 8)
        expected = layer(x, chunk_size=None)
        shapes = []
        hook = layer.mha.register_forward_pre_hook(
            lambda m, args, kw: shapes.append((kw["q_x"].shape, kw["biases"][1].shape)),
            with_kwargs=True,
        )

        def forbid_flattening(*args, **kwargs):
            raise AssertionError(
                "Batched triangle bias must not pass through expand/flatten."
            )

        monkeypatch.setattr(triangular, "chunk_layer", forbid_flattening)
        actual = layer(x, chunk_size=3)
        hook.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert [shape[0][-3] for shape in shapes] == [3, 3, 1]
    assert all(shape[1] == (2, 1, 2, 7, 7) for shape in shapes)
