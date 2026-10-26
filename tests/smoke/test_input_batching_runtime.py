# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Opt-in released-checkpoint parity test; requires CUDA and cached model assets.

OPENDDE_RUN_BATCH_SMOKE=1 LAYERNORM_TYPE=torch python -m pytest \
    tests/smoke/test_input_batching_runtime.py -q
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENDDE_RUN_BATCH_SMOKE") != "1",
    reason="Set OPENDDE_RUN_BATCH_SMOKE=1 with CUDA and cached inference assets.",
)


def test_released_checkpoint_mixed_lengths_and_confidence(tmp_path):
    from opendde.data.inference.infer_dataloader import InferenceDataset
    from opendde.utils.seed import seed_everything
    from opendde.utils.torch_utils import to_device
    from runner.batch_inference import get_default_runner
    from runner.batched_prediction import validate_prediction
    from runner.inference import _prepare_prediction_batch

    assert torch.cuda.is_available(), "This opt-in test requires CUDA."
    repo = Path(__file__).resolve().parents[2]
    jobs = json.loads((repo / "examples/cofold/no_msa_batch/inputs.json").read_text())[
        :3
    ]
    runner = get_default_runner(
        seeds=[101],
        dump_dir=str(tmp_path),
        use_msa=False,
        use_template=False,
        use_rna_msa=False,
        batch_size=3,
        n_sample=3,
        n_step=20,
        n_cycle=1,
        dtype="fp32",
        device="cuda",
        deterministic=True,
        trimul_kernel="torch",
        triatt_kernel="torch",
    )
    try:
        dataset = InferenceDataset(runner.configs, inputs=jobs)
        data = []
        for i in range(len(jobs)):
            seed_everything(seed=101, deterministic=True)
            item, atom_array, error = dataset[i]
            assert not error, error
            assert int(item["N_atom"]) == len(atom_array)
            _prepare_prediction_batch(runner, runner.configs, item, seed=101)
            data.append(item)
        expected = [
            to_device(
                runner.predict_batch([copy.deepcopy(item)])[0], torch.device("cpu")
            )
            for item in data
        ]
        batches_seen = []
        hook = runner.model.pairformer_stack.register_forward_pre_hook(
            lambda m, args, kw: batches_seen.append(kw["s"].shape[0]), with_kwargs=True
        )
        try:
            actual = runner.predict_batch(copy.deepcopy(data))
        finally:
            hook.remove()
        assert batches_seen == [3]
        for item, observed, wanted in zip(data, actual, expected, strict=True):
            validate_prediction(
                observed,
                n_atom=int(item["N_atom"]),
                n_token=int(item["N_token"]),
                n_sample=3,
            )
            observed = to_device(observed, torch.device("cpu"))
            rmsd = (
                (observed["coordinate"] - wanted["coordinate"])
                .square()
                .sum(-1)
                .mean(-1)
                .sqrt()
            )
            assert (rmsd < 0.01).all(), rmsd
            for key in ("plddt", "pae", "pde", "resolved", "contact_probs"):
                torch.testing.assert_close(
                    observed[key], wanted[key], atol=5e-3, rtol=2e-4
                )
            for obs, want in zip(
                observed["summary_confidence"],
                wanted["summary_confidence"],
                strict=True,
            ):
                for key in ("plddt", "ptm", "iptm", "ranking_score"):
                    torch.testing.assert_close(
                        obs[key], want[key], atol=1e-3, rtol=1e-4
                    )
            for obs, want in zip(
                observed["full_data"], wanted["full_data"], strict=True
            ):
                for key in ("token_asym_id", "atom_to_token_idx", "token_has_frame"):
                    torch.testing.assert_close(obs[key], want[key], atol=0, rtol=0)
    finally:
        runner.close()
