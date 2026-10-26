# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Exercise job/seed ownership and retries through the real batch scheduler."""

from types import SimpleNamespace

import pytest
import torch

from opendde.config.inference import build_inference_config
from opendde.data.inference.infer_dataloader import InferenceJobSampler
from runner import batched_prediction, inference


def _prediction(n_atom, n_token, seed):
    return {
        "coordinate": torch.full((3, n_atom, 3), float(seed)),
        "plddt": torch.zeros(3, n_atom, 4),
        "resolved": torch.zeros(3, n_atom, 2),
        "pae": torch.zeros(3, n_token, n_token, 4),
        "pde": torch.zeros(3, n_token, n_token, 4),
        "contact_probs": torch.zeros(n_token, n_token),
        "summary_confidence": [{"ranking_score": i} for i in range(3)],
        "full_data": [{"atom_plddt": torch.zeros(n_atom)} for _ in range(3)],
    }


def _run_scheduler(
    monkeypatch, *, rank=0, world_size=1, failure=None, unique_seeds=False
):
    jobs = [
        {
            "name": f"job{i}",
            "modelSeeds": seeds,
            "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1}}],
        }
        for i, (n, seeds) in enumerate([(5, [101, 102]), (2, [101]), (3, [102])])
    ]
    if unique_seeds:
        for i, job in enumerate(jobs):
            job["modelSeeds"] = [101 + i]
    cfg = build_inference_config(
        arg_str="--use_msa false --batch_size 2 --sample_diffusion.N_sample 3"
    )
    cfg.seeds = []
    sampler = InferenceJobSampler(jobs, rank=rank, num_replicas=world_size)

    class Loader:
        def __iter__(self):
            for i in sampler:
                n = len(jobs[i]["sequences"][0]["proteinChain"]["sequence"])
                yield [
                    (
                        {
                            "sample_name": f"job{i}",
                            "sample_index": i,
                            "N_token": torch.tensor(n),
                            "N_atom": torch.tensor(n * 4),
                            "N_asym": torch.tensor(1),
                            "N_msa": torch.tensor(1),
                            "input_feature_dict": {},
                            "entity_poly_type": {"1": "polypeptide(L)"},
                        },
                        [f"job{i}"] * (n * 4),
                        "",
                    )
                ]

    loader = Loader()
    loader.sampler = sampler
    monkeypatch.setattr(inference, "_load_inference_jobs_synchronized", lambda _: jobs)
    monkeypatch.setattr(
        batched_prediction, "get_inference_dataloader", lambda *a, **kw: loader
    )
    monkeypatch.setattr(batched_prediction, "seed_everything", lambda **kw: None)
    monkeypatch.setattr(inference, "cleanup_device_memory", lambda *a, **kw: None)
    reports, calls, written = [], [], []
    monkeypatch.setattr(
        inference, "_append_error_report", lambda *args: reports.append(args)
    )

    def predict(data):
        identities = [
            (d["sample_name"], int(d["input_feature_dict"]["inference_seed"]))
            for d in data
        ]
        calls.append(identities)
        if failure == "oom" and len(data) > 1:
            raise torch.cuda.OutOfMemoryError("simulated forward OOM")
        result = [
            _prediction(int(d["N_atom"]), int(d["N_token"]), seed)
            for d, (_, seed) in zip(data, identities, strict=True)
        ]
        return result[:-1] if failure == "missing" else result

    def dump(**kw):
        assert set(kw["atom_array"]) == {kw["pdb_id"]}
        assert kw["pred_dict"]["coordinate"].eq(kw["seed"]).all()
        if failure == "writer" and kw["pdb_id"] == "job1":
            raise OSError("simulated writer failure")
        written.append((kw["pdb_id"], kw["seed"]))

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir="unused",
        update_model_configs=lambda c: None,
        predict_batch=predict,
        dumper=SimpleNamespace(dump=dump),
    )
    if failure in {"missing", "writer"}:
        with pytest.raises(RuntimeError, match="inference job/seed"):
            batched_prediction.infer_predict_batched(runner, cfg)
    else:
        batched_prediction.infer_predict_batched(runner, cfg)
    return calls, written, reports


def test_scheduler_preserves_per_job_seeds_and_metadata(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch)
    assert calls == [[("job1", 101), ("job0", 101)], [("job2", 102), ("job0", 102)]]
    assert sorted(written) == [
        ("job0", 101),
        ("job0", 102),
        ("job1", 101),
        ("job2", 102),
    ]
    assert not reports


def test_scheduler_batches_different_seeds_and_flushes_partial_batch(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch, unique_seeds=True)
    assert calls == [[("job0", 101), ("job1", 102)], [("job2", 103)]]
    assert written == [("job0", 101), ("job1", 102), ("job2", 103)]
    assert not reports


def test_scheduler_oom_retries_halves_without_duplicate_outputs(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch, failure="oom")
    assert [len(call) for call in calls] == [2, 1, 1, 2, 1, 1]
    assert len(written) == len(set(written)) == 4
    assert not reports


def test_writer_failure_never_repeats_forward(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch, failure="writer")
    assert len(calls) == 2
    assert len(written) == 3
    assert len(reports) == 1


def test_missing_prediction_does_not_write_wrong_job(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch, failure="missing")
    assert len(calls) == 2
    assert not written
    assert len(reports) == 4


def test_scheduler_shards_jobs_and_skips_padding_on_empty_ranks(monkeypatch):
    all_written = []
    for rank in range(3):
        calls, written, reports = _run_scheduler(monkeypatch, rank=rank, world_size=3)
        assert all(1 <= len(call) <= 2 for call in calls)
        assert not reports
        if rank == 2:
            assert not calls and not written
        all_written.extend(written)
    assert sorted(all_written) == [
        ("job0", 101),
        ("job0", 102),
        ("job1", 101),
        ("job2", 102),
    ]
