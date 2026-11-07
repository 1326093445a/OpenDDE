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
    monkeypatch,
    *,
    rank=0,
    world_size=1,
    failure=None,
    unique_seeds=False,
    padding_ratio=0.0,
    pair_budget=0,
    batch_size=2,
    job_specs=None,
    structural_sizes=None,
):
    if job_specs is None:
        job_specs = [(5, [101, 102]), (2, [101]), (3, [102])]
    jobs = [
        {
            "name": f"job{i}",
            "modelSeeds": seeds,
            "sequences": [{"proteinChain": {"sequence": "A" * n, "count": 1}}],
        }
        for i, (n, seeds) in enumerate(job_specs)
    ]
    if unique_seeds:
        for i, job in enumerate(jobs):
            job["modelSeeds"] = [101 + i]
    cfg = build_inference_config(
        arg_str=f"--use_msa false --batch_size {batch_size} --sample_diffusion.N_sample 3"
    )
    cfg.seeds = []
    cfg.infer_setting.batch_max_padding_ratio = padding_ratio
    cfg.infer_setting.batch_max_pair_tokens = pair_budget
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
                            "input_feature_dict": (
                                {}
                                if structural_sizes is None
                                else {
                                    "parent_residue_idx": torch.zeros(
                                        structural_sizes[i], dtype=torch.long
                                    )
                                }
                            ),
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


def test_padding_limit_splits_without_losing_job_seeds(monkeypatch):
    calls, written, reports = _run_scheduler(monkeypatch, padding_ratio=1.0)
    assert all(len(call) == 1 for call in calls)
    assert sorted(written) == [
        ("job0", 101),
        ("job0", 102),
        ("job1", 101),
        ("job2", 102),
    ]
    assert not reports


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("padding_ratio", [0.0, 1.0, 1.05])
def test_equal_residue_lengths_can_split_on_structural_lengths(
    monkeypatch, caplog, batch_size, padding_ratio
):
    # Equal 320-residue inputs with different glycine counts can have these
    # structural lengths. Exercise the real scheduler, without a model/GPU.
    structural_sizes = [
        620,
        619,
        618,
        617,
        616,
        615,
        614,
        613,
        612,
        612,
        611,
        611,
        611,
        610,
        610,
        610,
    ]
    caplog.set_level("INFO", logger="runner.batched_prediction")
    calls, written, reports = _run_scheduler(
        monkeypatch,
        batch_size=batch_size,
        padding_ratio=padding_ratio,
        job_specs=[(320, [101])] * 16,
        structural_sizes=structural_sizes,
    )
    expected = (
        [1] * 8 + [2, 3, 3]
        if padding_ratio == 1.0
        else [batch_size] * (16 // batch_size)
    )
    assert [len(call) for call in calls] == expected
    assert written == [(f"job{i}", 101) for i in range(16)]
    assert not reports
    if padding_ratio == 1.0:
        assert "structural padding ratio exceeded" in caplog.text
        assert "candidate residue tokens=[320, 320]" in caplog.text
        assert "structural tokens=[620, 619]" in caplog.text
    else:
        assert "Flushing" not in caplog.text


def test_equal_both_token_spaces_fill_eight_job_batches(monkeypatch):
    calls, written, reports = _run_scheduler(
        monkeypatch,
        batch_size=8,
        padding_ratio=1.0,
        job_specs=[(4, [101])] * 16,
        structural_sizes=[8] * 16,
    )
    assert [len(call) for call in calls] == [8, 8]
    assert len(written) == 16 and not reports


def test_pair_budget_flush_log_distinguishes_the_limit(monkeypatch, caplog):
    caplog.set_level("INFO", logger="runner.batched_prediction")
    calls, written, reports = _run_scheduler(
        monkeypatch,
        batch_size=8,
        pair_budget=2 * 8**2,
        job_specs=[(4, [101])] * 4,
        structural_sizes=[8] * 4,
    )
    assert [len(call) for call in calls] == [2, 2]
    assert "structural pair-token budget exceeded (192 > 128)" in caplog.text
    assert "padding ratio exceeded" not in caplog.text
    assert len(written) == 4 and not reports


def test_precision_flush_log_distinguishes_the_boundary(monkeypatch, caplog):
    # Use tiny fake buckets to exercise the scheduling boundary without
    # allocating confidence maps for thousands of tokens.
    monkeypatch.setattr(batched_prediction, "_precision_bucket", lambda n: n // 3)
    caplog.set_level("INFO", logger="runner.batched_prediction")
    calls, written, reports = _run_scheduler(
        monkeypatch, batch_size=8, job_specs=[(2, [101]), (3, [101])]
    )
    assert [len(call) for call in calls] == [1, 1]
    assert "precision bucket changed (0 -> 1)" in caplog.text
    assert "padding ratio exceeded" not in caplog.text
    assert len(written) == 2 and not reports


@pytest.mark.parametrize(
    "sizes,ratio,budget,expected",
    [
        ([(5, 10)], 1.0, 1, False),
        ([(5, 10), (5, 10)], 1.0, 0, False),
        ([(5, 10), (5, 11)], 1.0, 0, True),
        ([(4, 10), (5, 10)], 1.0, 0, True),
        ([(5, 10), (5, 10)], 0.0, 199, True),
        ([(5, 10), (5, 10)], 0.0, 200, False),
    ],
)
def test_batch_limits_cover_both_token_spaces(sizes, ratio, budget, expected):
    settings = SimpleNamespace(
        batch_max_padding_ratio=ratio, batch_max_pair_tokens=budget
    )
    assert batched_prediction._exceeds_batch_budget(sizes, settings) is expected


@pytest.mark.parametrize("ratio", [-1, 0.5, float("inf"), float("nan")])
def test_invalid_padding_limits_fail_before_model_loading(monkeypatch, ratio):
    from runner.batch_inference import get_default_runner

    monkeypatch.setattr(
        inference.InferenceRunner,
        "__init__",
        lambda *a, **k: pytest.fail("loaded model"),
    )
    with pytest.raises(ValueError):
        get_default_runner(use_msa=False, batch_size=2, batch_max_padding_ratio=ratio)
