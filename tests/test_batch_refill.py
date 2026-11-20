"""Refill preserves surviving streams, refreshed conditioning, fixed B and ownership."""

import copy
from types import SimpleNamespace

import pytest
import torch

from opendde.model.early_stopping import FirstSampleGate
from opendde.model.opendde import OpenDDE
from opendde.model.refill import forward_refill
from opendde.model.denoiser_graph import DenoiserGraphCaptureError
from tests.test_input_batching import _toy_config, _toy_features
from tests.test_early_stopping import strip_metadata
from scripts.benchmark_inference import compare


@pytest.mark.parametrize(
    "cache,pair_space", [(False, "residue"), (True, "residue"), (True, "structural")]
)
def test_mixed_refill_matches_fixed_batch_baseline(monkeypatch, cache, pair_space):
    torch.manual_seed(42)
    cfg = _toy_config()
    cfg.batch_size = 2
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    cfg.infer_setting.confidence_sample_chunk_size = 1
    cfg.enable_diffusion_shared_vars_cache = cache
    cfg.model.structural_token_expansion.pair_output_space = pair_space
    cfg.need_atom_confidence = True
    model = OpenDDE(cfg).eval()
    features = [_toy_features(4, seed) for seed in range(101, 106)]
    baseline = []
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1 and not p.count_nonzero():
                p.normal_(std=0.02)
        for i in range(0, 5, 2):
            group = features[i : i + 2]
            padded = group + [group[-1]] * (2 - len(group))
            baseline.extend(model.forward_batch(copy.deepcopy(padded))[: len(group)])
        cfg.infer_setting.early_stop_ipsae_threshold = 0.05
        cfg.infer_setting.batch_refill = True
        original = FirstSampleGate.__call__

        def controlled_gate(self, coordinates):
            original(self, coordinates)
            seed = int(self.features[0]["inference_seed"])
            self.stopped = seed in (101, 104, 105)
            self.scores = [0.01 if self.stopped else 0.1]
            return self.stopped

        monkeypatch.setattr(FirstSampleGate, "__call__", controlled_gate)
        result = {}
        batches = [
            [
                (i, copy.deepcopy(features[i]), 101 + i)
                for i in range(start, min(start + 2, 5))
            ]
            for start in range(0, 5, 2)
        ]
        stats = forward_refill(
            model, batches, lambda key, pred: result.__setitem__(key, pred)
        )
    assert stats["completed"] == 5 and stats["stopped"] == 3
    assert stats["sample_rounds"] == 5 and stats["trunk_batches"] == 3
    assert stats["max_ready"] <= 1
    assert set(result) == set(range(5))
    for i, pred in result.items():
        expected = baseline[i]
        if i in (0, 3, 4):
            expected = {
                k: (v[:1] if k != "contact_probs" else v) for k, v in expected.items()
            }
        assert not compare(expected, strip_metadata(pred))
        assert pred["sampling_early_stop"]["batch_size"] == 2
        assert pred["sampling_early_stop"]["batch_policy"] == "preserve_shape_refill"


@pytest.mark.parametrize(
    "change,match",
    [
        ("residue", "equal"),
        ("structural", "equal"),
        ("budget", "batch_max_pair_tokens"),
        ("key", "non-None"),
    ],
)
def test_refill_rejects_shape_and_budget_changes(change, match):
    cfg = _toy_config()
    cfg.batch_size = 2
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    cfg.infer_setting.early_stop_ipsae_threshold = 0.05
    cfg.infer_setting.batch_refill = True
    model = OpenDDE(cfg).eval()
    a, b = _toy_features(4, 101), _toy_features(4, 102)
    key = 1
    if change == "residue":
        b = _toy_features(6, 102)
    if change == "structural":
        b["parent_residue_idx"] = torch.arange(9)
    if change == "budget":
        cfg.infer_setting.batch_max_pair_tokens = 127
    if change == "key":
        key = None
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        forward_refill(model, [[(0, a, 101), (key, b, 102)]], lambda *a: None)


def test_refill_scheduler_failure_ownership_and_shape_groups(monkeypatch):
    from runner.refill_prediction import run_refill_sessions

    cfg = _toy_config()
    cfg.batch_size = 2
    monkeypatch.setattr(
        "runner.inference._prepare_prediction_batch", lambda *a, **kw: None
    )
    monkeypatch.setattr("runner.inference.cleanup_device_memory", lambda *a: None)

    def data(i, n, st):
        return dict(
            sample_name=str(i),
            N_token=torch.tensor(n),
            input_feature_dict={"parent_residue_idx": torch.arange(st)},
        )

    prepared = [
        (data(i, n, st), None, 100 + i)
        for i, (n, st) in enumerate([(4, 8), (4, 8), (4, 8), (4, 9), (6, 12)])
    ]
    calls = []
    published = []
    errors = []

    def predict(batches, emit):
        first = next(batches)
        calls.append([x[1]["sample_name"] for x in first])
        if len(calls) == 1:
            emit(first[0][0], {})
            raise torch.cuda.OutOfMemoryError("injected")
        for key, _, _ in first:
            emit(key, {})
        for batch in batches:
            for key, _, _ in batch:
                emit(key, {})
        return {}

    runner = SimpleNamespace(device=torch.device("cpu"), predict_refill=predict)
    run_refill_sessions(
        runner,
        cfg,
        iter(prepared),
        lambda data, a, s, p: published.append(data["sample_name"]),
        lambda data, s, e: errors.append(data["sample_name"]),
    )
    assert published == ["0", "3", "4"] and errors == ["1", "2"]
    assert calls == [["0", "1"], ["3"], ["4"]]


def test_refill_capture_failure_is_fatal(monkeypatch):
    from runner.refill_prediction import run_refill_sessions

    cfg = _toy_config()
    cfg.batch_size = 2
    monkeypatch.setattr(
        "runner.inference._prepare_prediction_batch", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "runner.inference.cleanup_device_memory",
        lambda *a: pytest.fail("unsafe cleanup"),
    )
    data = dict(sample_name="job", N_token=torch.tensor(4), input_feature_dict={})

    def fail(*a):
        raise DenoiserGraphCaptureError("injected")

    runner = SimpleNamespace(device=torch.device("cpu"), predict_refill=fail)
    with pytest.raises(DenoiserGraphCaptureError):
        run_refill_sessions(
            runner,
            cfg,
            iter([(data, None, 101)]),
            lambda *a: None,
            lambda *a: pytest.fail("swallowed"),
        )


@pytest.mark.parametrize("failure", ["writer", "missing", "unconsumed", "duplicate"])
def test_refill_scheduler_detects_incomplete_or_failed_publication(
    monkeypatch, failure
):
    from runner.refill_prediction import run_refill_sessions

    cfg = _toy_config()
    cfg.batch_size = 2
    monkeypatch.setattr(
        "runner.inference._prepare_prediction_batch", lambda *a, **kw: None
    )
    monkeypatch.setattr("runner.inference.cleanup_device_memory", lambda *a: None)
    jobs = [
        (
            dict(sample_name=str(i), N_token=torch.tensor(4), input_feature_dict={}),
            None,
            101 + i,
        )
        for i in range(3)
    ]
    errors = []
    written = []

    def submit(data, *args):
        if failure == "writer" and data["sample_name"] == "0":
            raise OSError("writer failed")
        written.append(data["sample_name"])

    def predict(batches, emit):
        if failure == "unconsumed":
            return {}
        first = next(batches)
        if failure == "missing":
            return {}
        for key, _, _ in first:
            emit(key, {})
        if failure == "duplicate":
            emit(first[0][0], {})
        for batch in batches:
            for key, _, _ in batch:
                emit(key, {})
        return {}

    runner = SimpleNamespace(device=torch.device("cpu"), predict_refill=predict)
    run_refill_sessions(
        runner,
        cfg,
        iter(jobs),
        submit,
        lambda data, s, e: errors.append(data["sample_name"]),
    )
    if failure == "writer":
        assert errors == ["0"] and written == ["1", "2"]
    elif failure in ("missing", "unconsumed"):
        assert errors == ["0", "1", "2"] and not written
    else:
        assert errors == ["2"] and written == ["0", "1"]
