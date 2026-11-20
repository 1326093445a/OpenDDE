"""Screening is optional; retained samples, confidence and RNG streams stay owned."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from opendde.config.inference import build_inference_config, validate_input_batching
from opendde.model.early_stopping import FirstSampleGate
from opendde.model.opendde import OpenDDE
from runner.batched_prediction import validate_prediction
from tests.test_input_batching import _toy_config, _toy_features
from scripts.benchmark_inference import compare


@pytest.mark.parametrize(
    "arg",
    [
        "--infer_setting.early_stop_ipsae_threshold nan",
        "--infer_setting.early_stop_ipsae_threshold -0.1",
        "--infer_setting.early_stop_ipsae_threshold 1.1",
        "--infer_setting.batch_refill true",
    ],
)
def test_invalid_screening_config(arg):
    with pytest.raises(ValueError):
        cfg = build_inference_config(arg_str=arg)
        validate_input_batching(cfg)


def test_screening_requires_explicit_shape_compatible_settings():
    cfg = build_inference_config(arg_str="--use_msa false")
    assert cfg.infer_setting.early_stop_ipsae_threshold is None
    assert not cfg.infer_setting.batch_refill
    cfg.infer_setting.early_stop_ipsae_threshold = 0.05
    with pytest.raises(ValueError, match="sample_diffusion_chunk_size"):
        validate_input_batching(cfg)
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    cfg.infer_setting.confidence_sample_chunk_size = 1
    validate_input_batching(cfg)
    for key, value in (
        ("use_msa", True),
        ("use_template", True),
        ("foldcp_mode", "distributed"),
    ):
        bad = copy.deepcopy(cfg)
        setattr(bad, key, value)
        with pytest.raises(ValueError):
            validate_input_batching(bad)
    cfg.infer_setting.batch_refill = True
    with pytest.raises(ValueError, match="batch_size"):
        validate_input_batching(cfg)
    cfg.batch_size = 2
    validate_input_batching(cfg)
    cfg.sample_diffusion.N_sample = 1
    with pytest.raises(ValueError, match="N_sample"):
        validate_input_batching(cfg)


@pytest.mark.parametrize(
    "scores,threshold,stopped",
    [
        ([0.0], 0.0, True),
        ([0.05], 0.05, False),
        ([0.04, 0.02], 0.05, True),
        ([0.01, 0.1], 0.05, False),
        ([None, 0.0], 0.05, False),
        ([float("nan")], 0.05, False),
        ([float("inf")], 0.05, False),
    ],
)
def test_cutoff_and_undefined_scores(monkeypatch, scores, threshold, stopped):
    cfg = _toy_config()
    gate = FirstSampleGate(
        SimpleNamespace(configs=cfg),
        [{}] * len(scores),
        [(None, torch.zeros(1))] * len(scores),
        None,
        {},
        threshold,
    )
    monkeypatch.setattr(
        gate,
        "heads",
        lambda i, x: {
            "pae": x,
            "pde": x,
            "summary_confidence": [{"ipsae_min_10": scores[i]}],
        },
    )
    monkeypatch.setattr(
        "opendde.model.batching.summarize_prediction", lambda m, p, f, b: p
    )
    assert gate([torch.zeros(1)] * len(scores)) is stopped


def strip_metadata(pred):
    pred = copy.deepcopy(pred)
    pred.pop("sampling_early_stop", None)
    for item in pred["summary_confidence"]:
        item.pop("sampling_early_stop", None)
        item.pop("sample_generation_index", None)
    return pred


@pytest.mark.parametrize(
    "cache,pair_space,full",
    [(False, "residue", True), (True, "residue", False), (True, "structural", True)],
)
def test_continue_preserves_all_native_outputs(cache, pair_space, full):
    torch.manual_seed(42)
    cfg = _toy_config()
    cfg.enable_diffusion_shared_vars_cache = cache
    cfg.need_atom_confidence = full
    cfg.model.structural_token_expansion.pair_output_space = pair_space
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    cfg.infer_setting.confidence_sample_chunk_size = 1
    model = OpenDDE(cfg).eval()
    features = [_toy_features(4, 101), _toy_features(6, 102)]
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1 and not p.count_nonzero():
                p.normal_(std=0.02)
        baseline = model.forward_batch(copy.deepcopy(features))
        # Toy atom names do not identify CAs, so native ipSAE is undefined.
        cfg.infer_setting.early_stop_ipsae_threshold = 1.0
        screened = model.forward_batch(copy.deepcopy(features))
    for a, b in zip(baseline, screened, strict=True):
        assert b["sampling_early_stop"]["reason"] == "ipsae_unavailable"
        assert not compare(a, strip_metadata(b))
        validate_prediction(
            b,
            n_atom=a["coordinate"].shape[1],
            n_token=a["pae"].shape[1],
            n_sample=3,
            early_stop_threshold=1.0,
        )


def test_validator_rejects_unapproved_shortfall():
    from tests.test_batched_prediction import _prediction

    pred = _prediction(4, 2, 101)
    for key in (
        "coordinate",
        "plddt",
        "resolved",
        "pae",
        "pde",
        "summary_confidence",
        "full_data",
    ):
        pred[key] = pred[key][:1]
    info = dict(
        stopped=True,
        metric="ipsae_min_10",
        reason="first_sample_below_threshold",
        threshold=0.05,
        requested_samples=3,
        completed_samples=1,
        skipped_samples=2,
        first_sample_score=0.01,
    )
    pred["sampling_early_stop"] = info
    validate_prediction(
        pred, n_atom=4, n_token=2, n_sample=3, early_stop_threshold=0.05
    )
    for changes in (
        {"threshold": 0.1},
        {"stopped": False},
        {"first_sample_score": None},
        {"first_sample_score": float("nan")},
        {"requested_samples": 5},
    ):
        with pytest.raises(ValueError, match="shortfall"):
            validate_prediction(
                pred | {"sampling_early_stop": info | changes},
                n_atom=4,
                n_token=2,
                n_sample=3,
                early_stop_threshold=0.05,
            )
    with pytest.raises(ValueError):
        validate_prediction(pred, n_atom=4, n_token=2, n_sample=3)


def test_cli_passes_screening_options(monkeypatch, tmp_path):
    from runner import batch_inference as cli

    calls = []
    path = tmp_path / "input.json"
    path.write_text("[]")
    monkeypatch.setattr(cli, "inference_jsons", lambda *a, **kw: calls.append(kw))
    result = CliRunner().invoke(
        cli.predict,
        [
            "-i",
            str(path),
            "-o",
            str(tmp_path / "out"),
            "--use_msa",
            "false",
            "--batch_size",
            "2",
            "--sample",
            "5",
            "--sample_diffusion_chunk_size",
            "1",
            "--confidence_sample_chunk_size",
            "1",
            "--early_stop_ipsae_threshold",
            "0.05",
            "--batch_refill",
            "true",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["batch_refill"] and calls[0]["early_stop_ipsae_threshold"] == 0.05
    assert calls[0]["sample_diffusion_chunk_size"] == 1


def test_completion_marker_published_with_outputs(tmp_path, monkeypatch):
    from runner.dumper import DataDumper
    from tests.test_dumper_reliability import (
        _patch_minimal_writers,
        _minimal_prediction,
    )

    dumper = DataDumper(str(tmp_path))
    _patch_minimal_writers(monkeypatch, dumper)
    pred = _minimal_prediction()
    pred["sampling_early_stop"] = {
        "requested_samples": 5,
        "completed_samples": 1,
        "skipped_samples": 4,
    }
    dumper.dump_predictions(pred, str(tmp_path / "job"), "job", None, {}, 101)
    dest = tmp_path / "job" / "predictions"
    assert (
        json.loads((dest / "sampling_status.json").read_text())["status"] == "complete"
    )
    _patch_minimal_writers(monkeypatch, dumper, fail_confidence=True)
    with pytest.raises(OSError):
        dumper.dump_predictions(pred, str(tmp_path / "job"), "job", None, {}, 101)
    assert (
        json.loads((dest / "sampling_status.json").read_text())["completed_samples"]
        == 1
    )
    assert not list((tmp_path / "job").glob(".predictions-staging-*"))


def test_screening_routes_singleton_to_native_scheduler(monkeypatch):
    from runner import inference, batched_prediction

    cfg = build_inference_config(
        arg_str="--use_msa false --infer_setting.sample_diffusion_chunk_size 1 --infer_setting.confidence_sample_chunk_size 1 --infer_setting.early_stop_ipsae_threshold 0.05"
    )
    calls = []
    monkeypatch.setattr(
        batched_prediction, "infer_predict_batched", lambda r, c: calls.append(c)
    )
    inference.infer_predict(SimpleNamespace(device=torch.device("cpu")), cfg)
    assert calls == [cfg]


def test_python_runner_and_dotted_config_match_screening_options(monkeypatch):
    from runner import batch_inference

    monkeypatch.setattr(
        batch_inference,
        "InferenceRunner",
        lambda cfg, **kw: SimpleNamespace(configs=cfg),
    )
    cfg = batch_inference.get_default_runner(
        use_msa=False,
        batch_size=2,
        n_sample=3,
        sample_diffusion_chunk_size=1,
        confidence_sample_chunk_size=1,
        early_stop_ipsae_threshold=0.05,
        batch_refill=True,
    ).configs
    dotted = build_inference_config(
        arg_str="--use_msa false --batch_size 2 --sample_diffusion.N_sample 3 --infer_setting.sample_diffusion_chunk_size 1 --infer_setting.confidence_sample_chunk_size 1 --infer_setting.early_stop_ipsae_threshold 0.05 --infer_setting.batch_refill true"
    )
    for key in (
        "early_stop_ipsae_threshold",
        "batch_refill",
        "sample_diffusion_chunk_size",
        "confidence_sample_chunk_size",
    ):
        assert cfg.infer_setting[key] == dotted.infer_setting[key]
    validate_input_batching(cfg)


from tests.test_ipsae_interface_inference import dataset  # noqa: E402,F401


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_screening_uses_native_grouped_interface(dataset, selected, full):
    data, atoms, error = dataset[0]
    assert not error, error
    feat = data["input_feature_dict"]
    if not selected:
        feat.pop("ipsae_interface")
    cfg = dataset.configs
    cfg.need_atom_confidence = full
    cfg.infer_setting.sample_diffusion_chunk_size = 1
    model = OpenDDE(cfg).eval()
    with torch.no_grad():
        base = model.forward_batch([copy.deepcopy(feat)])[0]
        cfg.infer_setting.early_stop_ipsae_threshold = 1.0
        pred = model.forward_batch([copy.deepcopy(feat)])[0]
    expected_score = base["summary_confidence"][0]["ipsae_min_10"]
    assert pred["sampling_early_stop"]["first_sample_score"] == expected_score
    assert (expected_score is not None) == selected
    if selected:
        assert pred["coordinate"].shape[0] == 1
        base = {k: (v if k == "contact_probs" else v[:1]) for k, v in base.items()}
    else:
        assert pred["sampling_early_stop"]["reason"] == "ipsae_unavailable"
        assert pred["coordinate"].shape[0] == 3
    assert not compare(base, strip_metadata(pred))
