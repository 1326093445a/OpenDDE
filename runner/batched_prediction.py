# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Per-rank ragged prediction scheduling and per-job output validation."""

from __future__ import annotations

import time
from typing import Any

import torch

from opendde.config.inference import validate_input_batching
from opendde.data.inference.infer_dataloader import get_inference_dataloader
from opendde.utils.logger import get_logger
from opendde.utils.seed import seed_everything

logger = get_logger(__name__)


def validate_prediction(
    prediction: dict[str, Any], *, n_atom: int, n_token: int, n_sample: int
) -> None:
    """Validate explicit output axes before any file for this job is published."""
    coordinate = prediction["coordinate"]
    if tuple(coordinate.shape) != (n_sample, n_atom, 3):
        raise ValueError(
            f"Unexpected per-job coordinate shape: {tuple(coordinate.shape)}"
        )
    if not torch.isfinite(coordinate).all():
        raise ValueError("Prediction coordinates contain NaN or Inf.")
    for key in ("summary_confidence", "full_data"):
        if len(prediction[key]) != n_sample:
            raise ValueError(f"{key} does not contain one entry per sample.")
    for key in ("pae", "pde"):
        value = prediction[key]
        if value.ndim != 4 or tuple(value.shape[:3]) != (n_sample, n_token, n_token):
            raise ValueError(f"{key} contains the wrong per-job residue axes.")
    for key in ("plddt", "resolved"):
        value = prediction[key]
        if value.ndim != 3 or tuple(value.shape[:2]) != (n_sample, n_atom):
            raise ValueError(f"{key} contains the wrong per-job atom axis.")
    if tuple(prediction["contact_probs"].shape) != (n_token, n_token):
        raise ValueError(
            "Contact probabilities contain the wrong per-job residue axes."
        )
    for full in prediction["full_data"]:
        for key, shape in (
            ("atom_plddt", (n_atom,)),
            ("token_pair_pae", (n_token, n_token)),
            ("token_pair_pde", (n_token, n_token)),
            ("token_has_frame", (n_token,)),
            ("token_asym_id", (n_token,)),
            ("atom_to_token_idx", (n_atom,)),
            ("atom_coordinate", (n_atom, 3)),
            ("atom_is_polymer", (n_atom,)),
        ):
            if key in full and tuple(full[key].shape) != shape:
                raise ValueError(f"Full confidence {key} has the wrong per-job axes.")


def _precision_bucket(n_token: int) -> int:
    # Match update_inference_configs: never let batch membership silently
    # switch an individual job between its normal AMP policies.
    return 0 if n_token <= 2560 else 1 if n_token <= 3840 else 2


def infer_predict_batched(runner, configs) -> None:
    from runner.inference import (
        _append_error_report,
        _distributed_rank,
        _load_inference_jobs_synchronized,
        _prepare_inference_batch,
        _prepare_prediction_batch,
        _resolve_job_seed_schedule,
        _sampler_owns_inference_sample,
        cleanup_device_memory,
    )

    validate_input_batching(configs)
    jobs = _load_inference_jobs_synchronized(configs.input_json_path)
    schedule = _resolve_job_seed_schedule(jobs, list(configs.seeds) or None)
    loader = get_inference_dataloader(configs, inputs=jobs)
    errors: list[str] = []

    def report(data, seed, exc):
        message = f"{data['sample_name']} [seed:{seed}]: {type(exc).__name__}: {exc}"
        errors.append(message)
        logger.error(message)
        _append_error_report(runner.error_dir, f"{data['sample_name']}.txt", message)

    def predict(items):
        start = time.monotonic()
        predictions = None
        retry = False
        try:
            for data, _, seed in items:
                _prepare_prediction_batch(runner, configs, data, seed=seed)
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            predictions = runner.predict_batch([data for data, _, _ in items])
            if len(predictions) != len(items):
                raise ValueError(
                    "Model returned a different number of jobs than requested."
                )
        except torch.cuda.OutOfMemoryError as exc:
            # Drop traceback-held CUDA intermediates before retrying. Only
            # forward OOMs are retried; a writer failure never repeats inference.
            exc.__traceback__ = None
            if len(items) > 1:
                retry = True
                logger.warning(
                    "Batch of %d exceeded memory; retrying smaller batches.", len(items)
                )
            else:
                report(items[0][0], items[0][2], exc)
        except Exception as exc:
            predictions = None
            for data, _, seed in items:
                report(data, seed, exc)
            exc.__traceback__ = None
        if retry:
            cleanup_device_memory(runner.device)
            mid = len(items) // 2
            predict(items[:mid])
            predict(items[mid:])
            return
        if predictions is not None:
            completed = 0
            for (data, atom_array, seed), prediction in zip(
                items, predictions, strict=True
            ):
                try:
                    validate_prediction(
                        prediction,
                        n_atom=len(atom_array),
                        n_token=int(data["N_token"].item()),
                        n_sample=configs.sample_diffusion.N_sample,
                    )
                    runner.dumper.dump(
                        group_name="",
                        pdb_id=data["sample_name"],
                        seed=seed,
                        pred_dict=prediction,
                        atom_array=atom_array,
                        entity_poly_type={
                            k: v
                            for k, v in data["entity_poly_type"].items()
                            if v != "non-polymer"
                        },
                    )
                    completed += 1
                except Exception as exc:
                    report(data, seed, exc)
            del prediction
            logger.info(
                "[Rank %d] %d/%d jobs completed, %d samples/job, %.2fs including output",
                _distributed_rank(),
                completed,
                len(items),
                configs.sample_diffusion.N_sample,
                time.monotonic() - start,
            )
            if runner.device.type == "cuda":
                logger.info(
                    "[Rank %d] peak allocated %.3f GiB; peak reserved %.3f GiB",
                    _distributed_rank(),
                    torch.cuda.max_memory_allocated(runner.device) / 2**30,
                    torch.cuda.max_memory_reserved(runner.device) / 2**30,
                )
        del predictions
        cleanup_device_memory(runner.device)

    pending = []
    bucket = None
    for seed in dict.fromkeys(seed for seeds in schedule for seed in seeds):
        # Sort by approximate sequence length before sharding; exact token/atom
        # metadata remains authoritative after featurization.
        def estimated_size(index):
            return sum(
                len(next(iter(entity.values())).get("sequence", ""))
                * next(iter(entity.values())).get("count", 1)
                for entity in jobs[index]["sequences"]
            )

        loader.sampler.set_sample_indices(
            sorted(
                (i for i, seeds in enumerate(schedule) if seed in seeds),
                key=estimated_size,
            )
        )
        iterator = iter(loader)
        while True:
            seed_everything(seed=seed, deterministic=configs.deterministic)
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data, atom_array, error = _prepare_inference_batch(batch)
            owned = _sampler_owns_inference_sample(loader.sampler, data)
            if error:
                if owned or int(data.get("sample_index", -1)) == -1:
                    report(data, seed, ValueError(error))
                continue
            if not owned:
                continue
            current = _precision_bucket(int(data["N_token"].item()))
            if pending and current != bucket:
                predict(pending)
                pending = []
            bucket = current
            pending.append((data, atom_array, seed))
            if len(pending) == configs.batch_size:
                predict(pending)
                pending = []
    if pending:
        predict(pending)
    if errors:
        raise RuntimeError(
            f"{len(errors)} inference job/seed(s) failed. First error: {errors[0]}"
        )
