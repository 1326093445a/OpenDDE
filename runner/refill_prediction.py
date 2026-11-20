# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Session scheduling adapted from Mol-1, Copyright 2026 Molence Bio (Apache-2.0).
"""Bounded exact-shape sessions with per-job output and failure ownership."""

import itertools
import time


from opendde.model.denoiser_graph import DenoiserGraphCaptureError
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


def run_refill_sessions(runner, configs, prepared, submit, report):
    from runner.batched_prediction import _batch_token_sizes
    from runner.inference import _prepare_prediction_batch, cleanup_device_memory

    for _, jobs in itertools.groupby(
        prepared, key=lambda item: _batch_token_sizes(item[0])
    ):
        pending = {}
        counter = itertools.count()

        def batches():
            while items := list(itertools.islice(jobs, configs.batch_size)):
                batch = []
                # Register all consumed jobs before preparation, which can fail.
                for data, atoms, seed in items:
                    key = next(counter)
                    pending[key] = data, atoms, seed
                    batch.append((key, data, seed))
                for _, data, seed in batch:
                    _prepare_prediction_batch(runner, configs, data, seed=seed)
                yield batch

        def completed(key, pred):
            data, atoms, seed = pending.pop(key)
            try:
                submit(data, atoms, seed, pred)
            except Exception as exc:
                report(data, seed, exc)

        started = time.monotonic()
        try:
            stats = runner.predict_refill(batches(), completed)
            if pending:
                raise ValueError("Refill returned without emitting every admitted job")
            remaining = next(jobs, None)
            if remaining is not None:
                pending[next(counter)] = remaining
                raise ValueError(
                    "Refill returned before consuming every job in the session"
                )
            logger.info(
                "Fixed-slot refill: %s, %.2fs including output",
                stats,
                time.monotonic() - started,
            )
        except DenoiserGraphCaptureError:
            logger.critical("Denoiser capture failed; stopping refill worker.")
            raise
        except Exception as exc:
            detail = RuntimeError(
                f"Refill session aborted: {type(exc).__name__}: {exc}"
            )
            exc.__traceback__ = None
            failed = False
            for data, _, seed in pending.values():
                report(data, seed, detail)
                failed = True
            for data, _, seed in jobs:
                report(data, seed, detail)
                failed = True
            if not failed:
                raise detail from None
            pending.clear()
            # Never shrink B or retry already published jobs after an OOM.
            cleanup_device_memory(runner.device)
