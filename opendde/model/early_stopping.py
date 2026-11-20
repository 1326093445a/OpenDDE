# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# First-sample policy adapted from Mol-1, Copyright 2026 Molence Bio (Apache-2.0).
"""Optional first-generated-sample screening; never interpret missing ipSAE as zero."""

import math
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_rng(device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class FirstSampleGate:
    @classmethod
    def create(cls, model, features, pairs, chunk, kernels):
        threshold = model.configs.infer_setting.get("early_stop_ipsae_threshold")
        if threshold is None or model.configs.sample_diffusion.N_sample <= 1:
            return None
        if pairs[0][1].device.type not in ("cpu", "cuda"):
            raise ValueError("First-sample screening supports CPU and CUDA")
        return cls(model, features, pairs, chunk, kernels, threshold)

    def __init__(self, model, features, pairs, chunk, kernels, threshold):
        self.model, self.features, self.pairs = model, features, pairs
        self.chunk, self.kernels, self.threshold = chunk, kernels, float(threshold)
        self.requested = model.configs.sample_diffusion.N_sample
        self.first = [None] * len(features)
        self.first_summary = [None] * len(features)
        self.scores = [None] * len(features)
        self.stopped = False
        self.device = pairs[0][1].device
        # The callback runs inside diffusion's AMP scope. Restore the outer
        # prediction scope for confidence, which has its own precision policy.
        self.amp_enabled = torch.is_autocast_enabled(self.device.type)
        self.amp_dtype = torch.get_autocast_dtype(self.device.type)

    def heads(self, i, coordinate):
        from opendde.model.batching import run_heads

        return run_heads(
            self.model,
            self.features[i],
            self.pairs[i],
            coordinate,
            self.chunk,
            self.kernels,
            summarize=False,
        )

    def __call__(self, coordinates):
        with (
            preserve_rng(self.device),
            torch.autocast(
                self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype
            ),
        ):
            for i, coordinate in enumerate(coordinates):
                pred = self.heads(i, coordinate)
                # Baseline multi-sample heads retain pair logits on CPU.
                for key in ("pae", "pde"):
                    pred[key] = pred[key].cpu()
                self.first[i] = pred
                from opendde.model.batching import summarize_prediction

                scored = summarize_prediction(
                    self.model, dict(pred), self.features[i], self.pairs[i]
                )
                self.first_summary[i] = scored
                score = scored["summary_confidence"][0].get("ipsae_min_10")
                score = float(score) if score is not None else None
                self.scores[i] = (
                    score
                    if score is not None and math.isfinite(score) and 0 <= score <= 1
                    else None
                )
        self.stopped = all(
            score is not None
            and math.isfinite(score)
            and (score == 0 or score < self.threshold)
            for score in self.scores
        )
        return self.stopped

    def finish(
        self,
        i,
        coordinate,
        *,
        batch_policy="preserve_shape_all_inputs",
        batch_size=None,
    ):
        first = self.first[i]
        if first is None:
            raise RuntimeError("First-sample confidence was not computed")
        if coordinate.shape[0] == 1:
            pred = self.first_summary[i]
        else:
            from opendde.model.batching import summarize_prediction

            rest = self.heads(i, coordinate[1:])
            pred = {"coordinate": coordinate, "contact_probs": first["contact_probs"]}
            for key in ("plddt", "pae", "pde", "resolved"):
                # Match the ordinary multi-sample confidence offload policy,
                # including two samples whose individual head calls stay on GPU.
                device = (
                    torch.device("cpu") if key in ("pae", "pde") else rest[key].device
                )
                pred[key] = torch.cat(
                    (first[key].to(device), rest[key].to(device)), dim=0
                )
            summarize_prediction(self.model, pred, self.features[i], self.pairs[i])
        completed = coordinate.shape[0]
        score = self.scores[i]
        info = dict(
            metric="ipsae_min_10",
            threshold=self.threshold,
            first_sample_score=score,
            first_sample_generation_index=0,
            batch_policy=batch_policy,
            batch_size=batch_size or len(self.features),
            requested_samples=self.requested,
            completed_samples=completed,
            skipped_samples=self.requested - completed,
            stopped=self.stopped,
            reason=(
                "first_sample_below_threshold"
                if self.stopped
                else "ipsae_unavailable"
                if score is None or not math.isfinite(score)
                else "batch_not_all_below_threshold"
                if len(self.features) > 1
                else "first_sample_passed"
            ),
        )
        pred["sampling_early_stop"] = info
        for j, summary in enumerate(pred["summary_confidence"]):
            summary["sample_generation_index"] = j
            summary["sampling_early_stop"] = dict(info)
        return pred
