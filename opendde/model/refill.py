# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Fixed-slot scheduling adapted from Mol-1, Copyright 2026 Molence Bio (Apache-2.0).
"""Bounded refill between complete diffusion samples, preserving fixed slot shapes."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np
import torch

from opendde.config.inference import validate_input_batching
from opendde.model import batching
from opendde.model.early_stopping import FirstSampleGate


@dataclass
class Slot:
    key: object
    seed: int
    state: tuple
    schedule: torch.Tensor
    gate: object
    generator: torch.Generator
    numpy_rng: np.random.Generator
    coordinates: list = field(default_factory=list)


def forward_refill(model, batches, on_prediction):
    """Consume batches of (key, features, seed); callback owns output publication.

    Trunks and denoisers keep B even in partial tails. Retired slots remain
    placeholders. Replacing slots rebuilds batch projections and any replay graph;
    no stale conditioning or graph addresses survive a replacement.
    """
    cfg = model.configs
    validate_input_batching(cfg)
    if not cfg.infer_setting.batch_refill or model.training:
        raise ValueError("forward_refill requires batch_refill=true and eval mode")
    width = cfg.batch_size
    batches, ready = iter(batches), deque()
    lengths, chunk, exhausted = None, None, False
    stats = dict(
        completed=0,
        stopped=0,
        trunk_batches=0,
        sample_rounds=0,
        real_samples=0,
        placeholder_samples=0,
        refills=0,
        max_ready=0,
    )

    def prepare_next():
        nonlocal lengths, chunk, exhausted
        items = next(batches, None)
        if items is None:
            exhausted = True
            return []
        if not 1 <= len(items) <= width or any(key is None for key, _, _ in items):
            raise ValueError(
                "Refill preparation requires 1..batch_size jobs and non-None keys"
            )
        sizes = [
            (
                feat["token_index"].numel(),
                feat.get("parent_residue_idx", feat["token_index"]).numel(),
            )
            for _, feat, _ in items
        ]
        if lengths is None:
            lengths = sizes[0]
        if any(size != lengths for size in sizes):
            raise ValueError(
                "Refill requires exactly equal residue and structural token counts"
            )
        budget = cfg.infer_setting.batch_max_pair_tokens
        if budget and any(width * n**2 > budget for n in lengths):
            raise ValueError(
                "Fixed refill batch exceeds batch_max_pair_tokens; choose a smaller batch_size"
            )
        padded = list(items) + [(None, items[-1][1], 0)] * (width - len(items))
        features = [
            dict(feat, inference_seed=torch.tensor(seed)) for _, feat, seed in padded
        ]
        states, schedules, _, pairs, chunk, kernels = batching.prepare_batch(
            model, features
        )
        slots = []
        for (key, _, seed), state, schedule, pair in zip(
            padded, states, schedules, pairs, strict=True
        ):
            gate = (
                FirstSampleGate.create(model, [state[0]], [pair], chunk, kernels)
                if key is not None
                else None
            )
            slots.append(
                Slot(
                    key,
                    seed,
                    state,
                    schedule,
                    gate,
                    torch.Generator(device=state[1].device).manual_seed(seed),
                    np.random.default_rng(seed),
                )
            )
        stats["trunk_batches"] += 1
        return slots

    slots = prepare_next()
    if not slots:
        return stats
    dirty = True
    # ExitStack forwards capture failures to the graph's exception-aware exit.
    with ExitStack() as scope:
        while any(slot.key is not None for slot in slots):
            if dirty:
                scope.close()
                sample = None  # Release the old closure before allocating replacements.
                sample = scope.enter_context(
                    batching.sampling_session(
                        model,
                        [slot.state for slot in slots],
                        [slot.schedule for slot in slots],
                        [slot.seed for slot in slots],
                        chunk,
                    )
                )
                dirty = False
            coordinates = sample(
                n_sample=1,
                generators=[slot.generator for slot in slots],
                numpy_rngs=[slot.numpy_rng for slot in slots],
            )
            stats["sample_rounds"] += 1
            stats["real_samples"] += sum(slot.key is not None for slot in slots)
            stats["placeholder_samples"] += sum(slot.key is None for slot in slots)
            for slot, coordinate in zip(slots, coordinates, strict=True):
                if slot.key is None:
                    continue
                slot.coordinates.append(coordinate)
                stopped = (
                    slot.gate([coordinate]) if len(slot.coordinates) == 1 else False
                )
                if stopped or len(slot.coordinates) == cfg.sample_diffusion.N_sample:
                    pred = slot.gate.finish(
                        0,
                        torch.cat(slot.coordinates),
                        batch_policy="preserve_shape_refill",
                        batch_size=width,
                    )
                    on_prediction(slot.key, pred)
                    del pred
                    stats["completed"] += 1
                    stats["stopped"] += int(stopped)
                    slot.key, slot.gate = None, None
                    slot.coordinates.clear()
            del coordinates, coordinate
            for i, slot in enumerate(slots):
                if slot.key is not None:
                    continue
                if not ready and not exhausted:
                    scope.close()
                    sample = None
                    dirty = True
                    ready.extend(
                        item for item in prepare_next() if item.key is not None
                    )
                if ready:
                    slots[i] = ready.popleft()
                    stats["refills"] += 1
                    dirty = True
                stats["max_ready"] = max(stats["max_ready"], len(ready))
    return stats
