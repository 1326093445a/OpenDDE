#!/usr/bin/env python3
"""Warm, repeated checkpoint inference with retained predictions and provenance.

Run with the project's Python and cached assets. Output artifacts are external
benchmark data; do not commit them. Model timings exclude preparation/writing.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def compare(left, right):
    """Compare all tensor/array/scalar outputs, including confidence and mapping."""
    differences = []

    def visit(a, b, path):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            a, b = torch.as_tensor(a), torch.as_tensor(b)
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch at {path}: {a.shape}, {b.shape}")
            if not torch.equal(a, b):
                delta = a.double() - b.double()
                differences.append(
                    {
                        "path": path,
                        "max_abs": delta.abs().max().item(),
                        "rms": delta.square().mean().sqrt().item(),
                    }
                )
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f"Key mismatch at {path}")
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise ValueError(f"Length mismatch at {path}")
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, f"{path}[{i}]")
        elif a != b:
            differences.append({"path": path, "before": a, "after": b})

    visit(left, right, "prediction")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--backend", choices=["reference", "efficient"], default="reference"
    )
    parser.add_argument("--execution", choices=["batch", "singletons"], default="batch")
    parser.add_argument(
        "--trimul-kernel",
        choices=["auto", "cuequivariance", "torch", "flashpairformer"],
        default="auto",
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--confidence-chunk", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--pad-trunk-to", type=int, default=0)
    parser.add_argument("--random-padding", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two measured passes for repeatability.")
    sys.path.insert(0, str(args.repo.resolve()))
    from opendde.data.inference.infer_dataloader import InferenceDataset
    from opendde.utils.seed import seed_everything
    from runner.batch_inference import get_default_runner
    from runner.batched_prediction import validate_prediction
    from runner.inference import _prepare_prediction_batch

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.input.read_text())
    if args.names:
        jobs = [job for job in jobs if job["name"] in args.names]
    if not jobs:
        raise ValueError("No selected jobs")
    # No external alignments/templates; use the complete requested entities.
    for job in jobs:
        for entity in job["sequences"]:
            for value in entity.values():
                for key in ("msa", "templates"):
                    value.pop(key, None)
    hashes = {
        str(p.relative_to(args.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in ("opendde", "runner")
        for p in (args.repo / root).rglob("*.py")
    }
    report = {
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "sources": hashes,
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "gpu": torch.cuda.get_device_name(),
        "runs": [],
    }
    report["libraries"] = {}
    for package in (
        "triton",
        "opt_core",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    ):
        try:
            report["libraries"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["libraries"][package] = None
    report["cueq_tuning"] = {
        key: os.getenv(key) for key in ("CUEQ_TRITON_TUNING", "CUEQ_TRITON_CACHE_DIR")
    }
    with get_default_runner(
        dump_dir=str(args.output / "predictions"),
        batch_size=max(2, len(jobs)),
        seeds=[101],
        n_cycle=args.cycles,
        n_step=args.steps,
        n_sample=args.samples,
        dtype=args.dtype,
        device="cuda",
        use_msa=False,
        use_template=False,
        use_rna_msa=False,
        deterministic=args.deterministic,
        diffusion_attention_backend=args.backend,
        trimul_kernel=args.trimul_kernel,
        confidence_sample_chunk_size=args.confidence_chunk,
    ) as runner:
        if args.pad_trunk_to:
            # Diagnostic-only hooks: change trunk shape while holding inputs,
            # RNG, diffusion shape and all configured model settings fixed.
            original_lengths = []
            pad_rng = torch.Generator(device="cuda").manual_seed(770)

            def pad_trunk(module, positional, kw):
                kw = dict(kw)
                n = kw["s"].shape[-2]
                original_lengths.append(n)
                target = max(n, args.pad_trunk_to)
                for key in ("s", "z", "pair_mask", "extra_attn_bias"):
                    value = kw[key]
                    if key == "s":
                        shape = (*value.shape[:-2], target, value.shape[-1])
                        slices = (..., slice(0, n), slice(None))
                    elif key == "z":
                        shape = (*value.shape[:-3], target, target, value.shape[-1])
                        slices = (..., slice(0, n), slice(0, n), slice(None))
                    else:
                        shape = (*value.shape[:-2], target, target)
                        slices = (..., slice(0, n), slice(0, n))
                    if args.random_padding and key in ("s", "z"):
                        padded = torch.randn(
                            shape,
                            dtype=value.dtype,
                            device=value.device,
                            generator=pad_rng,
                        )
                    else:
                        padded = value.new_full(
                            shape, -1e9 if key == "extra_attn_bias" else 0
                        )
                    padded[slices] = value
                    kw[key] = padded
                return positional, kw

            def crop_trunk(module, positional, kw, output):
                n = original_lengths.pop()
                s, z = output
                return s[..., :n, :].clone(), z[..., :n, :n, :].clone()

            runner.model.pairformer_stack.register_forward_pre_hook(
                pad_trunk, with_kwargs=True
            )
            runner.model.pairformer_stack.register_forward_hook(
                crop_trunk, with_kwargs=True
            )
        data, atoms = [], []
        dataset = InferenceDataset(runner.configs, inputs=jobs)
        for index in range(len(jobs)):
            seed_everything(101, args.deterministic)
            item, atom_array, error = dataset[index]
            if error:
                raise RuntimeError(error)
            _prepare_prediction_batch(runner, runner.configs, item, seed=101)
            data.append(item)
            atoms.append(atom_array)
        report["tokens"] = [int(item["N_token"].item()) for item in data]
        report["runtime"] = {
            "seed": 101,
            "tf32": runner.configs.enable_tf32,
            "projection_cache_mb": runner.configs.infer_setting.diffusion_cache_max_mb,
            "skip_amp": runner.configs.skip_amp.to_dict(),
            "model": runner.configs.model_name,
        }
        report["kernels"] = {
            k: getattr(runner.configs, k)
            for k in ("triangle_multiplicative", "triangle_attention")
        }
        if args.trimul_kernel == "flashpairformer":
            from opendde.model.triangular.flashpairformer import flashpairformer_stats
        torch.save(data, args.output / "features.pt")
        first = None
        for run in range(args.repeats + 1):
            seed_everything(101, args.deterministic)
            inputs = copy.deepcopy(data)
            groups = [inputs] if args.execution == "batch" else [[x] for x in inputs]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            pred = [p for group in groups for p in runner.predict_batch(group)]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            record = {
                "seconds": elapsed,
                "warmup": run == 0,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_processes": subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                ).stdout,
            }
            if args.trimul_kernel == "flashpairformer":
                record["flashpairformer_counts_cumulative"] = flashpairformer_stats(
                    runner.model
                )
            for item, array, prediction in zip(data, atoms, pred, strict=True):
                validate_prediction(
                    prediction,
                    n_atom=len(array),
                    n_token=int(item["N_token"]),
                    n_sample=args.samples,
                )
            current = cpu_tree(pred)
            del pred
            if run == 1:
                first = current
            if run > 1:
                record["repeat_differences"] = compare(first, current)
            report["runs"].append(record)
            print(
                json.dumps(
                    {
                        "run": run,
                        "seconds": elapsed,
                        "peak_gib": record["peak_gib"],
                        "repeat_differences": len(record.get("repeat_differences", [])),
                    }
                ),
                flush=True,
            )
        torch.save(current, args.output / "last.pt")
        if args.compare:
            report["baseline_differences"] = compare(
                torch.load(args.compare, map_location="cpu", weights_only=False),
                current,
            )
        start = time.perf_counter()
        for item, array, prediction in zip(data, atoms, current, strict=True):
            runner.dumper.dump(
                group_name="",
                pdb_id=item["sample_name"],
                seed=101,
                pred_dict=prediction,
                atom_array=array,
                entity_poly_type={
                    k: v
                    for k, v in item["entity_poly_type"].items()
                    if v != "non-polymer"
                },
            )
        report["write_seconds"] = time.perf_counter() - start
        report["median_seconds"] = statistics.median(
            r["seconds"] for r in report["runs"][1:]
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
