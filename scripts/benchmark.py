"""Measure training speed and memory before paying for a long run.

    # every Stage 1 tier: largest batch that fits, seconds per step, peak memory
    python -m scripts.benchmark stage1 --tiers small base
    # ... and whether the data loader (CPU) can keep up, for a few worker counts
    python -m scripts.benchmark stage1 --tiers base --musdb-root /data/musdb18hq --workers 2 4 8

    # Stage 2 on synthetic latents shaped like the real ones, or read the shape from a manifest
    python -m scripts.benchmark stage2 --tiers small base --manifest /data/latents/manifest.json --workers 2 4

    # turn time into money
    python -m scripts.benchmark stage1 --tiers base --price-per-hour 0.45

For each tier it runs a few real optimizer steps (forward, loss, backward,
AdamW, with the configured precision) on random inputs of the training shape:

* batch: the largest power of two that fits in GPU memory (``--batch-size auto``,
  CUDA only), or the one you give;
* seconds per step and examples per second, after warm-up;
* peak GPU memory;
* the time for the config's ``max_steps`` at the config's effective batch size
  (batch_size x grad_accum), assuming throughput in examples per second stays
  what was measured; and its cost if ``--price-per-hour`` is given.

With a dataset (``--musdb-root`` / ``--manifest``) it also times the data
loader alone for each ``--workers`` value. If producing a batch takes longer
than a training step, the GPU waits for the CPU: use more workers, a machine
with more CPU cores, or (Stage 2) the in-memory latent cache.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from utils.config import load_yaml, resolve_tier
from utils.runtime import available_cpus, configure_backends, configure_threads, make_loader, maybe_compile
from utils.training import AmpContext, count_parameters, select_device

MAX_BATCH = 512


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def time_steps(make_step: Callable[[int], Callable[[], None]], batch: int, device: torch.device,
               steps: int, warmup: int) -> Tuple[float, float]:
    """(seconds per step, peak GB) for ``batch``; raises on out-of-memory."""
    step = make_step(batch)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    seconds = (time.perf_counter() - t0) / max(1, steps)
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else math.nan
    return seconds, peak


def largest_batch(make_step: Callable[[int], Callable[[], None]], device: torch.device) -> int:
    """Largest power of two whose step runs without running out of memory."""
    batch, best = 1, 0
    while batch <= MAX_BATCH:
        try:
            time_steps(make_step, batch, device, steps=1, warmup=1)
            best = batch
            batch *= 2
        except Exception as error:   # noqa: BLE001 - OOM surfaces as different exception types
            if not _is_oom(error):
                raise
            break
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best == 0:
        raise RuntimeError("even a batch of 1 does not fit; try a smaller tier, gradient checkpointing or fp16")
    return best


# --------------------------------------------------------------------------- the two models


def stage1_steps(config: Dict, tier: str, device: torch.device, amp: AmpContext, compile_model: bool):
    from models.bs_roformer import build_bs_roformer
    from utils.loss import MultiResolutionComplexSTFTLoss

    kwargs = resolve_tier(config, tier)
    stems = len(config.get("training", {}).get("target_stems", ["vocals"]))
    kwargs["num_stems"] = stems
    model = build_bs_roformer(kwargs).to(device).train()
    forward = maybe_compile(model, compile_model)
    criterion = MultiResolutionComplexSTFTLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    segment = int(round(float(config["audio"]["segment_seconds"]) * int(config["audio"]["sample_rate"])))

    def make_step(batch: int):
        mixture = torch.randn(batch, 2, segment, device=device) * 0.1
        target = torch.randn(batch, stems, 2, segment, device=device) * 0.1

        def step():
            with amp.autocast():
                out = forward(mixture)
            amp.backward(criterion(out, target))
            amp.step(optimizer, model.parameters(), 1.0)
        return step
    return model, make_step


def stage2_steps(config: Dict, tier: str, device: torch.device, amp: AmpContext, compile_model: bool,
                 channels: int, frames: int, slots: int, classes: int):
    from models.dit import build_dit
    from pipeline.flow_matching import MaskedOTFlowMatcher

    model = build_dit(resolve_tier(config, tier), channels, classes).to(device).train()
    forward = maybe_compile(model, compile_model)
    loss_fn = MaskedOTFlowMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def make_step(batch: int):
        x1 = torch.randn(batch, slots, channels, frames, device=device)
        parent = torch.randn(batch, channels, frames, device=device)
        labels = torch.randint(1, classes, (batch, slots), device=device)
        valid = torch.ones(batch, slots, dtype=torch.bool, device=device)

        def step():
            with amp.autocast():
                loss = loss_fn(forward, x1, parent, labels, valid)
            amp.backward(loss)
            amp.step(optimizer, model.parameters(), 1.0)
        return step
    return model, make_step


# --------------------------------------------------------------------------- data loaders


def time_loader(dataset, batch: int, workers: int, device: torch.device, batches: int, sampler=None) -> float:
    """Seconds per batch for the loader alone (after the workers have started)."""
    loader = make_loader(dataset, batch, workers, device, shuffle=sampler is None, sampler=sampler,
                         drop_last=True, persistent=False)
    iterator = iter(loader)
    next(iterator)                                   # worker start-up is not steady-state throughput
    t0 = time.perf_counter()
    count = 0
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            break
        count += 1
    del iterator
    return (time.perf_counter() - t0) / max(1, count)


# --------------------------------------------------------------------------- main


def run(args: argparse.Namespace) -> Dict:
    device = select_device(args.device)
    configure_backends(not args.no_tf32)
    configure_threads(device, 0)
    amp = AmpContext(device, precision=args.precision)
    config = load_yaml(args.config or ("configs/roformer_base.yaml" if args.stage == "stage1" else "configs/dit_latent.yaml"))
    training = config.get("training", {})
    effective = int(training.get("batch_size", 1)) * max(1, int(training.get("grad_accum", 1)))
    max_steps = int(args.max_steps or training.get("max_steps", 0))
    if args.stage == "stage1" and args.segment_seconds:
        config["audio"]["segment_seconds"] = args.segment_seconds

    shape = None
    dataset = sampler = None
    if args.stage == "stage2":
        channels, frames, slots, classes = args.latent_channels, None, args.slots, args.classes
        rate = args.frame_rate
        if args.manifest:
            from data.unified_dataset import UnifiedManifestDataset, load_manifest
            manifest = load_manifest(args.manifest)
            rate = float(manifest["latent_rate"])
            frames = max(8, int(round(float(config["data"]["segment_seconds"]) * rate)))
            dataset = UnifiedManifestDataset(args.manifest, "train", segment_frames=frames,
                                             cache=args.cache, full_set_prob=float(config["data"].get("full_set_prob", 0.5)))
            sampler = dataset.make_sampler()
            channels, slots, classes = dataset.channels, dataset.max_stems, dataset.num_classes
        frames = frames or max(8, int(round(float(config["data"]["segment_seconds"]) * rate)))
        shape = {"latent_channels": channels, "frames": frames, "slots": slots, "classes": classes}
    elif args.musdb_root:
        from data.musdb18 import MUSDBRandomMixDataset
        sample_rate = int(config["audio"]["sample_rate"])
        dataset = MUSDBRandomMixDataset(args.musdb_root, target_stems=tuple(training.get("target_stems", ["vocals"])),
                                        segment_samples=int(round(float(config["audio"]["segment_seconds"]) * sample_rate)),
                                        sample_rate=sample_rate)

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU ({available_cpus()} cores)"
    print(f"{args.stage} on {name} | {amp.describe()} | TF32 {'off' if args.no_tf32 else 'on'} | "
          f"compile {'on' if args.compile else 'off'}" + (f" | latents {shape}" if shape else ""))
    rows: List[Dict] = []
    for tier in args.tiers:
        if args.stage == "stage1":
            model, make_step = stage1_steps(config, tier, device, amp, args.compile)
        else:
            model, make_step = stage2_steps(config, tier, device, amp, args.compile, shape["latent_channels"],
                                            shape["frames"], shape["slots"], shape["classes"])
        if args.batch_size == "auto":
            batch = largest_batch(make_step, device) if device.type == "cuda" else 1
        else:
            batch = int(args.batch_size)
        seconds, peak = time_steps(make_step, batch, device, args.steps, args.warmup)
        per_second = batch / seconds
        row = {"tier": tier, "parameters_m": round(count_parameters(model) / 1e6, 2), "batch": batch,
               "sec_per_step": round(seconds, 4), "examples_per_sec": round(per_second, 2),
               "peak_gb": None if math.isnan(peak) else round(peak, 2)}
        if max_steps:
            hours = max_steps * effective / per_second / 3600
            row.update(effective_batch=effective, max_steps=max_steps, hours=round(hours, 1))
            if args.price_per_hour:
                row["cost"] = round(hours * args.price_per_hour, 2)
        rows.append(row)
        print(f"  {tier:8s} {row['parameters_m']:8.2f}M  batch {batch:4d}  {seconds:8.3f} s/step  "
              f"{per_second:8.1f} ex/s  peak {row['peak_gb'] if row['peak_gb'] is not None else '-'} GB"
              + (f"  | {max_steps} steps x batch {effective}: {row['hours']} h" if max_steps else "")
              + (f", ${row['cost']}" if "cost" in row else ""), flush=True)
        del model, make_step
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loaders: List[Dict] = []
    if dataset is not None:
        batch = rows[-1]["batch"] if rows else 1
        need = rows[-1]["sec_per_step"] if rows else None
        print(f"data loader, batch {batch} (one training step of the last tier takes {need} s):")
        for workers in args.workers:
            seconds = time_loader(dataset, batch, workers, device, args.loader_batches, sampler)
            verdict = None if need is None else ("keeps up" if seconds <= need else "TOO SLOW: the GPU will wait")
            loaders.append({"workers": workers, "sec_per_batch": round(seconds, 4), "verdict": verdict})
            print(f"  {workers:3d} workers  {seconds:8.3f} s/batch  {verdict or ''}", flush=True)
    return {"stage": args.stage, "device": name, "precision": amp.describe(), "tiers": rows, "loader": loaders,
            "shape": shape}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--config", help="default: the stage's config in configs/")
    parser.add_argument("--tiers", nargs="+", default=["small"])
    parser.add_argument("--batch-size", default="auto", help="'auto' (largest power of two that fits, CUDA) or a number")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-steps", type=int, help="default: training.max_steps of the config")
    parser.add_argument("--price-per-hour", type=float)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segment-seconds", type=float, help="Stage 1: override audio.segment_seconds")
    parser.add_argument("--musdb-root", help="Stage 1: also time the data loader")
    parser.add_argument("--manifest", help="Stage 2: latent shape from the manifest, and time the data loader")
    parser.add_argument("--cache", choices=["auto", "ram", "none"], default="auto", help="Stage 2 latent cache")
    parser.add_argument("--latent-channels", type=int, default=256, help="Stage 2 without a manifest")
    parser.add_argument("--frame-rate", type=float, default=25.0, help="Stage 2 without a manifest")
    parser.add_argument("--slots", type=int, default=7, help="Stage 2 without a manifest")
    parser.add_argument("--classes", type=int, default=16, help="Stage 2 without a manifest (including the null label)")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--loader-batches", type=int, default=20)
    parser.add_argument("--out", help="write the results as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    result = run(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
