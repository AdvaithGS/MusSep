"""Stage 1: train BS-RoFormer on MUSDB18-HQ.

    python -m pipeline.train_base --config configs/roformer_base.yaml \
        --musdb-root /data/musdb18hq --out runs/stage1_vocals --tier base

Recipe (configs/roformer_base.yaml): random-mix augmentation, waveform L1 +
multi-resolution complex STFT L1, AdamW 5e-4 decayed x0.9 every 40k steps,
EMA 0.999, 8-second excerpts. Validation runs the EMA model over the 14
official validation tracks with the paper's chunked inference (8 s windows,
4 s hop, overlap-average) and reports global SDR (uSDR) per stem.

The validation number is *not* the paper's metric. The paper reports BSSEval
v4 SDR (median over 1-second frames, then median over the 50 test tracks);
use scripts/evaluate.py with --metric museval on the test set for that, and
only for a model trained on MUSDB18-HQ alone (9.80 dB average is the paper's
target for that setting, not something this repository has measured).

``--init`` fine-tunes from a checkpoint (one of ours, or a community
BS-RoFormer state dict with the same layout); a model initialised from a
community checkpoint is no longer a MUSDB-only model.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from data.musdb18 import MUSDB_STEMS, MUSDBRandomMixDataset, MUSDBTrackDataset
from models.bs_roformer import build_bs_roformer, load_community_state_dict
from pipeline.infer import STAGE1_KIND, Stage1Model, run_stage1
from utils.config import get, load_yaml, resolve_tier, tier_name
from utils.loss import MultiResolutionComplexSTFTLoss
from utils.metrics import sdr
from utils.training import (AmpContext, ModelEMA, RunLog, count_parameters, cycle, load_checkpoint,
                            make_optimizer, save_checkpoint, seed_everything, select_device)


def model_kwargs(config: Dict[str, Any], tier: Optional[str], num_stems: int) -> Dict[str, Any]:
    kwargs = resolve_tier(config, tier)
    kwargs["num_stems"] = num_stems
    if kwargs.get("freqs_per_bands") is not None:
        kwargs["freqs_per_bands"] = tuple(kwargs["freqs_per_bands"])
    return kwargs


@torch.no_grad()
def validate(model, stems: Sequence[str], dataset: MUSDBTrackDataset, device, amp, sample_rate: int,
             segment: int, overlap: float, batch_size: int) -> Dict[str, float]:
    """Mean over tracks of the global SDR per stem (plus ``avg``)."""
    entry = Stage1Model(model, list(stems), sample_rate, segment)
    totals = {name: [] for name in stems}
    for item in dataset:
        estimates = run_stage1(entry, item["mixture"], device, amp, overlap, batch_size)
        for name in stems:
            if item["targets"][name].abs().max() > 0:   # an all-silent reference has no SDR
                totals[name].append(sdr(item["targets"][name], estimates[name]))
    scores = {name: sum(values) / len(values) for name, values in totals.items() if values}
    if scores:
        scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def payload(model, ema, optimizer, scheduler, amp, step, kwargs, stems, audio, config, best) -> Dict[str, Any]:
    return {
        "kind": STAGE1_KIND, "step": step, "model_kwargs": kwargs, "target_stems": list(stems),
        "audio": audio, "config": config, "best": best,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": amp.scaler.state_dict(),
    }


def train(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    train_cfg = config.get("training", {})
    for key in ("batch_size", "max_steps", "lr", "grad_accum"):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value
    stems = tuple(args.stems or get(config, "training.target_stems", ["vocals"]))
    unknown = [s for s in stems if s not in MUSDB_STEMS]
    if unknown:
        raise SystemExit(f"unknown stems {unknown}; choose from {MUSDB_STEMS}")
    root = args.musdb_root or get(config, "data.musdb_root")
    if not root:
        raise SystemExit("give --musdb-root (or data.musdb_root in the config)")
    out = Path(args.out)
    log = RunLog(out)

    seed_everything(int(get(config, "training.seed", 0)))
    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision or get(config, "training.precision", "auto"))

    sample_rate = int(get(config, "audio.sample_rate", 44100))
    segment = int(round(float(get(config, "audio.segment_seconds", 8.0)) * sample_rate))
    overlap = float(get(config, "audio.inference_overlap", 0.5))
    audio = {"sample_rate": sample_rate, "segment_samples": segment}

    kwargs = model_kwargs(config, args.tier, len(stems))
    model = build_bs_roformer(kwargs).to(device)
    if args.init:
        source: Any = args.init
        if not str(args.init).endswith(".safetensors"):
            source = torch.load(str(args.init), map_location="cpu", weights_only=True)
            if isinstance(source, dict) and source.get("kind") == STAGE1_KIND:
                source = source["ema"]          # one of ours: start from its EMA weights
        load_community_state_dict(model, source, strict=True)
        print(f"initialised from {args.init}")
    ema = ModelEMA(model, float(get(config, "training.ema_decay", 0.999)), warmup=bool(get(config, "training.ema_warmup", True)))
    optimizer, scheduler = make_optimizer(
        model.parameters(), float(train_cfg.get("lr", 5e-4)), float(train_cfg.get("weight_decay", 0.0)),
        int(train_cfg.get("warmup_steps", 0)), float(train_cfg.get("lr_decay", 1.0)), int(train_cfg.get("lr_decay_every", 0)),
    )
    criterion = MultiResolutionComplexSTFTLoss(
        tuple(get(config, "loss.stft_window_sizes", (4096, 2048, 1024, 512, 256))),
        int(get(config, "loss.stft_hop_length", 147)), float(get(config, "loss.stft_weight", 1.0)),
    ).to(device)

    step, best = 0, None
    resume = Path(args.resume) if args.resume else (out / "last.pt" if (out / "last.pt").is_file() and not args.fresh else None)
    if resume:
        state = load_checkpoint(resume)
        if state["model_kwargs"] != kwargs or tuple(state["target_stems"]) != stems:
            raise SystemExit(f"{resume} was trained with a different model or stem list; use --fresh or another --out")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        amp.scaler.load_state_dict(state["scaler"])
        step, best = int(state["step"]), state.get("best")
        ema.updates = step
        print(f"resumed from {resume} at step {step}")

    data_cfg = config.get("data", {})
    dataset = MUSDBRandomMixDataset(
        root, split=data_cfg.get("split", "train"), target_stems=stems, segment_samples=segment, sample_rate=sample_rate,
        epoch_size=int(data_cfg.get("epoch_size", 10000)), random_mix=bool(data_cfg.get("random_mix", True)),
        gain_range_db=data_cfg.get("gain_db", (-3.0, 3.0)), silence_prob=float(data_cfg.get("silence_prob", 0.1)),
        loudness_threshold_db=float(data_cfg.get("loudness_threshold_db", -50.0)),
        channel_swap_prob=float(data_cfg.get("channel_swap_prob", 0.0)),
    )
    workers = int(args.workers if args.workers is not None else data_cfg.get("num_workers", 4))
    batch_size, accum = int(train_cfg.get("batch_size", 4)), max(1, int(train_cfg.get("grad_accum", 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    valid = None
    if data_cfg.get("split", "train") == "train" and int(train_cfg.get("valid_every", 0)) > 0:
        valid = MUSDBTrackDataset(root, "valid", stems, sample_rate, train_cfg.get("valid_max_seconds"), train_cfg.get("valid_max_tracks"))

    print(f"Stage 1 | stems {list(stems)} | tier {tier_name(config, args.tier)} | "
          f"{count_parameters(model) / 1e6:.2f}M parameters | {amp.describe()} | device {device} | "
          f"batch {batch_size} x {accum} | {len(dataset.tracks)} training tracks")
    if args.init:
        print("note: initialised from an external checkpoint; results are not MUSDB-only")

    max_steps = int(train_cfg.get("max_steps", 400000))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_every, valid_every, save_every = (int(train_cfg.get(k, d)) for k, d in
                                          (("log_every", 50), ("valid_every", 5000), ("save_every", 5000)))
    batches = cycle(loader)
    model.train()
    running, seen, t0 = 0.0, 0, time.time()
    while step < max_steps:
        for _ in range(accum):
            mixture, targets = next(batches)
            mixture, targets = mixture.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with amp.autocast():
                estimate = model(mixture)
            loss = criterion(estimate, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; try --precision fp32 or a lower lr")
            amp.backward(loss / accum)
            running += float(loss.detach())
            seen += 1
        grad_norm = amp.step(optimizer, model.parameters(), grad_clip)
        scheduler.step()
        ema.update(model)
        step += 1

        if step % log_every == 0:
            log("train", step, loss=running / seen, grad_norm=grad_norm, lr=scheduler.get_last_lr()[0],
                sec_per_step=(time.time() - t0) / log_every)
            running, seen, t0 = 0.0, 0, time.time()
        if valid is not None and (step % valid_every == 0 or step == max_steps):
            ema.module.eval()
            scores = validate(ema.module, stems, valid, device, amp, sample_rate, segment, overlap,
                              int(train_cfg.get("valid_batch_size", 4)))
            log("valid", step, **{f"usdr_{k}": v for k, v in scores.items()})
            if scores and (best is None or scores["avg"] > best["usdr"]):
                best = {"step": step, "usdr": scores["avg"]}
                save_checkpoint(out / "best.pt", payload(model, ema, optimizer, scheduler, amp, step, kwargs, stems, audio, config, best))
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(out / "last.pt", payload(model, ema, optimizer, scheduler, amp, step, kwargs, stems, audio, config, best))
    print(f"done: {out / 'last.pt'}" + (f"; best validation uSDR {best['usdr']:.2f} dB at step {best['step']}" if best else ""))
    return out / "last.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/roformer_base.yaml")
    parser.add_argument("--musdb-root")
    parser.add_argument("--out", required=True, help="run folder (checkpoints and log.jsonl)")
    parser.add_argument("--tier", help="override model.tier")
    parser.add_argument("--stems", nargs="+", help="override training.target_stems")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--init", help="initial weights (fine-tuning)")
    parser.add_argument("--resume", help="checkpoint to resume (default: <out>/last.pt if present)")
    parser.add_argument("--fresh", action="store_true", help="ignore <out>/last.pt")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
