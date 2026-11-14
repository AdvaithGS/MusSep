"""Stage 2/3: train the joint sub-stem DiT with masked OT flow matching.

    python -m scripts.preprocess_latents ... --out /data/latents        # once
    python -m pipeline.train_generative --config configs/dit_latent.yaml \
        --manifest /data/latents/manifest.json --out runs/stage2 --tier base

Each example is one parent stem (vocals, drums or other) of one song: the
parent's latent conditions the model, and the latents of its sub-stem classes
are generated jointly, one slot per class (see data/unified_dataset.py for
the slot layout and the hybrid treatment of absent classes). Label dropout
trains the unconditional branch used by classifier-free guidance.

The DAC-VAE never runs here: everything is read from pre-encoded latents, and
the latent statistics, silence latent and taxonomy are copied from the
manifest into the checkpoint so that inference needs only the checkpoint and
the VAE.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from data.unified_dataset import UnifiedManifestDataset, latent_info, load_manifest
from models.dit import build_dit
from pipeline.flow_matching import MaskedOTFlowMatcher, drop_labels
from pipeline.infer import STAGE2_KIND
from utils.config import get, load_yaml, resolve_tier, tier_name
from utils.training import (AmpContext, ModelEMA, RunLog, count_parameters, cycle, load_checkpoint,
                            make_optimizer, save_checkpoint, seed_everything, select_device)


def segment_frames(config: Dict[str, Any], manifest: Dict[str, Any]) -> int:
    return max(8, int(round(float(get(config, "data.segment_seconds", 10.0)) * float(manifest["latent_rate"]))))


def make_loaders(config: Dict[str, Any], manifest_path: str, batch_size: int, workers: int, pin: bool
                 ) -> Tuple[UnifiedManifestDataset, DataLoader, Optional[DataLoader]]:
    manifest = load_manifest(manifest_path)
    data = config.get("data", {})
    frames = segment_frames(config, manifest)
    common = dict(segment_frames=frames, bleed_cond_prob=float(data.get("bleed_cond_prob", 0.5)),
                  min_active_fraction=float(data.get("min_active_fraction", 0.25)),
                  min_class_active_fraction=float(data.get("min_class_active_fraction", 0.05)))
    train_set = UnifiedManifestDataset(manifest_path, "train", full_set_prob=float(data.get("full_set_prob", 0.5)),
                                       slakh_fraction=float(data.get("slakh_fraction", 1 / 3)), **common)
    loader = DataLoader(train_set, batch_size=batch_size, sampler=train_set.make_sampler(), num_workers=workers,
                        drop_last=True, pin_memory=pin, persistent_workers=workers > 0)
    try:
        val_set = UnifiedManifestDataset(manifest_path, "val", deterministic=True, **common)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    except RuntimeError:
        val_loader = None
    return train_set, loader, val_loader


@torch.no_grad()
def validation_loss(loss_fn, model, loader, device, amp, max_batches: int, seed: int = 1234) -> float:
    """Flow-matching loss on held-out songs with fixed noise and times (comparable across steps)."""
    total, count = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index, (parent, stems, labels, valid) in enumerate(loader):
            if index >= max_batches:
                break
            parent, stems, labels, valid = (x.to(device) for x in (parent, stems, labels, valid))
            with amp.autocast():
                total += float(loss_fn(model, stems, parent, labels, valid))
            count += 1
    return total / max(1, count)


def stage2_payload(model, ema, kwargs, dataset: UnifiedManifestDataset, step: int, config, *, optimizer=None,
                   scheduler=None, amp=None, distilled: bool = False, num_steps: int = 50, cfg_scale: float = 1.0,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "kind": STAGE2_KIND, "step": step, "model_kwargs": kwargs, "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None, "taxonomy": dataset.taxonomy.to_dict(),
        "latent": dataset.info, "segment_frames": dataset.segment_frames, "distilled": distilled,
        "num_steps": num_steps, "cfg_scale": cfg_scale, "config": config,
    }
    if optimizer is not None:
        out.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=amp.scaler.state_dict())
    out.update(extra or {})
    return out


def train(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    tc = config.get("training", {})
    for key in ("batch_size", "max_steps", "lr", "grad_accum"):
        if getattr(args, key) is not None:
            tc[key] = getattr(args, key)
    manifest_path = args.manifest or get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("give --manifest (or data.manifest in the config)")
    out = Path(args.out)
    log = RunLog(out)
    seed_everything(int(tc.get("seed", 0)))
    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision or tc.get("precision", "auto"))
    batch_size, accum = int(tc.get("batch_size", 32)), max(1, int(tc.get("grad_accum", 1)))
    workers = int(args.workers if args.workers is not None else get(config, "data.num_workers", 4))
    dataset, loader, val_loader = make_loaders(config, manifest_path, batch_size, workers, device.type == "cuda")

    kwargs = resolve_tier(config, args.tier)
    kwargs.update(in_channels=dataset.channels, num_stem_classes=dataset.num_classes)
    model = build_dit(kwargs, dataset.channels, dataset.num_classes).to(device)
    ema = ModelEMA(model, float(tc.get("ema_decay", 0.9999)), warmup=bool(tc.get("ema_warmup", True)))
    flow = config.get("flow", {})
    loss_fn = MaskedOTFlowMatcher(float(flow.get("sigma_min", 1e-5)), flow.get("time_sampling", "uniform"),
                                  float(flow.get("logit_mean", 0.0)), float(flow.get("logit_std", 1.0)))
    label_dropout = float(flow.get("label_dropout", 0.1))
    optimizer, scheduler = make_optimizer(model.parameters(), float(tc.get("lr", 1e-4)), float(tc.get("weight_decay", 0.01)),
                                          int(tc.get("warmup_steps", 0)))

    step, best = 0, None
    resume = Path(args.resume) if args.resume else (out / "last.pt" if (out / "last.pt").is_file() and not args.fresh else None)
    if resume:
        state = load_checkpoint(resume)
        if state["model_kwargs"] != kwargs:
            raise SystemExit(f"{resume} has a different model configuration; use --fresh or another --out")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        amp.scaler.load_state_dict(state["scaler"])
        step, best = int(state["step"]), state.get("best")
        ema.updates = step
        print(f"resumed from {resume} at step {step}")

    sampling = config.get("sampling", {})
    meta = dict(num_steps=int(sampling.get("num_steps", 50)), cfg_scale=float(sampling.get("cfg_scale", 1.0)))
    sources: Dict[str, int] = {}
    for entry in dataset.entries:
        sources[entry["source"]] = sources.get(entry["source"], 0) + 1
    print(f"Stage 2 | tier {tier_name(config, args.tier)} | {count_parameters(model) / 1e6:.2f}M parameters | "
          f"{amp.describe()} | device {device} | latents {dataset.channels} x {dataset.segment_frames} frames | "
          f"{dataset.num_classes - 1} classes, {dataset.max_stems} slots | train entries {sources} | "
          f"validation {'yes' if val_loader else 'none'}")

    max_steps = int(tc.get("max_steps", 300000))
    grad_clip = float(tc.get("grad_clip", 1.0))
    log_every, valid_every, save_every = int(tc.get("log_every", 50)), int(tc.get("valid_every", 2000)), int(tc.get("save_every", 5000))
    batches = cycle(loader)
    model.train()
    running, seen, t0 = 0.0, 0, time.time()
    while step < max_steps:
        for _ in range(accum):
            parent, stems, labels, valid = (x.to(device, non_blocking=True) for x in next(batches))
            labels = drop_labels(labels, label_dropout)
            with amp.autocast():
                loss = loss_fn(model, stems, parent, labels, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
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
        if val_loader is not None and (step % valid_every == 0 or step == max_steps):
            value = validation_loss(loss_fn, ema.module, val_loader, device, amp, int(tc.get("valid_batches", 20)))
            log("valid", step, flow_loss=value)
            if best is None or value < best["flow_loss"]:
                best = {"step": step, "flow_loss": value}
                save_checkpoint(out / "best.pt", stage2_payload(model, ema, kwargs, dataset, step, config, **meta, extra={"best": best}))
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(out / "last.pt", stage2_payload(model, ema, kwargs, dataset, step, config, optimizer=optimizer,
                                                            scheduler=scheduler, amp=amp, **meta, extra={"best": best}))
    print(f"done: {out / 'last.pt'}")
    return out / "last.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dit_latent.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--fresh", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
