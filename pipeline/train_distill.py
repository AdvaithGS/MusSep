"""Stage 4: consistency distillation of the Stage 2 DiT (50 Euler steps -> 1-4).

    python -m pipeline.train_distill --config configs/distillation.yaml \
        --teacher runs/stage2/best.pt --manifest /data/latents/manifest.json --out runs/stage4

The student has the teacher's architecture and starts from its weights. Each
step it learns to map a point on the teacher's ODE trajectory and the point
one Euler step later to the same clean latent (Pseudo-Huber distance in latent
space, EMA-student target; pipeline/distillation.py). The result is written in
the Stage 2 checkpoint format with ``distilled = true``, so the inference code
picks the consistency sampler by itself.

Validation reports the consistency loss on held-out songs and, every
``valid_every`` steps, the latent MSE between the student's few-step samples
and the teacher's 50-step samples from the same noise: a direct measure of
how much the shortcut costs, without decoding audio.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data.taxonomy import Taxonomy
from models.dit import JointSubStemDiT
from pipeline.distillation import ConsistencyFlowDistiller
from pipeline.flow_matching import ODESolver
from pipeline.infer import STAGE2_KIND
from pipeline.train_generative import make_loaders, stage2_payload
from utils.config import get, load_yaml
from utils.runtime import CheckpointClock, configure_backends, configure_threads, resolve_workers
from utils.training import (AmpContext, RunLog, count_parameters, cycle, load_checkpoint, make_optimizer,
                            save_checkpoint, seed_everything, select_device)


@torch.no_grad()
def evaluate(distiller, loader, device, amp, max_batches: int, teacher_steps: int, student_steps: int, seed: int = 1234):
    """(consistency loss, latent MSE student vs teacher samples) on held-out data."""
    loss_total, mse_total, count = 0.0, 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index, (parent, stems, labels, valid) in enumerate(loader):
            if index >= max_batches:
                break
            parent, stems, labels, valid = (x.to(device) for x in (parent, stems, labels, valid))
            with amp.autocast():
                loss_total += float(distiller(stems, parent, labels, valid))
                noise = torch.randn_like(stems)
                teacher = ODESolver.sample_euler(distiller.teacher, labels, parent, num_steps=teacher_steps,
                                                 valid_mask=valid, noise=noise, cfg_scale=distiller.teacher_cfg_scale)
                student = ODESolver.sample_consistency(distiller.ema_student, labels, parent, num_steps=student_steps,
                                                       valid_mask=valid, noise=noise)
            per_slot = (student - teacher).pow(2).mean(dim=(2, 3))
            mse_total += float((per_slot * valid).sum() / valid.sum().clamp_min(1))
            count += 1
    return loss_total / max(1, count), mse_total / max(1, count)


def train(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    tc, dc = config.get("training", {}), config.get("distill", {})
    for key in ("batch_size", "max_steps", "lr", "grad_accum"):
        if getattr(args, key) is not None:
            tc[key] = getattr(args, key)
    out = Path(args.out)
    log = RunLog(out)
    seed_everything(int(tc.get("seed", 0)))
    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision or tc.get("precision", "auto"))

    teacher_state = load_checkpoint(args.teacher)
    if teacher_state.get("kind") != STAGE2_KIND or teacher_state.get("distilled"):
        raise SystemExit(f"{args.teacher} is not a Stage 2 flow-matching checkpoint")
    kwargs = teacher_state["model_kwargs"]
    use_ema = bool(dc.get("use_teacher_ema", True)) and teacher_state.get("ema") is not None
    teacher = JointSubStemDiT(**kwargs)
    teacher.load_state_dict(teacher_state["ema"] if use_ema else teacher_state["model"])
    student = JointSubStemDiT(**kwargs)
    if dc.get("init_from_teacher", True):
        student.load_state_dict(teacher.state_dict())

    batch_size, accum = int(tc.get("batch_size", 32)), max(1, int(tc.get("grad_accum", 1)))
    workers = resolve_workers(args.workers if args.workers is not None else get(config, "data.num_workers", "auto"), device)
    configure_threads(device, workers)
    configure_backends(bool(tc.get("tf32", True)))
    dataset, loader, val_loader = make_loaders(config, args.manifest, batch_size, workers, device)
    latent = teacher_state["latent"]
    for key in ("vae", "sample_rate", "hop_length", "latent_dim", "channels"):
        if latent[key] != dataset.info[key]:
            raise SystemExit(f"manifest and teacher disagree on {key}: {dataset.info[key]} vs {latent[key]}")
    if dataset.taxonomy.class_names != Taxonomy.from_dict(teacher_state["taxonomy"]).class_names:
        raise SystemExit("manifest and teacher use different taxonomies")
    if not (torch.allclose(latent["mean"], dataset.info["mean"]) and torch.allclose(latent["std"], dataset.info["std"])):
        raise SystemExit("manifest latent statistics differ from the teacher's; distil on the manifest the teacher used")

    distiller = ConsistencyFlowDistiller(
        teacher, student, num_scales=int(dc.get("num_scales", 50)), sigma_min=1e-5,
        teacher_cfg_scale=float(dc.get("teacher_cfg_scale", 1.0)), huber_c=dc.get("huber_c"),
    ).to(device)
    optimizer, scheduler = make_optimizer(distiller.student.parameters(), float(tc.get("lr", 2e-5)),
                                          float(tc.get("weight_decay", 0.0)), int(tc.get("warmup_steps", 0)))
    step = 0
    resume = Path(args.resume) if args.resume else (out / "last.pt" if (out / "last.pt").is_file() and not args.fresh else None)
    if resume:
        state = load_checkpoint(resume)
        distiller.student.load_state_dict(state["model"])
        distiller.ema_student.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        amp.scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        print(f"resumed from {resume} at step {step}")

    student_steps = int(get(config, "sampling.num_steps", 2))
    target_decay = float(dc.get("target_ema_decay", 0.999))
    print(f"Stage 4 | student {count_parameters(student) / 1e6:.2f}M parameters | N = {distiller.num_scales} | "
          f"{amp.describe()} | device {device} | teacher {'EMA ' if use_ema else ''}weights from {args.teacher}")

    max_steps = int(tc.get("max_steps", 50000))
    grad_clip = float(tc.get("grad_clip", 1.0))
    log_every, valid_every = int(tc.get("log_every", 50)), int(tc.get("valid_every", 2000))
    clock = CheckpointClock(int(tc.get("save_every", 5000)), float(tc.get("save_every_minutes", 20)))
    distiller.compile_student(bool(args.compile or tc.get("compile", False)))
    meta = dict(distilled=True, num_steps=student_steps, cfg_scale=1.0)
    teacher_info = {"teacher": str(args.teacher), "teacher_step": teacher_state.get("step")}
    batches = cycle(loader)
    distiller.train()
    running, seen, t0 = 0.0, 0, time.time()
    while step < max_steps:
        for _ in range(accum):
            parent, stems, labels, valid = (x.to(device, non_blocking=True) for x in next(batches))
            with amp.autocast():
                loss = distiller(stems, parent, labels, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            amp.backward(loss / accum)
            running += float(loss.detach())
            seen += 1
        grad_norm = amp.step(optimizer, distiller.student.parameters(), grad_clip)
        scheduler.step()
        distiller.update_ema(target_decay)
        step += 1

        if step % log_every == 0:
            log("train", step, loss=running / seen, grad_norm=grad_norm, lr=scheduler.get_last_lr()[0],
                sec_per_step=(time.time() - t0) / log_every)
            running, seen, t0 = 0.0, 0, time.time()
        if val_loader is not None and (step % valid_every == 0 or step == max_steps):
            consistency, mse = evaluate(distiller, val_loader, device, amp, int(tc.get("valid_batches", 20)),
                                        distiller.num_scales, student_steps)
            log("valid", step, consistency_loss=consistency, mse_vs_teacher=mse)
        if clock.due(step) or step == max_steps:
            save_checkpoint(out / "last.pt", stage2_payload(
                distiller.student, distiller.ema_student, kwargs, dataset, step, config, optimizer=optimizer,
                scheduler=scheduler, amp=amp, **meta, extra=teacher_info))
            clock.saved()
        if clock.stop_requested:
            print(f"stopped at step {step}; resume with the same command")
            break
    clock.close()
    print(f"done: {out / 'last.pt'} (use with --stage2; sampling defaults to {student_steps} consistency steps)")
    return out / "last.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/distillation.yaml")
    parser.add_argument("--teacher", required=True, help="Stage 2 checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", help="data-loader workers, or 'auto'")
    parser.add_argument("--compile", action="store_true", help="torch.compile the student's training forward pass")
    parser.add_argument("--resume")
    parser.add_argument("--fresh", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
