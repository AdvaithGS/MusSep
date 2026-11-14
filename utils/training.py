"""Training utilities shared by the three training scripts."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: Optional[str] = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    seen = set()
    total = 0
    for param in module.parameters():
        if id(param) in seen or (trainable_only and not param.requires_grad):
            continue
        seen.add(id(param))
        total += param.numel()
    return total


def trainable_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    return (param for param in module.parameters() if param.requires_grad)


def cycle(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class AmpContext:
    """Mixed-precision policy: bf16 when natively supported, else fp16 + GradScaler.

    * CUDA with native bfloat16 (Ampere and newer): autocast to bf16, no scaler.
    * Other CUDA devices: autocast to fp16 with a ``GradScaler``.
    * CPU / MPS or ``enabled=False``: plain float32.

    Losses, STFT/iSTFT and VAE decoding are forced to float32 by the modules
    that own them, regardless of this context.
    """

    def __init__(self, device: torch.device, enabled: bool = True, precision: str = "auto"):
        self.device = device
        self.enabled = bool(enabled) and device.type == "cuda" and precision != "fp32"
        self.dtype = torch.float32
        use_scaler = False
        if self.enabled:
            if precision == "bf16":
                use_bf16 = True
            elif precision == "fp16":
                use_bf16 = False
            else:
                use_bf16 = self._native_bf16()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float16
            use_scaler = not use_bf16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    @staticmethod
    def _native_bf16() -> bool:
        try:
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:  # older PyTorch without the keyword
            return bool(torch.cuda.is_bf16_supported())

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype if self.enabled else None,
                              enabled=self.enabled)

    def describe(self) -> str:
        if not self.enabled:
            return "fp32 (autocast off)"
        name = "bf16" if self.dtype == torch.bfloat16 else "fp16"
        return f"{name} autocast, GradScaler {'on' if self.scaler.is_enabled() else 'off'}"

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
        grad_clip: Optional[float] = None,
    ) -> float:
        """Backward + (unscale, clip) + optimizer step. Returns the gradient norm."""
        self.backward(loss)
        return self.step(optimizer, parameters, grad_clip)

    def backward(self, loss: torch.Tensor) -> None:
        """Backward only (use with :meth:`step` for gradient accumulation)."""
        self.scaler.scale(loss).backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
        grad_clip: Optional[float] = None,
    ) -> float:
        """(unscale, clip) + optimizer step + zero_grad. Returns the gradient norm."""
        self.scaler.unscale_(optimizer)
        max_norm = grad_clip if grad_clip and grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=max_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)


class ModelEMA:
    """Exponential moving average of a model's floating-point state.

    With ``warmup`` the decay after ``n`` updates is ``min(decay, (1 + n) / (10 + n))``:
    the average covers roughly the last ``n / 10`` steps early on and reaches the
    configured decay later (0.999 after ~9k updates, 0.9999 after ~90k). Without
    it, a short run's EMA would still be mostly the random initialisation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = False):
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, decay: Optional[float] = None) -> None:
        if decay is None:
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates)) if self.warmup else self.decay
        self.updates += 1
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            target = ema_state[name]
            if target.dtype.is_floating_point:
                target.mul_(decay).add_(value.detach().to(target.dtype), alpha=1.0 - decay)
            else:
                target.copy_(value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state)


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """Atomic ``torch.save`` (write to a temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Loads a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=True`` restricts unpickling to tensors and plain containers,
    which is all our checkpoints hold.
    """
    return torch.load(str(path), map_location=map_location, weights_only=True)


def warmup_lambda(warmup_steps: int):
    def fn(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return fn


class LRSchedule:
    """Linear warmup, then ``decay ** (step // decay_every)`` (step decay).

    ``decay_every <= 0`` keeps the rate constant after warmup. Picklable (a
    class, not a closure), so a ``LambdaLR`` built on it can be checkpointed.
    """

    def __init__(self, warmup_steps: int = 0, decay: float = 1.0, decay_every: int = 0):
        self.warmup_steps, self.decay, self.decay_every = int(warmup_steps), float(decay), int(decay_every)

    def __call__(self, step: int) -> float:
        factor = min(1.0, (step + 1) / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        if self.decay_every > 0:
            factor *= self.decay ** (step // self.decay_every)
        return factor


def make_optimizer(
    parameters: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    decay: float = 1.0,
    decay_every: int = 0,
):
    """AdamW plus a :class:`LRSchedule`. Returns ``(optimizer, scheduler)``."""
    params = [param for param in parameters if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, LRSchedule(warmup_steps, decay, decay_every))
    return optimizer, scheduler


class RunLog:
    """Prints a line and appends the same record to ``<out>/log.jsonl``."""

    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / "log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, kind: str, step: int, **values: Any) -> None:
        import json
        import time

        record = {"kind": kind, "step": step, "time": round(time.time(), 1), **values}
        shown = "  ".join(
            f"{key} {value:.4g}" if isinstance(value, float) else f"{key} {value}" for key, value in values.items()
        )
        print(f"[{kind}] step {step}  {shown}", flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
