"""Runtime settings for cheap and interruptible GPU machines.

* CPU balance. Data-loader workers are sized from the CPUs this process may
  actually use (container limits included), and every process is kept to a
  sensible number of intra-op threads: by default each worker would start one
  thread per core for its resampling and STFT work, and N workers x N threads
  oversubscribe the machine, which starves the process feeding the GPU.
* Backends. TF32 matmuls on Ampere and newer GPUs, and an optional
  ``torch.compile`` of the training forward pass.
* Checkpoint cadence. Save every N steps *or* every M minutes, and save once
  more and stop cleanly on SIGTERM / SIGINT (what spot instances, Colab and
  Kaggle send before they take the machine away).
"""
from __future__ import annotations

import os
import signal
import time
import warnings
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# --------------------------------------------------------------------------- CPUs and threads


def available_cpus() -> int:
    """CPUs this process may run on (respects taskset / cgroup affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - not Linux
        return max(1, os.cpu_count() or 1)


def resolve_workers(value: Any, device: torch.device, cap: int = 12) -> int:
    """Number of data-loader workers. ``"auto"`` / ``None``:

    * GPU training: every CPU but two (one for the main process, which launches
      kernels and runs the loss bookkeeping, one for the system), up to ``cap``;
    * CPU training: a quarter of the CPUs, since the model itself needs the rest.
    """
    if value not in (None, "auto"):
        return max(0, int(value))
    cpus = available_cpus()
    if device.type == "cuda":
        return max(0, min(cap, cpus - 2))
    return max(0, min(cap, cpus // 4))


def configure_threads(device: torch.device, workers: int) -> int:
    """Intra-op threads of the main process: what is left after the workers."""
    cpus = available_cpus()
    threads = max(1, cpus - workers) if device.type != "cuda" else max(1, min(4, cpus - workers))
    torch.set_num_threads(threads)
    return threads


def worker_init(_: int) -> None:
    """One intra-op thread per data-loader worker: parallelism comes from the workers."""
    torch.set_num_threads(1)


def make_loader(
    dataset: Dataset,
    batch_size: Optional[int],
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    sampler: Optional[Sampler] = None,
    drop_last: bool = False,
    prefetch_factor: int = 4,
    persistent: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    kwargs: dict = dict(batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
                        num_workers=workers, drop_last=drop_last, pin_memory=device.type == "cuda")
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if workers > 0:
        kwargs.update(worker_init_fn=worker_init, prefetch_factor=max(1, int(prefetch_factor)),
                      persistent_workers=persistent)
    return DataLoader(dataset, **kwargs)


def available_memory_bytes() -> Optional[int]:
    """Memory the system can give us now (MemAvailable on Linux)."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return None


# --------------------------------------------------------------------------- backends and compile


def configure_backends(tf32: bool = True) -> None:
    """TF32 tensor cores for float32 matmuls and convolutions (Ampere and newer).

    It only affects work that is still float32 under autocast; the STFTs and
    losses that are forced to float32 for accuracy are not matmul-bound.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = True       # fixed input shapes: pick the fastest kernels once
    if tf32:
        torch.set_float32_matmul_precision("high")


def maybe_compile(module: torch.nn.Module, enabled: bool) -> Callable:
    """``torch.compile(module)`` if enabled, else the module itself.

    Use the returned callable for the training forward pass and keep using
    ``module`` for state dicts, EMA and evaluation (a compiled module prefixes
    its state-dict keys). The first steps are slow while it compiles; measure
    the gain with scripts/benchmark.py before relying on it.
    """
    if not enabled:
        return module
    try:
        return torch.compile(module)
    except Exception as error:  # pragma: no cover - depends on the platform
        warnings.warn(f"torch.compile unavailable ({error}); running uncompiled")
        return module


# --------------------------------------------------------------------------- checkpoint clock


class CheckpointClock:
    """Decides when to save, and turns SIGTERM / SIGINT into a clean stop.

    ``due(step)`` is true every ``every_steps`` steps, after ``every_minutes``
    minutes without a save, or once a stop was requested. The first signal
    only sets ``stop_requested`` (the loop saves and exits at the end of the
    current step); a second one interrupts immediately.
    """

    def __init__(self, every_steps: int = 0, every_minutes: float = 0.0, handle_signals: bool = True):
        self.every_steps = int(every_steps)
        self.every_seconds = float(every_minutes) * 60.0
        self.last_save = time.monotonic()
        self.stop_requested = False
        self._previous = {}
        if handle_signals:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._previous[sig] = signal.signal(sig, self._on_signal)
                except ValueError:  # not the main thread
                    pass

    def _on_signal(self, signum, frame) -> None:
        if self.stop_requested:
            previous = self._previous.get(signum)
            self.close()
            if callable(previous):
                previous(signum, frame)
            raise KeyboardInterrupt
        self.stop_requested = True
        print(f"\nsignal {signum}: saving at the end of this step, then stopping "
              f"(send it again to stop immediately)", flush=True)

    def due(self, step: int) -> bool:
        if self.stop_requested:
            return True
        if self.every_steps > 0 and step % self.every_steps == 0:
            return True
        return self.every_seconds > 0 and time.monotonic() - self.last_save >= self.every_seconds

    def saved(self) -> None:
        self.last_save = time.monotonic()

    def close(self) -> None:
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError):
                pass
        self._previous = {}
