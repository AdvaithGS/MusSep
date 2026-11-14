"""Waveform augmentations used by the Stage 1 random-mix pipeline and by the
latent pre-encoding script. All functions take and return ``(C, T)`` tensors."""
from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from utils.audio import db_to_amp, rms_db


def random_gain_db(low_db: float, high_db: float, rng: Optional[random.Random] = None) -> float:
    rng = rng or random
    return rng.uniform(low_db, high_db)


def apply_gain(audio: torch.Tensor, gain_db: float) -> torch.Tensor:
    return audio * float(db_to_amp(gain_db))


def random_gain(
    audio: torch.Tensor,
    gain_range_db: Tuple[float, float] = (-3.0, 3.0),
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Scales by a gain drawn uniformly in dB from ``gain_range_db``."""
    return apply_gain(audio, random_gain_db(gain_range_db[0], gain_range_db[1], rng))


def channel_swap(audio: torch.Tensor) -> torch.Tensor:
    """Swaps left and right (no-op for mono)."""
    return audio.flip(0) if audio.shape[0] == 2 else audio


def polarity_flip(audio: torch.Tensor) -> torch.Tensor:
    return -audio


def is_silent(audio: torch.Tensor, threshold_db: float = -50.0) -> bool:
    """True if the RMS level of the excerpt is below ``threshold_db`` dBFS."""
    return rms_db(audio) < threshold_db
