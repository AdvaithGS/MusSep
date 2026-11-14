"""Random-mix augmentation for Stage 1 (BS-RoFormer) training.

Each training example is assembled stem by stem: every stem is cut from a
randomly chosen song at a random offset, re-drawn if the excerpt is (nearly)
silent, scaled by a random gain and occasionally replaced by silence. The
mixture is the sum of the stems. This is the augmentation described in the
BS-RoFormer paper (random mixing, +-3 dB gain, 10 % silent stems, -50 dB
loudness gate).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .dataset import read_segment
from .transforms import apply_gain, channel_swap, is_silent


@dataclass
class StemFile:
    path: Path
    frames: int


class RandomMixer:
    """Draws random-mix training examples.

    Args:
        tracks: one dict per song mapping stem name -> :class:`StemFile`.
        stems: every stem that goes into the mixture (e.g. the four MUSDB stems).
        segment_samples: excerpt length.
        random_mix: draw each stem from an independent song/offset. If False all
            stems come from the same song and offset (a coherent excerpt).
        gain_range_db: per-stem random gain; ``None`` disables it.
        silence_prob: probability of replacing a stem by silence.
        loudness_threshold_db: excerpts whose RMS is below this are re-drawn...
        max_retries: ...at most this many times, after which the last draw is kept.
        channel_swap_prob: probability of swapping L/R per stem (off by default;
            not part of the paper's recipe).
    """

    def __init__(
        self,
        tracks: Sequence[Dict[str, StemFile]],
        stems: Sequence[str],
        segment_samples: int,
        sample_rate: int = 44100,
        channels: int = 2,
        random_mix: bool = True,
        gain_range_db: Optional[Tuple[float, float]] = (-3.0, 3.0),
        silence_prob: float = 0.1,
        loudness_threshold_db: float = -50.0,
        max_retries: int = 10,
        channel_swap_prob: float = 0.0,
    ):
        if not tracks:
            raise ValueError("RandomMixer needs at least one track")
        self.tracks = list(tracks)
        self.stems = tuple(stems)
        self.segment_samples = int(segment_samples)
        self.sample_rate = sample_rate
        self.channels = channels
        self.random_mix = random_mix
        self.gain_range_db = gain_range_db
        self.silence_prob = silence_prob
        self.loudness_threshold_db = loudness_threshold_db
        self.max_retries = max_retries
        self.channel_swap_prob = channel_swap_prob

    # -- helpers ----------------------------------------------------------

    def _random_offset(self, frames: int, rng: random.Random) -> int:
        return rng.randint(0, max(0, frames - self.segment_samples))

    def _read(self, stem_file: StemFile, offset: int) -> torch.Tensor:
        return read_segment(stem_file.path, offset, self.segment_samples, self.channels, self.sample_rate)

    def _draw_stem(self, stem: str, rng: random.Random) -> torch.Tensor:
        audio = torch.zeros(self.channels, self.segment_samples)
        for _ in range(max(1, self.max_retries)):
            stem_file = rng.choice(self.tracks)[stem]
            audio = self._read(stem_file, self._random_offset(stem_file.frames, rng))
            if not is_silent(audio, self.loudness_threshold_db):
                break
        return audio

    def _draw_coherent(self, rng: random.Random) -> Dict[str, torch.Tensor]:
        chosen: Dict[str, torch.Tensor] = {}
        for _ in range(max(1, self.max_retries)):
            track = rng.choice(self.tracks)
            offset = self._random_offset(min(track[stem].frames for stem in self.stems), rng)
            chosen = {stem: self._read(track[stem], offset) for stem in self.stems}
            if not is_silent(sum(chosen.values()), self.loudness_threshold_db):
                break
        return chosen

    # -- public -----------------------------------------------------------

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, torch.Tensor]:
        """Returns ``{stem: (channels, segment_samples)}`` after augmentation."""
        rng = rng or random
        if self.random_mix:
            stems = {stem: self._draw_stem(stem, rng) for stem in self.stems}
        else:
            stems = self._draw_coherent(rng)

        for stem in self.stems:
            audio = stems[stem]
            if self.silence_prob > 0 and rng.random() < self.silence_prob:
                stems[stem] = torch.zeros_like(audio)
                continue
            if self.gain_range_db is not None:
                audio = apply_gain(audio, rng.uniform(*self.gain_range_db))
            if self.channel_swap_prob > 0 and rng.random() < self.channel_swap_prob:
                audio = channel_swap(audio)
            stems[stem] = audio
        return stems


def mix(stems: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Linear sum of all stems."""
    return torch.stack(list(stems.values())).sum(dim=0)


def stack_targets(stems: Dict[str, torch.Tensor], targets: Sequence[str]) -> torch.Tensor:
    """``(len(targets), C, T)`` in the order given by ``targets``."""
    return torch.stack([stems[name] for name in targets])
