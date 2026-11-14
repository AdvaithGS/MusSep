"""Shared building blocks for the dataset modules.

* :func:`read_segment` - random-access read of a fixed-length audio excerpt.
* :class:`HierTrack` - one multitrack song whose source files are labelled
  with ``parent stem`` and candidate ``sub-stem class``; produced by the
  MoisesDB and Slakh indexers and consumed by ``scripts/taxonomy_stats.py``
  and ``scripts/preprocess_latents.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from utils.audio import load_audio

if TYPE_CHECKING:  # pragma: no cover
    from .taxonomy import Taxonomy

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")


def read_segment(
    path: str | Path,
    start: int,
    frames: int,
    channels: int = 2,
    sample_rate: Optional[int] = None,
) -> torch.Tensor:
    """Reads ``frames`` samples starting at ``start`` as float32 ``(channels, frames)``.

    The excerpt is zero-padded if the file ends early. ``sample_rate``, when
    given, is only *checked* (datasets used for Stage 1 are expected to be
    stored at the training rate; resampling random crops on the fly would be
    slow and would shift the crop boundaries).
    """
    audio, file_sr = load_audio(path, channels=channels, start=max(0, start), frames=frames)
    if sample_rate is not None and file_sr != sample_rate:
        raise ValueError(f"{path}: sample rate {file_sr} != expected {sample_rate}")
    if audio.shape[-1] < frames:
        audio = F.pad(audio, (0, frames - audio.shape[-1]))
    return audio


def load_and_sum(
    paths: Sequence[str | Path],
    sample_rate: int,
    channels: int = 2,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Loads several files, resamples to ``sample_rate`` and sums them.

    Returns ``(channels, T)`` where ``T`` is ``length`` if given, otherwise the
    longest file. Shorter files are zero-padded, longer ones truncated.
    """
    signals = [load_audio(path, sample_rate=sample_rate, channels=channels)[0] for path in paths]
    if length is None:
        length = max((signal.shape[-1] for signal in signals), default=0)
    total = torch.zeros(channels, length)
    for signal in signals:
        usable = min(length, signal.shape[-1])
        total[:, :usable] += signal[:, :usable]
    return total


@dataclass
class SourceFile:
    path: Path
    parent: str
    raw_class: Optional[str]      # candidate class (before merges); None if the parent is not split
    label: str                    # where it came from, e.g. "drums / kick drum"
    matched: bool = True          # False if a default rule decided the class


@dataclass
class HierTrack:
    """A multitrack song organised by the two-level MusSep taxonomy."""

    track_id: str
    source: str                   # "moisesdb" | "slakh"
    path: Path
    split: str = "train"          # train | val | test
    files: List[SourceFile] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)   # labels dropped by an exclude rule
    missing: List[str] = field(default_factory=list)    # audio files listed in metadata but absent

    def parent_files(self, parent: str) -> List[Path]:
        """Every file that belongs to ``parent`` (their sum is the parent stem)."""
        return [item.path for item in self.files if item.parent == parent]

    def other_parent_files(self, parent: str) -> List[Path]:
        return [item.path for item in self.files if item.parent != parent]

    def raw_groups(self, parent: str) -> Dict[str, List[Path]]:
        groups: Dict[str, List[Path]] = {}
        for item in self.files:
            if item.parent == parent and item.raw_class is not None:
                groups.setdefault(item.raw_class, []).append(item.path)
        return groups

    def groups(self, parent: str, taxonomy: "Taxonomy") -> Dict[str, List[Path]]:
        """``{modelled class: [files]}`` for ``parent`` with merges applied,
        in the taxonomy's canonical class order."""
        merged: Dict[str, List[Path]] = {}
        for raw_class, paths in self.raw_groups(parent).items():
            resolved = taxonomy.resolve(raw_class)
            if resolved is not None:
                merged.setdefault(resolved, []).extend(paths)
        return {name: merged[name] for name in taxonomy.parent_classes(parent) if name in merged}
