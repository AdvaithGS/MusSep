"""MUSDB18-HQ datasets for Stage 1.

Expected layout (the official MUSDB18-HQ distribution, uncompressed wav)::

    <root>/train/<track name>/{mixture,vocals,bass,drums,other}.wav   (100 tracks)
    <root>/test/<track name>/{mixture,vocals,bass,drums,other}.wav    ( 50 tracks)

The 86 / 14 train / validation split of the training set is the one defined by
the ``musdb`` package (``musdb.DB(subsets='train', split='valid')``).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils.audio import audio_info, load_audio

from .mix_pipeline import RandomMixer, StemFile, mix, stack_targets

MUSDB_STEMS: Tuple[str, ...] = ("vocals", "bass", "drums", "other")

# Validation tracks of the official `musdb` split.
MUSDB_VALIDATION_TRACKS: Tuple[str, ...] = (
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
)

SPLITS = ("train", "valid", "train_full", "test")


def index_musdb(root: str | Path, split: str = "train") -> List[Dict]:
    """Lists the tracks of a split.

    ``split``: ``train`` (86 tracks), ``valid`` (14), ``train_full`` (100) or
    ``test`` (50). Each entry is ``{"name", "path", "stems": {stem: StemFile}}``.
    Folders that miss one of the four stems are skipped with a warning.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    root = Path(root)
    subset_dir = root / ("test" if split == "test" else "train")
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"MUSDB18-HQ folder not found: {subset_dir}")

    tracks: List[Dict] = []
    for track_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
        name = track_dir.name
        if split == "train" and name in MUSDB_VALIDATION_TRACKS:
            continue
        if split == "valid" and name not in MUSDB_VALIDATION_TRACKS:
            continue
        stems: Dict[str, StemFile] = {}
        for stem in MUSDB_STEMS:
            path = track_dir / f"{stem}.wav"
            if not path.is_file():
                break
            stems[stem] = StemFile(path=path, frames=audio_info(path)[0])
        else:
            tracks.append({"name": name, "path": track_dir, "stems": stems})
            continue
        print(f"[musdb] skipping '{name}': missing {stem}.wav")
    if not tracks:
        raise RuntimeError(f"no usable tracks found in {subset_dir} for split '{split}'")
    return tracks


class MUSDBRandomMixDataset(Dataset):
    """Random-mix training examples.

    ``__getitem__`` returns ``(mixture, targets)`` with shapes ``(2, T)`` and
    ``(len(target_stems), 2, T)``. The mixture is always the sum of all four
    (augmented) stems, whatever ``target_stems`` is.

    The dataset is "infinite": ``epoch_size`` only defines ``len()``. Random
    draws use Python's ``random`` module, which PyTorch seeds differently in
    every DataLoader worker.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        target_stems: Sequence[str] = ("vocals",),
        segment_samples: int = 352800,
        sample_rate: int = 44100,
        epoch_size: int = 10000,
        random_mix: bool = True,
        gain_range_db: Optional[Tuple[float, float]] = (-3.0, 3.0),
        silence_prob: float = 0.1,
        loudness_threshold_db: float = -50.0,
        max_retries: int = 10,
        channel_swap_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        unknown = [stem for stem in target_stems if stem not in MUSDB_STEMS]
        if unknown:
            raise ValueError(f"unknown target stems {unknown}; choose from {MUSDB_STEMS}")
        self.target_stems = tuple(target_stems)
        self.epoch_size = int(epoch_size)
        self.tracks = index_musdb(root, split)
        self.mixer = RandomMixer(
            [track["stems"] for track in self.tracks],
            stems=MUSDB_STEMS,
            segment_samples=segment_samples,
            sample_rate=sample_rate,
            random_mix=random_mix,
            gain_range_db=tuple(gain_range_db) if gain_range_db is not None else None,
            silence_prob=silence_prob,
            loudness_threshold_db=loudness_threshold_db,
            max_retries=max_retries,
            channel_swap_prob=channel_swap_prob,
        )
        # A fixed seed makes item `index` reproducible (used by the tests).
        self.seed = seed

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed * 1_000_003 + index) if self.seed is not None else None
        stems = self.mixer.sample(rng)
        return mix(stems), stack_targets(stems, self.target_stems)


class MUSDBTrackDataset(Dataset):
    """Whole tracks for validation / testing.

    Items are dicts ``{"name", "mixture": (2, T), "targets": {stem: (2, T)}}``.
    Use it with ``DataLoader(batch_size=None)`` or iterate over it directly.
    ``max_seconds`` optionally keeps only a centred excerpt (quick validation).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "valid",
        stems: Sequence[str] = MUSDB_STEMS,
        sample_rate: int = 44100,
        max_seconds: Optional[float] = None,
        max_tracks: Optional[int] = None,
    ):
        self.tracks = index_musdb(root, split)
        if max_tracks is not None:
            self.tracks = self.tracks[:max_tracks]
        self.stems = tuple(stems)
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> Dict:
        track = self.tracks[index]
        frames = min(stem_file.frames for stem_file in track["stems"].values())
        start, count = 0, frames
        if self.max_seconds is not None:
            count = min(frames, int(self.max_seconds * self.sample_rate))
            start = (frames - count) // 2
        all_stems = {
            stem: self._read(track["stems"][stem].path, start, count) for stem in MUSDB_STEMS
        }
        # The official mixture is the sum of the four stems; summing avoids a fifth read.
        mixture = torch.stack(list(all_stems.values())).sum(dim=0)
        return {
            "name": track["name"],
            "mixture": mixture,
            "targets": {stem: all_stems[stem] for stem in self.stems},
        }

    def _read(self, path: Path, start: int, count: int) -> torch.Tensor:
        audio, file_sr = load_audio(path, channels=2, start=start, frames=count)
        if file_sr != self.sample_rate:
            raise ValueError(f"{path}: sample rate {file_sr} != expected {self.sample_rate}")
        return audio
