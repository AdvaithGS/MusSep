"""Shared fixtures: tiny synthetic datasets laid out like the real ones.

Nothing here needs network access, real datasets or the real DAC-VAE; the
end-to-end tests use ``models.dac_vae.ToyVAE`` ("toy:8000:80:8").
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 44100
TOY_VAE = "toy:8000:80:8"


def tone(freq: float, seconds: float, sr: int = SR, channels: int = 2, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    """A decaying tone plus a little noise, (frames, channels) float32."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = 0.6 + 0.4 * np.cos(2 * math.pi * 1.5 * t) ** 2
    mono = amp * env * np.sin(2 * math.pi * freq * t) + 0.01 * amp * rng.standard_normal(len(t))
    pan = np.array([0.8, 1.0][:channels])
    return (mono[:, None] * pan[None, :]).astype(np.float32)


def write(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


# --------------------------------------------------------------------------- MUSDB18-HQ

MUSDB_TRAIN = ["Alpha - One", "Beta - Two", "Gamma - Three"]
MUSDB_VALID = ["Actions - One Minute Smile", "Leaf - Summerghost"]
MUSDB_TEST = ["Delta - Four"]


@pytest.fixture(scope="session")
def musdb_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("musdb18hq")
    freqs = {"vocals": 440.0, "bass": 55.0, "drums": 2000.0, "other": 660.0}
    for subset, names in (("train", MUSDB_TRAIN + MUSDB_VALID), ("test", MUSDB_TEST)):
        for k, name in enumerate(names):
            stems = {s: tone(f * (1 + 0.05 * k), 3.0, seed=k) for s, f in freqs.items()}
            for s, a in stems.items():
                write(root / subset / name / f"{s}.wav", a)
            write(root / subset / name / "mixture.wav", sum(stems.values()))
    return root


# --------------------------------------------------------------------------- MoisesDB

# (stemName, trackType, frequency)
MOISES_LAYOUT = [
    ("vocals", "lead male singer", 330.0),
    ("vocals", "background vocals", 495.0),
    ("drums", "kick drum", 60.0),
    ("drums", "snare drum", 220.0),
    ("drums", "hi-hat", 3000.0),
    ("drums", "overheads", 1500.0),
    ("bass", "bass guitar", 70.0),
    ("guitar", "clean electric guitar", 196.0),
    ("piano", "grand piano", 262.0),
    ("other", "click track", 1000.0),
]


@pytest.fixture(scope="session")
def moisesdb_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("moisesdb")
    for n in range(8):
        track_id = f"track-{n:02d}"
        track = root / "provider" / track_id
        stems = {}
        for i, (stem, kind, freq) in enumerate(MOISES_LAYOUT):
            if stem == "piano" and n % 2:           # some songs have no piano
                continue
            file_id = f"{stem}-{i}"
            write(track / stem / f"{file_id}.wav", tone(freq * (1 + 0.03 * n), 2.5, seed=n * 31 + i, amp=0.08))
            stems.setdefault(stem, []).append({"id": file_id, "trackType": kind, "extension": "wav", "has_bleed": False})
        meta = {"song": f"song {n}", "artist": "test", "stems": [{"stemName": s, "tracks": t} for s, t in stems.items()]}
        (track / "data.json").write_text(json.dumps(meta))
    return root


# --------------------------------------------------------------------------- Slakh2100

@pytest.fixture(scope="session")
def slakh_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("slakh")
    programs = [(0, False, 262.0), (25, False, 196.0), (33, False, 55.0), (0, True, 2000.0), (73, False, 880.0)]
    for split in ("train", "validation"):
        for n in range(2):
            track = root / split / f"Track{n:05d}"
            meta = {"stems": {}}
            for i, (program, drum, freq) in enumerate(programs):
                sid = f"S{i:02d}"
                write(track / "stems" / f"{sid}.flac", tone(freq, 2.0, channels=1, seed=i)[:, 0])
                meta["stems"][sid] = {"inst_class": "x", "program_num": program, "is_drum": drum, "audio_rendered": True}
            meta["stems"]["S99"] = {"program_num": 40, "is_drum": False, "audio_rendered": False}
            track.mkdir(parents=True, exist_ok=True)
            (track / "metadata.yaml").write_text(yaml.safe_dump(meta))
    return root


def tmp_config(source: str, path: Path, **sections) -> Path:
    """Copy a repository config with nested overrides, e.g. training={"max_steps": 2}."""
    from utils.config import load_yaml

    config = load_yaml(ROOT / source)
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    path.write_text(yaml.safe_dump(config))
    return path
