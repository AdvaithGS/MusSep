"""Stage 2 dataset: random crops of pre-encoded latents described by the
manifest written by ``scripts/preprocess_latents.py`` (MoisesDB + Slakh2100
under one taxonomy).

Every item is ``(parent, stems, labels, valid_mask)``:

    parent      (C, T)      conditioning latent of the parent stem
    stems       (S, C, T)   target latents, one slot per sub-stem class
    labels      (S,) long   class id per slot (0 = null / padding)
    valid_mask  (S,) bool   False for padding slots

Latents are standardised with the manifest's per-channel statistics.
``S = taxonomy.max_slots``; slots follow the taxonomy's class order.

Absent classes ("hybrid" policy): with probability ``full_set_prob`` an
example lists *every* class of its parent and the absent ones get silence as
target (so the model learns to output silence for a requested instrument that
is not playing, e.g. backing vocals); otherwise only the present classes are
listed and the remaining slots are padding.

"Present" is decided per crop: a class counts as present if it is active
(above the presence level, see scripts/preprocess_latents.py) in at least
``min_class_active_fraction`` of the crop's frames. A guitar that plays only
in the chorus is absent in verse crops. There its slot is listed only in the
full-set layout, with the encoding of that (silent) stretch as target.
Classes never stored for the song get the manifest's latent of digital silence.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .taxonomy import Taxonomy


def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def latent_info(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The tensors and numbers a checkpoint needs to be usable without the manifest."""
    return {
        "mean": torch.tensor(manifest["latent_stats"]["mean"], dtype=torch.float32),
        "std": torch.tensor(manifest["latent_stats"]["std"], dtype=torch.float32),
        "silence": torch.tensor(manifest["silence_latent"], dtype=torch.float32),
        "vae": manifest["vae"],
        "sample_rate": int(manifest["sample_rate"]),
        "hop_length": int(manifest["hop_length"]),
        "latent_dim": int(manifest["latent_dim"]),
        "channels": int(manifest["channels"]),
    }


class UnifiedManifestDataset(Dataset):
    """
    Args:
        manifest_path: ``manifest.json``; latent paths are relative to its folder.
        split: ``train`` or ``val`` (``test`` exists for Slakh).
        segment_frames: crop length in latent frames.
        full_set_prob: probability of the "all classes, silence for absent" layout.
        sources / parents: optional filters.
        bleed_cond_prob: probability of conditioning on the stored ``cond``
            latent (parent + bleed) instead of the clean parent, when present.
        min_active_fraction / max_retries: crops in which the parent is mostly
            silent are re-drawn.
        min_class_active_fraction: share of a crop's frames a class must be active in
            to count as present in that crop.
        slakh_fraction: target share of Slakh examples among the examples of each
            parent Slakh supplies ("other" only), for :meth:`make_sampler`.
        deterministic: centre crop, always full set (validation).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        segment_frames: int = 250,
        full_set_prob: float = 0.5,
        sources: Optional[Sequence[str]] = None,
        parents: Optional[Sequence[str]] = None,
        bleed_cond_prob: float = 0.5,
        min_active_fraction: float = 0.25,
        min_class_active_fraction: float = 0.05,
        max_retries: int = 8,
        slakh_fraction: float = 1.0 / 3.0,
        deterministic: bool = False,
    ):
        self.root = Path(manifest_path).parent
        self.manifest = load_manifest(manifest_path)
        self.taxonomy = Taxonomy.from_dict(self.manifest["taxonomy"])
        self.info = latent_info(self.manifest)
        self.entries: List[Dict[str, Any]] = [
            entry for entry in self.manifest["entries"]
            if entry["split"] == split
            and (sources is None or entry["source"] in sources)
            and (parents is None or entry["parent"] in parents)
        ]
        if not self.entries:
            raise RuntimeError(f"manifest has no entries for split '{split}' with the given filters")
        self.segment_frames = int(segment_frames)
        self.full_set_prob = full_set_prob
        self.bleed_cond_prob = bleed_cond_prob
        self.min_active_fraction = min_active_fraction
        self.min_class_active_fraction = min_class_active_fraction
        self.max_retries = max_retries
        self.slakh_fraction = slakh_fraction
        self.deterministic = deterministic

        self.max_stems = self.taxonomy.max_slots
        self.num_classes = self.taxonomy.num_classes
        self.channels = len(self.info["mean"])
        self._mean = self.info["mean"][:, None]
        self._std = self.info["std"][:, None].clamp_min(1e-4)
        self._silence = (self.info["silence"][:, None] - self._mean) / self._std     # (C, 1), normalised

    def __len__(self) -> int:
        return len(self.entries)

    # -- sampling ---------------------------------------------------------

    def make_sampler(self, num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """Every real (MoisesDB) entry has weight 1. Slakh entries of a parent
        share weight so that they make up ``slakh_fraction`` of that parent's
        examples. Slakh only supplies "other", so vocals and drums examples are
        all real and Slakh's share is counted within "other" alone (it has ~9x
        as many songs as MoisesDB but is synthetic)."""
        real: Dict[str, int] = {}
        slakh: Dict[str, int] = {}
        for entry in self.entries:
            counts = slakh if entry["source"] == "slakh" else real
            counts[entry["parent"]] = counts.get(entry["parent"], 0) + 1
        f = min(max(self.slakh_fraction, 0.0), 0.99)
        weight = {parent: (real.get(parent, 0) * f / (1.0 - f) / n if real.get(parent) else 1.0)
                  for parent, n in slakh.items()}
        weights = [weight[entry["parent"]] if entry["source"] == "slakh" else 1.0 for entry in self.entries]
        return WeightedRandomSampler(weights, num_samples or len(self.entries), replacement=True)

    def _crop_start(self, total: int, active: Optional[torch.Tensor]) -> int:
        length = self.segment_frames
        if total <= length:
            return 0
        if self.deterministic:
            return (total - length) // 2
        start = random.randint(0, total - length)
        if active is None:
            return start
        for _ in range(self.max_retries):
            if active[start:start + length].float().mean().item() >= self.min_active_fraction:
                break
            start = random.randint(0, total - length)
        return start

    def _crop(self, latents: torch.Tensor, start: int) -> torch.Tensor:
        """Crop ``(..., C, F)`` to the segment, normalise, pad with silence."""
        piece = latents[..., start:start + self.segment_frames].float()
        piece = (piece - self._mean) / self._std
        short = self.segment_frames - piece.shape[-1]
        if short > 0:
            pad = self._silence.expand(*piece.shape[:-1], short)
            piece = torch.cat([piece, pad], dim=-1)
        return piece

    # -- items ------------------------------------------------------------

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[index]
        payload = torch.load(self.root / entry["latent_path"], map_location="cpu", weights_only=True)
        total = payload["parent"].shape[-1]
        start = self._crop_start(total, payload.get("active"))

        cond_source = payload["parent"]
        if "cond" in payload and not self.deterministic and random.random() < self.bleed_cond_prob:
            cond_source = payload["cond"]
        parent = self._crop(cond_source, start)
        stored = self._crop(payload["stems"], start)                         # (P, C, T)
        stored_ids = payload["class_ids"].tolist()
        if "class_active" in payload:
            share = payload["class_active"][:, start:start + self.segment_frames].float().sum(-1) / self.segment_frames
            in_crop = (share >= self.min_class_active_fraction).tolist()
        else:                                                                # manifests without per-frame activity
            in_crop = [True] * len(stored_ids)
        present_ids = [c for c, keep in zip(stored_ids, in_crop) if keep]

        stems = torch.zeros(self.max_stems, self.channels, self.segment_frames)
        labels = torch.zeros(self.max_stems, dtype=torch.long)
        valid = torch.zeros(self.max_stems, dtype=torch.bool)

        if self.deterministic or random.random() < self.full_set_prob:
            slot_ids = [self.taxonomy.class_id(name) for name in self.taxonomy.parent_classes(entry["parent"])]
        else:
            slot_ids = sorted(present_ids)
        for slot, class_id in enumerate(slot_ids[: self.max_stems]):
            labels[slot] = class_id
            valid[slot] = True
            if class_id in stored_ids:          # its real encoding, silent stretch or not
                stems[slot] = stored[stored_ids.index(class_id)]
            else:
                stems[slot] = self._silence.expand(self.channels, self.segment_frames)
        return parent, stems, labels, valid
