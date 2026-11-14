"""Pre-encodes MoisesDB / Slakh2100 sub-stems into DAC-VAE latents and writes
the JSON manifest that Stage 2 trains from (the VAE never runs in the
training loop).

    python -m scripts.preprocess_latents --moisesdb /data/moisesdb --slakh /data/slakh2100 \
        --taxonomy configs/taxonomy.frozen.yaml --out /data/mussep_latents

For every song and every split parent (vocals / drums / other) one file
``latents/<source>/<track>__<parent>__aug<k>.pt`` is written with

    parent     (C, F) float16   latent of the parent stem (sum of its classes)
    stems      (S, C, F) float16  latents of the classes that are present
    class_ids  (S,) int64       taxonomy ids of those classes
    active     (F,) bool        frames where the parent is above the presence level
    cond       (C, F) float16   optional: parent with bleed from the other stems

``C = channels x latent_dim`` laid out ``[left, right]``. Classes whose
    class_active (S, F) bool    per class: frames above the presence level

Presence is decided per latent frame (RMS of each hop above ``--presence-db``),
not per song. A class is stored if it is active for at least
``--min-present-seconds`` in the song; the training loader then decides per
crop whether it is present (data/unified_dataset.py), so a guitar that plays
only in the chorus is "absent" in the crops where it is silent.

If the taxonomy is not frozen, songs and minutes per class are counted first
and the minimum-data rule is applied (see scripts/taxonomy_stats.py); the
frozen taxonomy is saved next to the manifest and embedded in it.

Disk use: float16 latents take ``2 * C * frame_rate`` bytes per second per
stem - check the printed estimate after the first few songs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data.dataset import HierTrack, load_and_sum
from data.taxonomy import Taxonomy
from data.transforms import apply_gain
from models.dac_vae import DEFAULT_VAE, DACVAEWrapper, build_vae
from scripts.taxonomy_stats import freeze_taxonomy, index_sources

MANIFEST_NAME = "manifest.json"


class RunningStats:
    """Per-latent-dimension mean / std, shared between the channel halves."""

    def __init__(self, latent_dim: int, channels: int):
        self.latent_dim, self.channels = latent_dim, channels
        self.count = 0
        self.total = torch.zeros(latent_dim, dtype=torch.float64)
        self.total_sq = torch.zeros(latent_dim, dtype=torch.float64)

    def update(self, latents: torch.Tensor) -> None:
        """``latents``: (..., C, F)."""
        x = latents.double().reshape(-1, self.channels, self.latent_dim, latents.shape[-1])
        self.count += x.shape[0] * x.shape[1] * x.shape[3]
        self.total += x.sum(dim=(0, 1, 3))
        self.total_sq += x.pow(2).sum(dim=(0, 1, 3))

    def result(self) -> Tuple[List[float], List[float]]:
        if self.count == 0:
            return [0.0] * (self.latent_dim * self.channels), [1.0] * (self.latent_dim * self.channels)
        mean = self.total / self.count
        std = (self.total_sq / self.count - mean.pow(2)).clamp_min(1e-12).sqrt()
        return mean.repeat(self.channels).tolist(), std.repeat(self.channels).tolist()


def frame_activity(audio: torch.Tensor, hop_length: int, num_frames: int, threshold_db: float) -> torch.Tensor:
    """``(C, T) -> (num_frames,)`` bool: RMS of each latent hop above the threshold."""
    needed = num_frames * hop_length
    audio = torch.nn.functional.pad(audio, (0, max(0, needed - audio.shape[-1])))[..., :needed]
    power = audio.reshape(audio.shape[0], num_frames, hop_length).pow(2).mean(dim=(0, 2))
    return 10.0 * torch.log10(power + 1e-10) > threshold_db


def _load_classes(
    track: HierTrack, parent: str, taxonomy: Taxonomy, sample_rate: int, channels: int, presence_db: float,
    hop_length: int, min_present_seconds: float,
) -> Dict[str, torch.Tensor]:
    groups = track.groups(parent, taxonomy)
    loaded = {name: load_and_sum(paths, sample_rate, channels) for name, paths in groups.items()}

    def active_seconds(audio: torch.Tensor) -> float:
        frames = -(-audio.shape[-1] // hop_length)
        return float(frame_activity(audio, hop_length, frames, presence_db).sum()) * hop_length / sample_rate

    loaded = {name: audio for name, audio in loaded.items()
              if audio.shape[-1] > 0 and active_seconds(audio) >= min_present_seconds}
    if not loaded:
        return {}
    length = max(audio.shape[-1] for audio in loaded.values())
    return {
        name: torch.nn.functional.pad(audio, (0, length - audio.shape[-1])) for name, audio in loaded.items()
    }


def preprocess(
    tracks: Sequence[HierTrack],
    taxonomy: Taxonomy,
    vae: DACVAEWrapper,
    out_dir: str | Path,
    vae_name: str = DEFAULT_VAE,
    num_aug: int = 0,
    gain_range_db: Tuple[float, float] = (-3.0, 3.0),
    bleed_range_db: Optional[Tuple[float, float]] = None,
    presence_db: float = -60.0,
    min_present_seconds: float = 1.0,
    split_info: Optional[Dict[str, Any]] = None,
    channels: int = 2,
    seed: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Encodes ``tracks`` and writes ``<out_dir>/manifest.json``. Returns the manifest.

    ``num_aug`` extra versions of every *training* song are encoded with an
    independent random gain per class (gain is not linear in latent space,
    so it has to be applied before encoding).
    """
    out_dir = Path(out_dir)
    sample_rate, hop = vae.sample_rate, vae.hop_length
    stats = RunningStats(vae.latent_dim, channels)
    entries: List[Dict[str, Any]] = []
    written_bytes, written_seconds = 0, 0.0

    for index, track in enumerate(tracks):
        for parent in taxonomy.parents:
            classes = _load_classes(track, parent, taxonomy, sample_rate, channels, presence_db, hop, min_present_seconds)
            if not classes:
                continue
            names = list(classes)
            class_ids = torch.tensor([taxonomy.class_id(name) for name in names], dtype=torch.long)
            bleed = None
            if bleed_range_db is not None and track.other_parent_files(parent):
                length = next(iter(classes.values())).shape[-1]
                bleed = load_and_sum(track.other_parent_files(parent), sample_rate, channels, length)

            for aug in range(1 + (num_aug if track.split == "train" else 0)):
                relative = Path("latents") / track.source / f"{track.track_id}__{parent}__aug{aug}.pt"
                path = out_dir / relative
                if path.is_file() and not overwrite:
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    rng = random.Random(f"{seed}:{track.source}:{track.track_id}:{parent}:{aug}")
                    stems = [
                        classes[name] if aug == 0 else apply_gain(classes[name], rng.uniform(*gain_range_db))
                        for name in names
                    ]
                    parent_audio = torch.stack(stems).sum(dim=0)
                    parent_latent = vae.encode(parent_audio)
                    payload = {
                        "parent": parent_latent.half(),
                        "stems": torch.stack([vae.encode(stem) for stem in stems]).half(),
                        "class_ids": class_ids,
                        "active": frame_activity(parent_audio, hop, parent_latent.shape[-1], presence_db),
                        "class_active": torch.stack([frame_activity(stem, hop, parent_latent.shape[-1], presence_db)
                                                     for stem in stems]),
                    }
                    if bleed is not None:
                        cond_audio = parent_audio + apply_gain(bleed, rng.uniform(*bleed_range_db))
                        payload["cond"] = vae.encode(cond_audio).half()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, path)
                    written_bytes += path.stat().st_size
                    written_seconds += parent_latent.shape[-1] * hop / sample_rate

                if track.split == "train" and aug == 0:
                    stats.update(payload["stems"].float())
                    stats.update(payload["parent"].float())
                entries.append({
                    "id": f"{track.source}/{track.track_id}/{parent}/aug{aug}",
                    "track_id": track.track_id,
                    "source": track.source,
                    "split": track.split,
                    "parent": parent,
                    "aug": aug,
                    "latent_path": relative.as_posix(),
                    "num_frames": int(payload["parent"].shape[-1]),
                    "classes": names,
                })
        if verbose and (index + 1) % 10 == 0:
            rate = f", {written_bytes / max(written_seconds, 1e-9) * 3600 / 1e9:.2f} GB per encoded hour" if written_seconds else ""
            print(f"[preprocess] {index + 1}/{len(tracks)} songs, {len(entries)} entries{rate}", flush=True)

    mean, std = stats.result()
    manifest = {
        "version": 1,
        "vae": vae_name,
        "sample_rate": sample_rate,
        "hop_length": hop,
        "latent_dim": vae.latent_dim,
        "channels": channels,
        "latent_rate": sample_rate / hop,
        "presence_db": presence_db,
        "min_present_seconds": min_present_seconds,
        "split": split_info or {},
        "taxonomy": taxonomy.to_dict(),
        "latent_stats": {"mean": mean, "std": std},
        "silence_latent": vae.silence_latent(channels).tolist(),
        "entries": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    if verbose:
        by_split: Dict[str, int] = {}
        for entry in entries:
            by_split[entry["split"]] = by_split.get(entry["split"], 0) + 1
        print(f"[preprocess] wrote {out_dir / MANIFEST_NAME}: {len(entries)} entries {by_split}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moisesdb", help="MoisesDB root")
    parser.add_argument("--slakh", help="Slakh2100 root")
    parser.add_argument("--out", required=True, help="output folder (latents/ and manifest.json)")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--vae", default=DEFAULT_VAE, help="DAC-VAE name / path, or 'toy' for a smoke run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-aug", type=int, default=0, help="extra random-gain versions per training song")
    parser.add_argument("--gain-db", type=float, nargs=2, default=(-3.0, 3.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--bleed-db", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="also store a conditioning latent with the other stems mixed in at a random "
                             "level in this range, e.g. -30 -15 (imitates imperfect Stage 1 output)")
    parser.add_argument("--presence-db", type=float, default=-60.0, help="per-frame level for 'active'")
    parser.add_argument("--min-present-seconds", type=float, default=1.0,
                        help="a class is stored if it is active this long in the song")
    parser.add_argument("--min-val-songs", type=int, default=2, help="held-out songs guaranteed per surviving class")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="MoisesDB validation fraction")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-tracks", type=int, help="per source, for quick experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-songs", type=int, help="override the taxonomy's min_data.min_songs")
    parser.add_argument("--no-min-data-rule", action="store_true",
                        help="use the taxonomy as given even if it is not frozen")
    parser.add_argument("--dry-run", action="store_true", help="print the class counts and stop")
    args = parser.parse_args(argv)

    taxonomy = Taxonomy.load(args.taxonomy)
    tracks = index_sources(taxonomy, args.moisesdb, args.slakh, args.val_fraction, args.split_seed,
                           args.min_val_songs, args.min_songs)
    out_dir = Path(args.out)

    if args.dry_run or (not taxonomy.frozen and not args.no_min_data_rule):
        frozen, report, _ = freeze_taxonomy(taxonomy, tracks, args.min_songs)
        if args.dry_run:
            return
        frozen.save(out_dir / "taxonomy.frozen.yaml", header="Frozen by scripts/preprocess_latents.py.\n\n" + report)
        print(f"\nwrote {out_dir / 'taxonomy.frozen.yaml'}\n")
        taxonomy = frozen

    if args.max_tracks:
        kept: List[HierTrack] = []
        seen: Dict[str, int] = {}
        for track in tracks:
            seen[track.source] = seen.get(track.source, 0) + 1
            if seen[track.source] <= args.max_tracks:
                kept.append(track)
        tracks = kept

    from utils.training import select_device

    vae = build_vae(args.vae).to(select_device(args.device))
    print(f"[preprocess] VAE {args.vae}: {vae.sample_rate} Hz, hop {vae.hop_length} "
          f"({vae.frame_rate:.2f} frames/s), latent_dim {vae.latent_dim}")
    preprocess(
        tracks, taxonomy, vae, out_dir, vae_name=args.vae, num_aug=args.num_aug,
        gain_range_db=tuple(args.gain_db), bleed_range_db=tuple(args.bleed_db) if args.bleed_db else None,
        presence_db=args.presence_db, min_present_seconds=args.min_present_seconds, seed=args.seed,
        overwrite=args.overwrite, split_info={"moisesdb": "stratified", "val_fraction": args.val_fraction,
                                              "seed": args.split_seed, "min_val_songs": args.min_val_songs},
    )


if __name__ == "__main__":
    main()
