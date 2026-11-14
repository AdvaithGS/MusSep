"""Slakh2100 indexer.

Expected layout (Slakh2100-FLAC-Redux)::

    <root>/{train,validation,test}/TrackXXXXX/metadata.yaml
    <root>/{train,validation,test}/TrackXXXXX/stems/S00.flac ...

``metadata.yaml`` holds ``stems: {S00: {inst_class, program_num, is_drum,
audio_rendered, ...}}``. Stems are mono, 44.1 kHz. Tracks under ``omitted``
are ignored. The split is taken from the folder name.

Slakh is synthetic (MIDI rendered with sample libraries). It supervises only
the "other" parent and is down-weighted at training time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from .dataset import HierTrack, SourceFile
from .taxonomy import Taxonomy

SOURCE = "slakh"
_SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}
_YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _split_from_path(track_dir: Path, root: Path) -> Optional[str]:
    for part in reversed(track_dir.relative_to(root).parts[:-1]):
        if part in _SPLIT_DIRS:
            return _SPLIT_DIRS[part]
        if part == "omitted":
            return None
    return "train"


def index_slakh(root: str | Path, taxonomy: Taxonomy) -> List[HierTrack]:
    """Returns one :class:`HierTrack` per song, files labelled with their
    parent and *candidate* class (from the General MIDI program number)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Slakh folder not found: {root}")
    tracks: List[HierTrack] = []
    for meta_path in sorted(root.rglob("metadata.yaml")):
        track_dir = meta_path.parent
        split = _split_from_path(track_dir, root)
        if split is None:
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = yaml.load(handle, Loader=_YamlLoader)  # noqa: S506 - safe loader
        except (OSError, yaml.YAMLError) as error:
            print(f"[slakh] skipping {track_dir}: {error}")
            continue
        track = HierTrack(track_id=track_dir.name, source=SOURCE, path=track_dir, split=split)
        for stem_id, info in ((meta or {}).get("stems") or {}).items():
            info = info or {}
            if not info.get("audio_rendered", True):
                continue
            path = next(
                (p for p in (track_dir / "stems" / f"{stem_id}{ext}" for ext in (".flac", ".wav")) if p.is_file()),
                None,
            )
            if path is None:
                track.missing.append(str(track_dir / "stems" / f"{stem_id}.flac"))
                continue
            is_drum = bool(info.get("is_drum", False))
            program = info.get("program_num")
            label = "drums" if is_drum else f"{info.get('inst_class', '?')} / program {program}"
            assignment = taxonomy.slakh_lookup(program, is_drum)
            if assignment.excluded:
                track.excluded.append(label)
                continue
            track.files.append(SourceFile(path, assignment.parent, assignment.raw_class, label, assignment.matched))
        if track.files:
            tracks.append(track)
    if not tracks:
        raise RuntimeError(f"no Slakh tracks (metadata.yaml files) found under {root}")
    return tracks
