"""MoisesDB indexer.

Layout of the official distribution (see ``moisesdb.track.MoisesDBTrack``)::

    <root>/<provider>/<track id>/data.json
    <root>/<provider>/<track id>/<stemName>/<file id>.<extension>

``data.json`` holds ``stems: [{stemName, tracks: [{id, trackType, extension,
has_bleed}]}]``. A root that points directly at a folder of track directories
works too: every ``data.json`` below ``root`` is picked up.

MoisesDB ships no official train / validation split. Ours is deterministic
and stratified (:func:`stratified_val`): every class that can survive the
minimum-data rule gets up to ``min_val`` held-out songs, without taking so
many that its training count drops below the rule's threshold; the rest of
the held-out set is filled in hash order up to ``val_fraction``. A plain
random 10% of ~240 songs can easily hold no example of a rare class, which
would make that class impossible to validate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set

from .dataset import HierTrack, SourceFile
from .taxonomy import Taxonomy

SOURCE = "moisesdb"


def split_of(track_id: str, val_fraction: float = 0.1, seed: int = 0) -> str:
    """Deterministic ``train`` / ``val`` assignment that does not depend on
    which other tracks are present."""
    return "val" if _rank(track_id, seed) < val_fraction else "train"


def _rank(track_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{track_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def stratified_val(
    classes_of: Dict[str, Set[str]],
    val_fraction: float = 0.1,
    min_songs: int = 20,
    min_val: int = 2,
    seed: int = 0,
) -> Set[str]:
    """Held-out song ids.

    ``classes_of[song]`` is the set of (candidate) classes in that song. A class
    with ``n >= min_songs`` songs can survive the minimum-data rule; it gets
    ``min(min_val, n - min_songs)`` held-out songs, rarest class first, and no
    song is ever held out if that would push such a class below ``min_songs``
    training songs. The set is then filled in hash order up to
    ``round(val_fraction * number of songs)``, under the same constraint.
    """
    if val_fraction <= 0 or not classes_of:
        return set()
    total = Counter(c for classes in classes_of.values() for c in classes)
    cap = {c: n - min_songs for c, n in total.items() if n >= min_songs}
    order = sorted(classes_of, key=lambda song: (_rank(song, seed), song))
    val: Set[str] = set()
    held = Counter()

    def take(song: str) -> bool:
        if song in val or any(held[c] >= cap[c] for c in classes_of[song] if c in cap):
            return False
        val.add(song)
        held.update(classes_of[song])
        return True

    for c in sorted(cap, key=lambda c: (total[c], c)):
        quota = min(min_val, cap[c])
        for song in order:
            if held[c] >= quota:
                break
            if c in classes_of[song]:
                take(song)
    target = int(round(val_fraction * len(classes_of)))
    for song in order:
        if len(val) >= target:
            break
        take(song)
    return val


def index_moisesdb(
    root: str | Path,
    taxonomy: Taxonomy,
    val_fraction: float = 0.1,
    seed: int = 0,
    min_val: int = 2,
    min_songs: Optional[int] = None,
) -> List[HierTrack]:
    """Returns one :class:`HierTrack` per song, files labelled with their
    parent and *candidate* class, split by :func:`stratified_val`.

    The split depends only on candidate classes, so indexing with the
    candidate or the frozen taxonomy gives the same split. ``min_songs``
    defaults to the taxonomy's minimum-data threshold."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MoisesDB folder not found: {root}")
    tracks: List[HierTrack] = []
    for meta_path in sorted(root.rglob("data.json")):
        track_dir = meta_path.parent
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[moisesdb] skipping {track_dir}: {error}")
            continue
        track = HierTrack(
            track_id=track_dir.name,
            source=SOURCE,
            path=track_dir,
            split="train",
        )
        for stem in meta.get("stems") or []:
            stem_name = str(stem.get("stemName", ""))
            for item in stem.get("tracks") or []:
                track_type = str(item.get("trackType", ""))
                label = f"{stem_name} / {track_type}"
                assignment = taxonomy.moisesdb_lookup(stem_name, track_type)
                if assignment.excluded:
                    track.excluded.append(label)
                    continue
                path = track_dir / stem_name / f"{item.get('id')}.{item.get('extension', 'wav')}"
                if not path.is_file():
                    track.missing.append(str(path))
                    continue
                track.files.append(
                    SourceFile(path, assignment.parent, assignment.raw_class, label, assignment.matched)
                )
        if track.files:
            tracks.append(track)
    if not tracks:
        raise RuntimeError(f"no MoisesDB tracks (data.json files) found under {root}")
    classes_of = {t.track_id: {f.raw_class for f in t.files if f.raw_class is not None} for t in tracks}
    threshold = taxonomy.min_songs if min_songs is None else int(min_songs)
    val = stratified_val(classes_of, val_fraction, threshold, min_val, seed)
    for track in tracks:
        track.split = "val" if track.track_id in val else "train"
    return tracks
