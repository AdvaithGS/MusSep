"""Counts songs and minutes per candidate sub-stem class and applies the
minimum-data rule, producing the frozen taxonomy used for Stage 2.

    python -m scripts.taxonomy_stats --moisesdb /data/moisesdb --slakh /data/slakh2100 \
        --taxonomy configs/taxonomy.yaml --freeze configs/taxonomy.frozen.yaml

It reads metadata and audio headers only (no decoding), so it takes seconds
to minutes. Three tables are printed:

1. every dataset label seen and the class it maps to (labels decided by a
   default rule are flagged - check them);
2. songs and minutes per candidate class, per dataset;
3. the merges decided by the minimum-data rule and the resulting class list.

"Minutes" is the summed length of the songs' class stems (file duration), not
the time the instrument is actually playing.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data.dataset import HierTrack
from data.moisesdb18 import index_moisesdb
from data.slakh import index_slakh
from data.taxonomy import Taxonomy, plan_merges
from utils.audio import audio_info


@dataclass
class ClassStats:
    # seconds[source][raw_class][track_id] = duration of that song's class stem
    seconds: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    split: Dict[Tuple[str, str], str] = field(default_factory=dict)          # (source, track_id) -> split
    # labels[(source, label, parent, raw_class, matched)] = [files, {track ids}]
    labels: Dict[Tuple, List] = field(default_factory=dict)
    excluded: Dict[Tuple[str, str], int] = field(default_factory=dict)
    missing: int = 0
    num_tracks: Dict[str, int] = field(default_factory=dict)

    def songs(self, raw_class: str, sources: Sequence[str], split: str = "all") -> Set[str]:
        found: Set[str] = set()
        for source in sources:
            for track_id in self.seconds.get(source, {}).get(raw_class, {}):
                if split == "all" or self.split[(source, track_id)] == split:
                    found.add(f"{source}:{track_id}")
        return found

    def minutes(self, raw_classes: Sequence[str], source: str) -> float:
        per_track: Dict[str, float] = defaultdict(float)
        for raw_class in raw_classes:
            for track_id, seconds in self.seconds.get(source, {}).get(raw_class, {}).items():
                per_track[track_id] = max(per_track[track_id], seconds)
        return sum(per_track.values()) / 60.0


def collect_stats(tracks: Sequence[HierTrack], measure_duration: bool = True) -> ClassStats:
    stats = ClassStats()
    for track in tracks:
        stats.num_tracks[track.source] = stats.num_tracks.get(track.source, 0) + 1
        stats.split[(track.source, track.track_id)] = track.split
        stats.missing += len(track.missing)
        for label in track.excluded:
            stats.excluded[(track.source, label)] = stats.excluded.get((track.source, label), 0) + 1
        for item in track.files:
            key = (track.source, item.label, item.parent, item.raw_class, item.matched)
            entry = stats.labels.setdefault(key, [0, set()])
            entry[0] += 1
            entry[1].add(track.track_id)
            if item.raw_class is None:
                continue
            seconds = 0.0
            if measure_duration:
                frames, sample_rate, _ = audio_info(item.path)
                seconds = frames / float(sample_rate)
            per_class = stats.seconds.setdefault(track.source, {}).setdefault(item.raw_class, {})
            per_class[track.track_id] = max(per_class.get(track.track_id, 0.0), seconds)
    return stats


def _table(header: Sequence[str], rows: Sequence[Sequence], align_left: int = 2) -> str:
    cells = [list(map(str, header))] + [list(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = []
    for index, row in enumerate(cells):
        lines.append("  ".join(
            cell.ljust(widths[i]) if i < align_left else cell.rjust(widths[i]) for i, cell in enumerate(row)
        ))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def label_report(stats: ClassStats, max_rows: Optional[int] = None) -> str:
    rows = []
    for (source, label, parent, raw_class, matched), (files, songs) in sorted(
        stats.labels.items(), key=lambda kv: (kv[0][0], kv[0][2] or "", kv[0][3] or "", kv[0][1])
    ):
        rows.append([source, label, parent, raw_class or "(not split)", files, len(songs), "" if matched else "DEFAULT RULE"])
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows] + [["...", f"({len(rows) - max_rows} more)", "", "", "", "", ""]]
    text = _table(["source", "dataset label", "parent", "candidate class", "files", "songs", "note"], rows, align_left=4)
    for (source, label), count in sorted(stats.excluded.items()):
        text += f"\nexcluded: {source}: {label}  ({count} files)"
    if stats.missing:
        text += f"\nWARNING: {stats.missing} audio files listed in metadata were not found on disk"
    return text


def class_report(taxonomy: Taxonomy, stats: ClassStats, min_songs: int, sources: Sequence[str], split: str) -> str:
    all_sources = sorted(stats.num_tracks)
    header = ["parent", "candidate class"]
    for source in all_sources:
        header += [f"{source} songs (train/all)", f"{source} min"]
    header += ["held out (moisesdb)", f"counted ({'+'.join(sources)}, {split})", f">= {min_songs}?"]
    rows = []
    for parent, names in taxonomy.candidates.items():
        for name in names:
            row = [parent, name]
            for source in all_sources:
                row += [
                    f"{len(stats.songs(name, [source], 'train'))}/{len(stats.songs(name, [source], 'all'))}",
                    f"{stats.minutes([name], source):.0f}",
                ]
            counted = len(stats.songs(name, sources, split))
            held = len(stats.songs(name, ["moisesdb"], "val"))
            verdict = "yes" if counted >= min_songs else "NO"
            if name == taxonomy.residual[parent]:
                verdict = f"residual, {verdict}" if taxonomy.collapse_if_residual_rare[parent] else "residual"
            rows.append(row + [held, counted, verdict])
    return _table(header, rows)


def final_report(frozen: Taxonomy, stats: ClassStats, sources: Sequence[str], split: str) -> str:
    members: Dict[str, List[str]] = defaultdict(list)
    for parent, names in frozen.candidates.items():
        for name in names:
            resolved = frozen.resolve(name)
            if resolved is not None:
                members[resolved].append(name)
    all_sources = sorted(stats.num_tracks)
    header = ["id", "class", "parent", "made of", "counted songs"] + [f"{s} min" for s in all_sources]
    rows = []
    for name in frozen.class_names[1:]:
        counted: Set[str] = set()
        for member in members[name]:
            counted |= stats.songs(member, sources, split)
        rows.append(
            [frozen.class_id(name), name, frozen.parent_of(name), "+".join(members[name]), len(counted)]
            + [f"{stats.minutes(members[name], s):.0f}" for s in all_sources]
        )
    return _table(header, rows, align_left=4)


def freeze_taxonomy(
    taxonomy: Taxonomy,
    tracks: Sequence[HierTrack],
    min_songs: Optional[int] = None,
    sources: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    measure_duration: bool = True,
    verbose: bool = True,
) -> Tuple[Taxonomy, str, ClassStats]:
    """Counts, applies the minimum-data rule and returns
    ``(frozen taxonomy, printable report, stats)``."""
    min_songs = taxonomy.min_songs if min_songs is None else int(min_songs)
    sources = tuple(sources or taxonomy.count_sources)
    split = split or taxonomy.count_split
    stats = collect_stats(tracks, measure_duration=measure_duration)

    songs = {
        name: stats.songs(name, sources, split)
        for names in taxonomy.candidates.values() for name in names
    }
    merges, collapsed, log = plan_merges(taxonomy, songs, min_songs)
    frozen = taxonomy.with_merges(merges, collapsed, frozen=True)
    frozen.spec["min_data"] = {"min_songs": min_songs, "sources": list(sources), "split": split}
    frozen.min_songs, frozen.count_sources, frozen.count_split = min_songs, sources, split

    parts = [
        "Tracks indexed: " + ", ".join(f"{source} {count}" for source, count in sorted(stats.num_tracks.items())),
        "",
        "== 1. Dataset labels -> candidate classes ==",
        label_report(stats),
        "",
        "== 2. Songs and minutes per candidate class ==",
        class_report(taxonomy, stats, min_songs, sources, split),
        "",
        f"== 3. Minimum-data rule (min_songs = {min_songs}; counting {'+'.join(sources)}, split '{split}') ==",
        "\n".join(log) if log else "no class is below the threshold; nothing merged",
        "",
        f"Frozen taxonomy: {frozen.num_classes - 1} classes over parents {frozen.parents} "
        f"(max {frozen.max_slots} slots)",
        final_report(frozen, stats, sources, split),
    ]
    unvalidated = [name for name in frozen.class_names[1:]
                   if not any(stats.songs(raw, ["moisesdb"], "val") for raw in taxonomy.candidates[frozen.parent_of(name)]
                              if frozen.resolve(raw) == name)]
    if unvalidated and "moisesdb" in stats.num_tracks:
        parts.append(f"\nWARNING: no held-out MoisesDB song contains {unvalidated}; these classes cannot be validated.")
    missing_sources = [source for source in sources if source not in stats.num_tracks]
    if missing_sources:
        parts.append(
            f"\nWARNING: no tracks from {missing_sources} were indexed, but the rule counts only "
            f"{list(sources)}. Every class looks empty; do not use this result."
        )
    report = "\n".join(parts)
    if verbose:
        print(report)
    return frozen, report, stats


def index_sources(
    taxonomy: Taxonomy,
    moisesdb: Optional[str] = None,
    slakh: Optional[str] = None,
    val_fraction: float = 0.1,
    split_seed: int = 0,
    min_val: int = 2,
    min_songs: Optional[int] = None,
) -> List[HierTrack]:
    """MoisesDB is split with the same ``min_songs`` the minimum-data rule
    will use, so the held-out songs never sink a class below it."""
    tracks: List[HierTrack] = []
    if moisesdb:
        tracks += index_moisesdb(moisesdb, taxonomy, val_fraction=val_fraction, seed=split_seed,
                                 min_val=min_val, min_songs=min_songs)
    if slakh:
        tracks += index_slakh(slakh, taxonomy)
    if not tracks:
        raise SystemExit("give at least one of --moisesdb / --slakh")
    return tracks


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moisesdb", help="MoisesDB root")
    parser.add_argument("--slakh", help="Slakh2100 root")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--min-songs", type=int, help="override min_data.min_songs")
    parser.add_argument("--count-sources", nargs="+", help="override min_data.sources (moisesdb, slakh)")
    parser.add_argument("--count-split", choices=["train", "all"], help="override min_data.split")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="MoisesDB validation fraction")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--min-val-songs", type=int, default=2, help="held-out songs guaranteed per class that can survive")
    parser.add_argument("--no-duration", action="store_true", help="skip reading audio headers (no minutes)")
    parser.add_argument("--freeze", help="write the frozen taxonomy YAML here")
    parser.add_argument("--report", help="also write the printed report to this text file")
    parser.add_argument("--json", help="write raw counts as JSON")
    args = parser.parse_args(argv)

    taxonomy = Taxonomy.load(args.taxonomy)
    if taxonomy.frozen:
        print(f"note: {args.taxonomy} is already frozen; its merges are kept and the rule is re-applied on top")
    tracks = index_sources(taxonomy, args.moisesdb, args.slakh, args.val_fraction, args.split_seed,
                           args.min_val_songs, args.min_songs)
    frozen, report, stats = freeze_taxonomy(
        taxonomy, tracks, args.min_songs, args.count_sources, args.count_split,
        measure_duration=not args.no_duration,
    )
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
    if args.json:
        payload = {
            source: {name: {"songs": len(per_track), "minutes": sum(per_track.values()) / 60.0}
                     for name, per_track in classes.items()}
            for source, classes in stats.seconds.items()
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.freeze:
        frozen.save(args.freeze, header="Frozen by scripts/taxonomy_stats.py.\n\n" + report)
        print(f"\nwrote {args.freeze}")
    else:
        print("\n(dry run: pass --freeze <path> to write the frozen taxonomy)")


if __name__ == "__main__":
    main()
