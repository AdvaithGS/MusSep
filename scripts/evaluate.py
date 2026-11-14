"""Benchmarks.

Stage 1 on MUSDB18-HQ::

    python -m scripts.evaluate stage1 --musdb-root /data/musdb18hq \
        --stage1 runs/vocals/best.pt runs/bass/best.pt runs/drums/best.pt runs/other/best.pt \
        --metric museval --out results/stage1_test.json

``--metric museval`` is BSSEval v4 through the ``museval`` package: SDR per
track = median over 1-second frames, benchmark = median over the 50 test
tracks. That is the protocol of the BS-RoFormer paper's MUSDB18-HQ table and
the only number here that may be set next to it (the paper's MUSDB-only model
averages 9.80 dB). It is comparable only if the evaluated models were trained
on MUSDB18-HQ alone. ``--metric usdr`` is the global SDR of the Sound Demixing
Challenges (mean over tracks), a different number.

Stage 2 / 4 on MoisesDB held-out songs (oracle parent stems)::

    python -m scripts.evaluate stage2 --moisesdb /data/moisesdb --stage2 runs/stage2/best.pt \
        --manifest /data/latents/manifest.json --out results/stage2_val.json

For every held-out song and every split parent, the true parent stem (sum of
its sub-stems) is given to the generator and its outputs are scored against
the true sub-stems: SDR and SI-SDR for classes that are present, and for
absent classes the level of what was generated relative to the parent (lower
is better; it should be silence). Held-out songs are those marked ``val`` in
the manifest, so they were never trained on. The parent is the oracle stem,
so this measures Stage 2 alone.

The full cascade, on the same songs::

    python -m scripts.evaluate cascade --moisesdb /data/moisesdb --stage2 runs/stage4/last.pt \
        --manifest /data/latents/manifest.json --stage1 runs/vocals/best.pt runs/drums/best.pt runs/bass/best.pt

The mixture is rebuilt from every MoisesDB file of the song (minus excluded
ones such as the click track), Stage 1 separates it, and each estimated
parent is split by the generator. Scores are against the same true sub-stems,
and the oracle is run on the same excerpts unless ``--no-oracle``, so the
"cascade - oracle" column is the cost of Stage 1's errors. Stage 1's own SDR
on these songs is reported too; MoisesDB is not what MUSDB18-HQ models were
trained on, so read it as a cross-dataset number, not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from utils.audio import load_audio, rms_db
from utils.metrics import aggregate, format_table, museval_available, museval_sdr, sdr, si_sdr
from utils.training import AmpContext, select_device


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def evaluate_stage1(args: argparse.Namespace) -> Dict:
    from data.musdb18 import MUSDBTrackDataset
    from pipeline.infer import Stage1Separator, load_stage1

    if args.metric == "museval" and not museval_available():
        raise SystemExit("museval is not installed (pip install museval), or use --metric usdr")
    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision)
    names = args.stage1_stems.split(",") if args.stage1_stems else None
    models = [load_stage1(path, args.stage1_config, names) for path in args.stage1]
    separator = Stage1Separator(models, device, amp, overlap=args.overlap, batch_size=args.batch_size)
    stems = [s for s in ("vocals", "bass", "drums", "other") if s in separator.stems or
             (s == "other" and all(x in separator.stems for x in ("vocals", "bass", "drums")))]
    dataset = MUSDBTrackDataset(args.musdb_root, args.split, stems, max_seconds=args.max_seconds, max_tracks=args.max_tracks)
    per_track: List[Dict[str, float]] = []
    rows = []
    for index, item in enumerate(dataset):
        t0 = time.time()
        estimates = separator.separate(item["mixture"], dataset.sample_rate)
        if args.metric == "museval":
            scores = museval_sdr(item["targets"], {k: estimates[k] for k in stems}, dataset.sample_rate)
        else:
            scores = {k: sdr(item["targets"][k], estimates[k]) for k in stems if item["targets"][k].abs().max() > 0}
        per_track.append(scores)
        rows.append({"track": item["name"], **scores})
        print(f"[{index + 1}/{len(dataset)}] {item['name']}: " + ", ".join(f"{k} {v:.2f}" for k, v in scores.items())
              + f"  ({time.time() - t0:.0f} s)", flush=True)
    how = "median" if args.metric == "museval" else "mean"
    summary = aggregate(per_track, how)
    print(format_table(summary, f"\n{args.metric} SDR, {how} over {len(per_track)} {args.split} tracks (dB)"))
    if args.metric == "museval" and (args.split != "test" or args.max_seconds or args.max_tracks):
        print("note: not the full test protocol (subset, excerpts or another split); not comparable with published tables")
    return {"stage": 1, "metric": args.metric, "aggregate": how, "split": args.split, "summary": summary, "tracks": rows,
            "checkpoints": list(args.stage1), "full_protocol": args.split == "test" and not args.max_seconds and not args.max_tracks}


# ---------------------------------------------------------------------------
# Stage 2 / 4
# ---------------------------------------------------------------------------

def held_out_ids(manifest_path: Optional[str]) -> Optional[set]:
    if not manifest_path:
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)["entries"]
    return {e["track_id"] for e in entries if e["source"] == "moisesdb" and e["split"] == "val"}


def _excerpt(signal: torch.Tensor, start: int, length: int) -> torch.Tensor:
    piece = signal[..., start:start + length]
    return torch.nn.functional.pad(piece, (0, length - piece.shape[-1]))


def _summary(rows: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {k: {"median": float(torch.tensor(v).median()), "mean": float(sum(v) / len(v)), "n": len(v)}
            for k, v in rows.items()}


class _Scores:
    """Per-class SDR / SI-SDR where the class is present, output level where it is absent."""

    def __init__(self):
        self.sdr: Dict[str, List[float]] = {}
        self.si_sdr: Dict[str, List[float]] = {}
        self.absent: Dict[str, List[float]] = {}

    def add(self, estimates: Dict[str, torch.Tensor], refs: Dict[str, torch.Tensor], parent_ref: torch.Tensor) -> Dict:
        out = {}
        for name, estimate in estimates.items():
            ref = refs.get(name)
            if ref is not None and rms_db(ref) >= -60:
                a, b = sdr(ref, estimate), si_sdr(ref, estimate)
                self.sdr.setdefault(name, []).append(a)
                self.si_sdr.setdefault(name, []).append(b)
                out[name] = {"sdr": a, "si_sdr": b}
            else:
                leak = rms_db(estimate) - rms_db(parent_ref)       # against the true parent in both modes
                self.absent.setdefault(name, []).append(leak)
                out[name] = {"absent_level_db": leak}
        return out

    def result(self) -> Dict:
        return {"sdr": _summary(self.sdr), "si_sdr": _summary(self.si_sdr), "absent_level_db": _summary(self.absent)}


def evaluate_substems(args: argparse.Namespace, cascade: bool) -> Dict:
    """Held-out MoisesDB songs. ``cascade``: the generator gets Stage 1's estimate
    of each parent (and the oracle parent too, unless --no-oracle)."""
    from data.dataset import load_and_sum
    from data.moisesdb18 import index_moisesdb
    from models.dac_vae import build_vae
    from pipeline.infer import Stage1Separator, SubStemGenerator, load_stage1, load_stage2

    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision)
    loaded = load_stage2(args.stage2)
    taxonomy = loaded.taxonomy
    generator = SubStemGenerator(loaded, build_vae(args.vae or loaded.info["vae"]), device, amp, batch_size=args.batch_size)
    separator = None
    if cascade:
        names = args.stage1_stems.split(",") if args.stage1_stems else None
        separator = Stage1Separator([load_stage1(path, args.stage1_config, names) for path in args.stage1], device, amp,
                                    overlap=args.overlap, batch_size=args.batch_size)
        missing = [p for p in taxonomy.parents if p not in separator.stems and not (
            p == "other" and all(x in separator.stems for x in ("vocals", "bass", "drums")))]
        if missing:
            print(f"note: Stage 1 does not produce {missing}; those parents are scored with the oracle only")
    oracle = not (cascade and args.no_oracle)

    keep = held_out_ids(args.manifest)
    tracks = index_moisesdb(args.moisesdb, taxonomy, val_fraction=args.val_fraction, seed=args.split_seed,
                            min_val=args.min_val_songs)
    tracks = [t for t in tracks if (t.track_id in keep if keep is not None else t.split == "val")][: args.max_tracks]
    if not tracks:
        raise SystemExit("no held-out MoisesDB songs found (check --manifest / --val-fraction)")
    if keep is None:
        print("note: held-out songs recomputed from the split settings; make sure they match preprocessing (or pass --manifest)")
    sample_rate = args.sample_rate
    sampling = {"sampler": args.sampler, "num_steps": args.steps, "cfg_scale": args.cfg_scale,
                "project": False if args.no_project else None, "seed": args.seed}

    scores = {"oracle": _Scores(), "cascade": _Scores()}
    primary: Dict[str, List[float]] = {}
    tracks_out = []
    for index, track in enumerate(tracks):
        parents = sorted({f.parent for f in track.files})
        stems = {p: load_and_sum(track.parent_files(p), sample_rate) for p in parents}
        length = max(x.shape[-1] for x in stems.values())
        segment = length if not args.max_seconds else min(length, int(args.max_seconds * sample_rate))
        start = (length - segment) // 2
        stems = {p: _excerpt(x, start, segment) for p, x in stems.items()}
        row: Dict = {"track": track.track_id, "parents": {}}
        estimated: Dict[str, torch.Tensor] = {}
        if separator is not None:
            mixture = torch.stack(list(stems.values())).sum(dim=0)   # every file except excluded ones (click track)
            estimated = separator.separate(mixture, sample_rate)
            row["stage1"] = {}
            for p, est in estimated.items():
                if p in stems and rms_db(stems[p]) >= -60:
                    value = sdr(stems[p], est)
                    primary.setdefault(p, []).append(value)
                    row["stage1"][p] = value
        for parent in taxonomy.parents:
            groups = track.groups(parent, taxonomy)
            if not groups or parent not in stems or rms_db(stems[parent]) < -60:
                continue
            refs = {name: _excerpt(load_and_sum(paths, sample_rate, length=length), start, segment)
                    for name, paths in groups.items()}
            entry: Dict = {}
            if oracle or parent not in estimated:
                entry["oracle"] = scores["oracle"].add(
                    generator.separate(stems[parent], sample_rate, parent, **sampling), refs, stems[parent])
            if parent in estimated:
                entry["cascade"] = scores["cascade"].add(
                    generator.separate(estimated[parent], sample_rate, parent, **sampling), refs, stems[parent])
            row["parents"][parent] = entry
            shown = "; ".join(f"{mode}: " + ", ".join(f"{k} {v.get('sdr', v.get('absent_level_db')):.1f}" for k, v in got.items())
                              for mode, got in entry.items())
            print(f"[{index + 1}/{len(tracks)}] {track.track_id} {parent}: {shown}", flush=True)
        tracks_out.append(row)

    result = {"stage": "cascade" if cascade else 2, "checkpoint": args.stage2, "distilled": loaded.distilled,
              "sampling": sampling, "max_seconds": args.max_seconds, "sample_rate": sample_rate, "tracks": tracks_out}
    if oracle:
        result["oracle"] = scores["oracle"].result()
    if cascade:
        result["cascade"] = scores["cascade"].result()
        result["stage1"] = {"checkpoints": list(args.stage1), "sdr": _summary(primary)}
        print("\nStage 1 on these songs (global SDR, dB): " + ", ".join(
            f"{k} {v['median']:.2f} (n={v['n']})" for k, v in result["stage1"]["sdr"].items()))
    modes = [m for m in ("oracle", "cascade") if m in result]
    print("\nclass              " + "".join(f"{m + ' SDR':>14s}{'n':>5s}" for m in modes)
          + ("   cascade - oracle" if len(modes) == 2 else "") + "   absent: level vs parent")
    for name in taxonomy.class_names[1:]:
        line, medians = f"{name:18s}", []
        for m in modes:
            got = result[m]["sdr"].get(name)
            medians.append(got["median"] if got else math.nan)
            line += f"{medians[-1]:14.2f}{got['n'] if got else 0:5d}"
        if len(modes) == 2:
            line += f"{medians[1] - medians[0]:19.2f}"
        absent = result[modes[-1]]["absent_level_db"].get(name)
        line += f"   {absent['median'] if absent else math.nan:.1f} dB ({absent['n'] if absent else 0})"
        print(line)
    print("(medians over held-out songs; the cascade's parents are Stage 1 estimates, the oracle's the true stems)")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    one = sub.add_parser("stage1", help="MUSDB18-HQ")
    one.add_argument("--musdb-root", required=True)
    one.add_argument("--stage1", nargs="+", required=True)
    one.add_argument("--stage1-config")
    one.add_argument("--stage1-stems")
    one.add_argument("--split", default="test", choices=["test", "valid"])
    one.add_argument("--metric", default="museval", choices=["museval", "usdr"])
    one.add_argument("--overlap", type=float, default=0.5)
    two = sub.add_parser("stage2", help="MoisesDB held-out songs, oracle parents")
    three = sub.add_parser("cascade", help="MoisesDB held-out songs, parents estimated by Stage 1 (and the oracle for comparison)")
    for p in (two, three):
        p.add_argument("--moisesdb", required=True)
        p.add_argument("--stage2", required=True)
        p.add_argument("--manifest", help="the manifest the model was trained on (defines the held-out songs)")
        p.add_argument("--vae")
        p.add_argument("--val-fraction", type=float, default=0.1)
        p.add_argument("--split-seed", type=int, default=0)
        p.add_argument("--min-val-songs", type=int, default=2)
        p.add_argument("--sampler", choices=["euler", "consistency"])
        p.add_argument("--steps", type=int)
        p.add_argument("--cfg-scale", type=float, default=None, help="default: the checkpoint's")
        p.add_argument("--no-project", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--sample-rate", type=int, default=44100, help="rate at which audio is loaded and scored")
    three.add_argument("--stage1", nargs="+", required=True)
    three.add_argument("--stage1-config")
    three.add_argument("--stage1-stems")
    three.add_argument("--overlap", type=float, default=0.5)
    three.add_argument("--no-oracle", action="store_true", help="skip the oracle run (half the generation time)")
    for p in (one, two, three):
        p.add_argument("--max-tracks", type=int)
        p.add_argument("--max-seconds", type=float, help="centred excerpt per track")
        p.add_argument("--batch-size", type=int, default=2)
        p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
        p.add_argument("--device", default="auto")
        p.add_argument("--out", help="write the results as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    result = evaluate_stage1(args) if args.stage == "stage1" else evaluate_substems(args, args.stage == "cascade")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
