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
so this measures Stage 2 alone, not the full cascade.
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


def evaluate_stage2(args: argparse.Namespace) -> Dict:
    from data.dataset import load_and_sum
    from data.moisesdb18 import index_moisesdb
    from models.dac_vae import build_vae
    from pipeline.infer import SubStemGenerator, load_stage2

    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision)
    loaded = load_stage2(args.stage2)
    taxonomy = loaded.taxonomy
    generator = SubStemGenerator(loaded, build_vae(args.vae or loaded.info["vae"]), device, amp, batch_size=args.batch_size)
    keep = held_out_ids(args.manifest)
    tracks = index_moisesdb(args.moisesdb, taxonomy, val_fraction=args.val_fraction, seed=args.split_seed,
                            min_val=args.min_val_songs)
    tracks = [t for t in tracks if (t.track_id in keep if keep is not None else t.split == "val")][: args.max_tracks]
    if not tracks:
        raise SystemExit("no held-out MoisesDB songs found (check --manifest / --val-fraction)")
    if keep is None:
        print("note: held-out songs recomputed from the split settings; make sure they match preprocessing (or pass --manifest)")
    sample_rate = loaded.info["sample_rate"]
    sampling = {"sampler": args.sampler, "num_steps": args.steps, "cfg_scale": args.cfg_scale,
                "project": False if args.no_project else None, "seed": args.seed}

    sdr_rows: Dict[str, List[float]] = {}
    sisdr_rows: Dict[str, List[float]] = {}
    leak_rows: Dict[str, List[float]] = {}
    tracks_out = []
    for index, track in enumerate(tracks):
        for parent in taxonomy.parents:
            groups = track.groups(parent, taxonomy)
            if not groups:
                continue
            refs = {name: load_and_sum(paths, sample_rate) for name, paths in groups.items()}
            length = max(x.shape[-1] for x in refs.values())
            refs = {k: torch.nn.functional.pad(v, (0, length - v.shape[-1])) for k, v in refs.items()}
            segment = length if not args.max_seconds else min(length, int(args.max_seconds * sample_rate))
            start = (length - segment) // 2
            refs = {k: _excerpt(v, start, segment) for k, v in refs.items()}
            parent_audio = torch.stack(list(refs.values())).sum(dim=0)
            if rms_db(parent_audio) < -60:
                continue
            t0 = time.time()
            estimates = generator.separate(parent_audio, sample_rate, parent, **sampling)
            row = {"track": track.track_id, "parent": parent, "seconds": round(time.time() - t0, 1), "classes": {}}
            for name, estimate in estimates.items():
                ref = refs.get(name)
                if ref is not None and rms_db(ref) >= -60:
                    a, b = sdr(ref, estimate), si_sdr(ref, estimate)
                    sdr_rows.setdefault(name, []).append(a)
                    sisdr_rows.setdefault(name, []).append(b)
                    row["classes"][name] = {"sdr": a, "si_sdr": b}
                else:
                    leak = rms_db(estimate) - rms_db(parent_audio)
                    leak_rows.setdefault(name, []).append(leak)
                    row["classes"][name] = {"absent_level_db": leak}
            tracks_out.append(row)
            print(f"[{index + 1}/{len(tracks)}] {track.track_id} {parent}: " + ", ".join(
                f"{k} {v.get('sdr', v.get('absent_level_db')):.1f}" for k, v in row["classes"].items()), flush=True)

    def summary(rows: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        return {k: {"median": float(torch.tensor(v).median()), "mean": float(sum(v) / len(v)), "n": len(v)} for k, v in rows.items()}
    result = {"stage": 2, "checkpoint": args.stage2, "distilled": loaded.distilled, "sampling": sampling,
              "max_seconds": args.max_seconds, "sdr": summary(sdr_rows), "si_sdr": summary(sisdr_rows),
              "absent_level_db": summary(leak_rows), "tracks": tracks_out}
    print("\nclass               n   SDR median   SI-SDR median   absent: level vs parent (n)")
    for name in taxonomy.class_names[1:]:
        s, i, l = result["sdr"].get(name), result["si_sdr"].get(name), result["absent_level_db"].get(name)
        print(f"{name:18s} {s['n'] if s else 0:3d}   {s['median'] if s else math.nan:9.2f}   {i['median'] if i else math.nan:12.2f}"
              f"   {l['median'] if l else math.nan:8.1f} dB ({l['n'] if l else 0})")
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
    two.add_argument("--moisesdb", required=True)
    two.add_argument("--stage2", required=True)
    two.add_argument("--manifest", help="the manifest the model was trained on (defines the held-out songs)")
    two.add_argument("--vae")
    two.add_argument("--val-fraction", type=float, default=0.1)
    two.add_argument("--split-seed", type=int, default=0)
    two.add_argument("--min-val-songs", type=int, default=2)
    two.add_argument("--sampler", choices=["euler", "consistency"])
    two.add_argument("--steps", type=int)
    two.add_argument("--cfg-scale", type=float, default=1.0)
    two.add_argument("--no-project", action="store_true")
    two.add_argument("--seed", type=int, default=0)
    for p in (one, two):
        p.add_argument("--max-tracks", type=int)
        p.add_argument("--max-seconds", type=float, help="centred excerpt per track")
        p.add_argument("--batch-size", type=int, default=2)
        p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
        p.add_argument("--device", default="auto")
        p.add_argument("--out", help="write the results as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    result = evaluate_stage1(args) if args.stage == "stage1" else evaluate_stage2(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
