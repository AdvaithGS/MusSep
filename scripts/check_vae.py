"""Check the audio VAE before building a latent dataset on it.

    python -m scripts.check_vae --moisesdb /data/moisesdb --max-tracks 5
    python -m scripts.check_vae vocals.wav drums.wav

Reports, for the VAE that Stage 2 will use:

1. its real attributes: sample rate, hop length, latent width, frame rate, and
   the channel count and storage cost of the stereo latents MusSep builds;
2. reconstruction quality (encode, then decode) per sub-stem class and per
   parent stem, as SDR and SI-SDR at the VAE's sample rate. Every Stage 2
   output passes through the decoder, so this is a ceiling on what the
   generator can reach before the projection onto the parent;
3. chunked against whole-signal processing: long songs are encoded and decoded
   in pieces (``--chunk-seconds`` with margins), so the two must agree;
4. repeatability: decoding the same latent twice (the watermarking decoder may
   add a random message each time).

Excerpts (``--max-seconds``, centred) keep it fast.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from models.dac_vae import DEFAULT_VAE, DACVAEWrapper, build_vae
from utils.audio import load_audio, rms_db
from utils.metrics import sdr, si_sdr
from utils.training import select_device


def _excerpt(audio: torch.Tensor, seconds: Optional[float], sample_rate: int) -> torch.Tensor:
    if not seconds:
        return audio
    length = min(audio.shape[-1], int(seconds * sample_rate))
    start = (audio.shape[-1] - length) // 2
    return audio[..., start:start + length]


def _signals(args: argparse.Namespace, sample_rate: int) -> List[Dict]:
    """[{"song", "kind" (class or parent), "name", "audio" (2, T)}] at the VAE's rate."""
    out: List[Dict] = []
    for path in args.inputs:
        audio, _ = load_audio(path, sample_rate=sample_rate, channels=2)
        out.append({"song": Path(path).stem, "kind": "file", "name": Path(path).stem,
                    "audio": _excerpt(audio, args.max_seconds, sample_rate)})
    if args.moisesdb:
        from data.dataset import load_and_sum
        from data.moisesdb18 import index_moisesdb
        from data.taxonomy import Taxonomy

        taxonomy = Taxonomy.load(args.taxonomy)
        tracks = index_moisesdb(args.moisesdb, taxonomy)[: args.max_tracks]
        for track in tracks:
            parents = sorted({f.parent for f in track.files})
            stems = {p: load_and_sum(track.parent_files(p), sample_rate) for p in parents}
            length = max(x.shape[-1] for x in stems.values())
            segment = length if not args.max_seconds else min(length, int(args.max_seconds * sample_rate))
            start = (length - segment) // 2
            stop = start + segment
            for parent, audio in stems.items():
                out.append({"song": track.track_id, "kind": "parent", "name": parent,
                            "audio": torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))[..., start:stop]})
                for name, paths in track.raw_groups(parent).items():
                    sub = load_and_sum(paths, sample_rate, length=length)[..., start:stop]
                    out.append({"song": track.track_id, "kind": "class", "name": name, "audio": sub})
    return out


@torch.no_grad()
def check(args: argparse.Namespace) -> Dict:
    device = select_device(args.device)
    whole = build_vae(args.vae, chunk_seconds=1e9).to(device)               # never chunked
    chunked = DACVAEWrapper(vae=whole.vae, chunk_seconds=args.chunk_seconds, margin_seconds=args.margin_seconds)
    rate, hop, width = whole.sample_rate, whole.hop_length, whole.latent_dim
    channels = 2 * width
    attributes = {"vae": args.vae, "sample_rate": rate, "hop_length": hop, "latent_dim": width,
                  "frame_rate": whole.frame_rate, "stereo_latent_channels": channels,
                  "float16_mb_per_minute_per_stem": round(2 * channels * whole.frame_rate * 60 / 1e6, 3)}
    print("VAE: " + ", ".join(f"{k} {v}" for k, v in attributes.items()))

    signals = _signals(args, rate)
    if not signals:
        raise SystemExit("give audio files and/or --moisesdb")
    rows, audio_seconds, compute = [], 0.0, 0.0
    for item in signals:
        audio = item["audio"]
        if rms_db(audio) < -60:
            continue
        t0 = time.time()
        latent = whole.encode(audio)
        recon = whole.decode(latent, length=audio.shape[-1]).cpu()
        compute += time.time() - t0
        audio_seconds += audio.shape[-1] / rate
        row = {"song": item["song"], "kind": item["kind"], "name": item["name"],
               "sdr": sdr(audio, recon), "si_sdr": si_sdr(audio, recon)}
        if args.chunk_seconds and audio.shape[-1] / rate > args.chunk_seconds:
            latent_c = chunked.encode(audio)
            row["chunk_latent_rel_diff"] = float((latent_c - latent).norm() / latent.norm().clamp_min(1e-12))
            row["chunk_decode_sdr"] = sdr(recon, chunked.decode(latent, length=audio.shape[-1]).cpu())
        row["repeat_decode_sdr"] = sdr(recon, whole.decode(latent, length=audio.shape[-1]).cpu())
        rows.append(row)
        print(f"{item['song']:>14s} {item['kind']:6s} {item['name']:16s} SDR {row['sdr']:6.2f}  SI-SDR {row['si_sdr']:6.2f}"
              + (f"  chunked: latent diff {row['chunk_latent_rel_diff']:.2e}, decode SDR vs whole {row['chunk_decode_sdr']:.1f}"
                 if "chunk_latent_rel_diff" in row else "")
              + f"  repeat decode SDR {row['repeat_decode_sdr']:.1f}", flush=True)

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted({(r["kind"], r["name"]) for r in rows}):
        values = sorted(r["sdr"] for r in rows if (r["kind"], r["name"]) == key)
        summary[f"{key[0]}:{key[1]}"] = {"median_sdr": values[len(values) // 2], "n": len(values)}
    print("\nmedian reconstruction SDR (dB):")
    for key, value in summary.items():
        print(f"  {key:28s} {value['median_sdr']:6.2f}  (n={value['n']})")
    print(f"speed: {compute / max(audio_seconds, 1e-9):.3f} s of compute per second of audio (encode + decode, {device})")
    print("Repeat-decode SDR far below 100 dB means decoding is not deterministic (e.g. a random watermark message).")
    return {"attributes": attributes, "summary": summary, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="audio files")
    parser.add_argument("--moisesdb", help="MoisesDB root: check every class and parent stem of a few songs")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--max-tracks", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--chunk-seconds", type=float, default=10.0, help="chunk size for the chunked-vs-whole check")
    parser.add_argument("--margin-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", help="write the results as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    args = build_parser().parse_args(argv)
    result = check(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
