"""Separate songs into primary stems and, optionally, labelled sub-stems.

    # primary stems with one or more Stage 1 models (one per stem, as in the paper)
    python -m scripts.separate song.wav --out separated/ \
        --stage1 runs/vocals/best.pt runs/drums/best.pt runs/bass/best.pt

    # ... then split vocals / drums / other into sub-stems with Stage 2 or Stage 4
    python -m scripts.separate song.wav --out separated/ --stage1 ... --stage2 runs/stage4/last.pt

    # only some sub-stems (the whole set is generated for the masks; only these are written)
    python -m scripts.separate song.wav --out separated/ --stage1 ... --stage2 ... --stems guitar,piano

    # the input already is a parent stem (e.g. an isolated drum track)
    python -m scripts.separate drums.wav --out separated/ --stage2 ... --input-is drums

    # a community BS-RoFormer checkpoint needs the YAML it was trained with
    python -m scripts.separate song.wav --out separated/ --stage1 model.ckpt --stage1-config model.yaml

Output: ``<out>/<song>/<stem>.wav`` and ``<out>/<song>/<parent>/<sub-stem>.wav``.
If ``other`` is not produced by a model but vocals, bass and drums are, it is
the mixture minus those three.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Sequence

from models.dac_vae import DEFAULT_VAE, build_vae
from pipeline.infer import (Stage1Separator, SubStemGenerator, load_stage1, load_stage2,
                            separate_hierarchical)
from utils.audio import load_audio, save_audio
from utils.training import AmpContext, select_device


def build(args: argparse.Namespace):
    device = select_device(args.device)
    amp = AmpContext(device, precision=args.precision)
    stage1 = None
    if args.stage1:
        if args.stage1_config and len(args.stage1) != 1:
            raise SystemExit("--stage1-config applies to a single --stage1 checkpoint")
        names = args.stage1_stems.split(",") if args.stage1_stems else None
        models = [load_stage1(path, args.stage1_config, names) for path in args.stage1]
        stage1 = Stage1Separator(models, device, amp, overlap=args.overlap, batch_size=args.batch_size)
    stage2 = None
    if args.stage2:
        loaded = load_stage2(args.stage2)
        vae = build_vae(args.vae or loaded.info.get("vae") or DEFAULT_VAE)
        stage2 = SubStemGenerator(loaded, vae, device, amp, batch_size=args.batch_size)
    if stage1 is None and args.input_is is None:
        raise SystemExit("give --stage1 checkpoint(s), or --input-is <parent> to split an isolated stem")
    if stage1 is None and stage2 is None:
        raise SystemExit("nothing to do: give --stage1 and/or --stage2")
    return stage1, stage2


def run(args: argparse.Namespace) -> List[Path]:
    stage1, stage2 = build(args)
    written: List[Path] = []
    sampling = {"sampler": args.sampler, "num_steps": args.steps, "cfg_scale": args.cfg_scale,
                "project": False if args.no_project else None, "seed": args.seed}
    for path in args.inputs:
        t0 = time.time()
        audio, sample_rate = load_audio(path, channels=2)
        result = separate_hierarchical(
            audio, sample_rate, stage1, stage2,
            parents=args.parents.split(",") if args.parents else None,
            stems=args.stems.split(",") if args.stems else None,
            input_is=args.input_is, **sampling,
        )
        folder = Path(args.out) / Path(path).stem
        for name, signal in result["primary"].items():
            if args.input_is is None:
                written.append(_save(folder / f"{name}.{args.format}", signal, sample_rate))
        for parent, subs in result["sub"].items():
            for name, signal in subs.items():
                written.append(_save(folder / parent / f"{name}.{args.format}", signal, sample_rate))
        seconds = audio.shape[-1] / sample_rate
        print(f"{path}: {len(result['primary'])} primary, {sum(len(s) for s in result['sub'].values())} sub-stems "
              f"in {time.time() - t0:.1f} s ({(time.time() - t0) / max(seconds, 1e-9):.2f}x real time) -> {folder}")
    return written


def _save(path: Path, signal, sample_rate: int) -> Path:
    peak = float(signal.abs().max())
    if peak > 1.0:
        print(f"  note: {path.name} peaks at {peak:.2f}; written as float to avoid clipping")
    save_audio(path, signal, sample_rate, subtype="FLOAT" if peak > 1.0 else "PCM_16")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="audio files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage1", nargs="+", help="Stage 1 checkpoint(s)")
    parser.add_argument("--stage1-config", help="training YAML of a community checkpoint")
    parser.add_argument("--stage1-stems", help="comma-separated stem names of a community checkpoint")
    parser.add_argument("--stage2", help="Stage 2 or distilled (Stage 4) checkpoint")
    parser.add_argument("--vae", help=f"VAE override (default: the one the checkpoint was trained with, {DEFAULT_VAE})")
    parser.add_argument("--parents", help="comma-separated parents to split (default: all)")
    parser.add_argument("--stems", help="comma-separated sub-stem classes to generate (default: every class)")
    parser.add_argument("--input-is", help="the input already is this parent stem; skip Stage 1")
    parser.add_argument("--sampler", choices=["euler", "consistency"], help="default: from the checkpoint")
    parser.add_argument("--steps", type=int, help="sampling steps (default: from the checkpoint)")
    parser.add_argument("--cfg-scale", type=float, help="classifier-free guidance, Euler sampler (default: the checkpoint's)")
    parser.add_argument("--no-project", action="store_true",
                        help="return the decoded sub-stems instead of masks on the parent's spectrogram "
                             "(with --stems, only those slots are generated: faster, but outputs do not sum to the parent)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.5, help="Stage 1 chunk overlap")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--format", choices=["wav", "flac"], default="wav")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
