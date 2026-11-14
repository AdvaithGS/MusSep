"""Compare our Stage 1 BS-RoFormer with the published reference implementation.

    pip install "BS-RoFormer==0.4.1"          # lucidrains' package; pick the release to compare with
    python -m scripts.compare_reference

Builds the reference ``bs_roformer.BSRoformer`` and ours with the same small
configuration, then reports:

1. weight names and shapes present in one model but not the other;
2. if the layouts match, the largest output difference on random audio after
   copying the reference's weights into ours (float32, eval mode).

A difference near 1e-6 means the forward passes agree, so checkpoints trained
with that layout behave the same here. Recent releases of the package (1.x)
add components such as hyper-connections that public checkpoints do not use,
so compare against the release closest to the checkpoints you intend to load.

This script imports and runs the third-party package; it is not part of the
test suite, and you choose whether to install and run it.
"""
from __future__ import annotations

import argparse
import inspect

import torch

from models.bs_roformer import BSRoformer as Ours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--stems", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    try:
        import bs_roformer
        from bs_roformer import BSRoformer as Reference
    except ImportError:
        print("the reference package is not installed: pip install BS-RoFormer==<version>")
        return 2

    version = getattr(bs_roformer, "__version__", "unknown")
    kwargs = dict(dim=args.dim, depth=args.depth, stereo=True, num_stems=args.stems, time_transformer_depth=1,
                  freq_transformer_depth=1, dim_head=8, heads=2)
    accepted = inspect.signature(Reference.__init__).parameters
    if "mask_estimator_depth" in accepted:
        kwargs["mask_estimator_depth"] = 2
    torch.manual_seed(0)
    reference = Reference(**kwargs).eval()
    ours = Ours(**{**kwargs, "mask_estimator_depth": kwargs.get("mask_estimator_depth", 2)}).eval()

    theirs, mine = reference.state_dict(), ours.state_dict()
    only_ref = sorted(set(theirs) - set(mine))
    only_ours = sorted(set(mine) - set(theirs))
    shapes = sorted(k for k in set(theirs) & set(mine) if theirs[k].shape != mine[k].shape)
    print(f"reference package version: {version}")
    print(f"weights: reference {len(theirs)}, ours {len(mine)}")
    print(f"only in reference ({len(only_ref)}): {only_ref[:8]}")
    print(f"only in ours ({len(only_ours)}): {only_ours[:8]}")
    print(f"shape mismatches ({len(shapes)}): {shapes[:8]}")
    if only_ref or only_ours or shapes:
        print("RESULT: layouts differ; a checkpoint from this release would not load. Try another release.")
        return 1

    ours.load_state_dict(theirs)
    audio = torch.randn(2, 2, int(args.seconds * 44100))
    with torch.no_grad():
        expected = reference(audio)
        got = ours(audio)
    if expected.shape != got.shape:
        expected = expected.reshape(got.shape)          # the reference squeezes the stem axis for one stem
    diff = float((expected - got).abs().max())
    scale = float(expected.abs().max())
    print(f"max |reference - ours| = {diff:.3e} (output scale {scale:.3e})")
    ok = diff <= 1e-4 * max(scale, 1e-6)
    print("RESULT: " + ("outputs match" if ok else "outputs differ: the forward passes are not equivalent"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
