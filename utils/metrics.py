"""Separation metrics.

Three different numbers are all called "SDR" in the source-separation
literature. They are *not* interchangeable, so each has its own function here:

``sdr`` (a.k.a. uSDR / "utterance-level" SDR)
    ``10 log10(|s|^2 / |s - s_hat|^2)`` over a whole track. Definition used by
    the Music Demixing / Sound Demixing Challenges.

``chunked_sdr``
    The same ratio computed on consecutive 1-second windows. This is a fast
    proxy only: it does **not** apply BSSEval's distortion filter and therefore
    does **not** reproduce museval numbers.

``museval_sdr`` (a.k.a. cSDR)
    BSSEval v4 through the ``museval`` package: median over 1-second frames per
    track; the benchmark figure is then the median over tracks. This is the
    protocol behind the MUSDB18HQ tables of the BS-RoFormer paper (including
    the 9.80 dB average), and the only function in this file whose output may
    be compared with those tables.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Dict, Iterable, List, Optional

import torch


def sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Global SDR in dB over all channels and samples of one source."""
    reference = reference.double()
    estimate = estimate.double()
    length = min(reference.shape[-1], estimate.shape[-1])
    reference, estimate = reference[..., :length], estimate[..., :length]
    num = reference.pow(2).sum().item() + eps
    den = (reference - estimate).pow(2).sum().item() + eps
    return 10.0 * math.log10(num / den)


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB (signals flattened over channels)."""
    reference = reference.double().reshape(-1)
    estimate = estimate.double().reshape(-1)
    length = min(reference.numel(), estimate.numel())
    reference, estimate = reference[:length], estimate[:length]
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = (estimate @ reference) / (reference @ reference + eps)
    target = scale * reference
    noise = estimate - target
    return 10.0 * math.log10((target.pow(2).sum().item() + eps) / (noise.pow(2).sum().item() + eps))


def chunked_sdr(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    window_seconds: float = 1.0,
    eps: float = 1e-8,
) -> List[float]:
    """Per-window SDR values (dB). Windows where the reference is silent are skipped.

    Proxy metric - see the module docstring; not comparable with museval.
    """
    length = min(reference.shape[-1], estimate.shape[-1])
    window = max(1, int(round(window_seconds * sample_rate)))
    values: List[float] = []
    for start in range(0, length - window + 1, window):
        ref = reference[..., start:start + window].double()
        est = estimate[..., start:start + window].double()
        ref_energy = ref.pow(2).sum().item()
        if ref_energy <= eps:
            continue
        err_energy = (ref - est).pow(2).sum().item()
        values.append(10.0 * math.log10((ref_energy + eps) / (err_energy + eps)))
    return values


def chunked_sdr_median(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> float:
    values = chunked_sdr(reference, estimate, sample_rate)
    return float(median(values)) if values else float("nan")


def museval_available() -> bool:
    try:
        import museval  # noqa: F401
    except Exception:
        return False
    return True


def museval_sdr(
    references: Dict[str, torch.Tensor],
    estimates: Dict[str, torch.Tensor],
    sample_rate: int = 44100,
) -> Dict[str, float]:
    """BSSEval v4 SDR per source for one track: median over 1-second frames.

    ``references`` / ``estimates`` map source name -> ``(C, T)``. All sources of
    the track should be passed together, exactly as museval does for MUSDB.
    Requires ``pip install museval``.
    """
    try:
        import museval
        import numpy as np
    except Exception as error:  # pragma: no cover - optional dependency
        raise RuntimeError("museval is not installed; run `pip install museval`") from error

    names = [name for name in references if name in estimates]
    if not names:
        raise ValueError("no common sources between references and estimates")
    length = min(min(references[n].shape[-1], estimates[n].shape[-1]) for n in names)
    ref = np.stack([references[n][..., :length].T.double().cpu().numpy() for n in names])  # (S, T, C)
    est = np.stack([estimates[n][..., :length].T.double().cpu().numpy() for n in names])
    sdr_frames, _, _, _ = museval.evaluate(ref, est, win=sample_rate, hop=sample_rate)
    return {name: float(np.nanmedian(sdr_frames[index])) for index, name in enumerate(names)}


def aggregate(per_track: Iterable[Dict[str, float]], how: str = "median") -> Dict[str, float]:
    """Aggregates per-track dictionaries of scores into one score per source.

    ``how='median'`` matches the MUSDB reporting convention (median over
    tracks); ``how='mean'`` matches the SDX challenge convention. NaNs (e.g. a
    silent reference) are ignored. An ``avg`` entry holds the mean across sources.
    """
    collected: Dict[str, List[float]] = {}
    for scores in per_track:
        for name, value in scores.items():
            if value is None or math.isnan(value):
                continue
            collected.setdefault(name, []).append(value)
    result: Dict[str, float] = {}
    for name, values in collected.items():
        result[name] = float(median(values)) if how == "median" else float(sum(values) / len(values))
    if result:
        result["avg"] = float(sum(result.values()) / len(result))
    return result


def format_table(scores: Dict[str, float], title: Optional[str] = None) -> str:
    lines = [title] if title else []
    width = max((len(name) for name in scores), default=4)
    for name, value in scores.items():
        lines.append(f"  {name:<{width}}  {value:7.3f} dB")
    return "\n".join(lines)
