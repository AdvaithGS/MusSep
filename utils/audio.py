"""Audio I/O, STFT helpers, chunked (overlap-average) inference and mask projection.

Everything spectral in this module runs in float32 with autocast disabled:
half-precision STFT/iSTFT is numerically unsafe (see the precision notes in the
README).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Level helpers
# --------------------------------------------------------------------------- #

def db_to_amp(db: float | torch.Tensor) -> float | torch.Tensor:
    return 10.0 ** (db / 20.0)


def rms_db(audio: torch.Tensor, eps: float = 1e-10) -> float:
    """RMS level of a whole tensor in dBFS."""
    power = audio.float().pow(2).mean().item()
    return 10.0 * math.log10(power + eps)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def audio_info(path: str | Path) -> Tuple[int, int, int]:
    """Returns ``(num_frames, sample_rate, channels)`` without decoding."""
    info = sf.info(str(path))
    return info.frames, info.samplerate, info.channels


def to_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    """Converts ``(C, T)`` audio to the requested channel count.

    mono -> stereo duplicates the channel, stereo -> mono averages, anything
    wider than stereo keeps the first two channels.
    """
    current = audio.shape[0]
    if current == channels:
        return audio
    if current == 1:
        return audio.expand(channels, -1).clone()
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    if current > channels:
        return audio[:channels]
    raise ValueError(f"cannot convert {current} channels to {channels}")


def resample(audio: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Band-limited resampling along the last axis (no-op if the rates match)."""
    if orig_sr == new_sr:
        return audio
    import torchaudio.functional as AF

    return AF.resample(
        audio.float(),
        orig_freq=orig_sr,
        new_freq=new_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def load_audio(
    path: str | Path,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    start: int = 0,
    frames: int = -1,
) -> Tuple[torch.Tensor, int]:
    """Reads (part of) an audio file as float32 ``(C, T)``.

    ``start`` / ``frames`` are expressed in samples of the *file's* sample rate.
    If ``sample_rate`` is given the result is resampled to it.
    """
    data, file_sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    audio = torch.from_numpy(np.ascontiguousarray(data.T))
    if channels is not None:
        audio = to_channels(audio, channels)
    if sample_rate is not None and sample_rate != file_sr:
        audio = resample(audio, file_sr, sample_rate)
        file_sr = sample_rate
    return audio, file_sr


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int, subtype: str = "PCM_16") -> None:
    """Writes ``(C, T)`` or ``(T,)`` audio. Integer subtypes are clipped to [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.detach().float().cpu()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if subtype.upper().startswith("PCM"):
        data = data.clamp(-1.0, 1.0)
    sf.write(str(path), data.T.numpy(), sample_rate, subtype=subtype)


# --------------------------------------------------------------------------- #
# STFT
# --------------------------------------------------------------------------- #

def _autocast_off(tensor: torch.Tensor):
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Complex STFT of ``(..., T)`` audio -> ``(..., F, frames)`` (float32 / complex64)."""
    win_length = win_length or n_fft
    lead = audio.shape[:-1]
    flat = audio.reshape(-1, audio.shape[-1]).float()
    if window is None:
        window = torch.hann_window(win_length, device=flat.device, dtype=torch.float32)
    with _autocast_off(flat):
        spec = torch.stft(
            flat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window.float(), return_complex=True,
        )
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Inverse of :func:`stft`: ``(..., F, frames)`` -> ``(..., T)``."""
    win_length = win_length or n_fft
    lead = spec.shape[:-2]
    flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1])
    if flat.dtype != torch.complex64:
        flat = flat.to(torch.complex64)
    if window is None:
        window = torch.hann_window(win_length, device=flat.device, dtype=torch.float32)
    with _autocast_off(flat):
        audio = torch.istft(
            flat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window.float(), length=length,
        )
    return audio.reshape(*lead, audio.shape[-1])


# --------------------------------------------------------------------------- #
# Chunked inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def chunked_apply(
    fn: Callable[[torch.Tensor], torch.Tensor],
    audio: torch.Tensor,
    chunk_size: int,
    hop_size: int,
    batch_size: int = 1,
    window: str = "flat",
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Runs ``fn`` over overlapping chunks of a long signal and overlap-averages.

    ``audio`` is ``(C, T)``. ``fn`` maps a batch ``(B, C, chunk_size)`` to
    ``(B, N, C, chunk_size)`` (N outputs per input). Returns ``(N, C, T)`` on
    the CPU.

    The signal is zero-padded by ``chunk_size - hop_size`` on both sides so that
    every sample is covered by the same number of chunks. ``window='flat'`` is
    the plain overlap-average deframing used in the BS-RoFormer paper;
    ``window='hann'`` weights chunk centres more than chunk edges, which hides
    seams better when chunk estimates disagree.
    """
    if hop_size <= 0 or hop_size > chunk_size:
        raise ValueError("hop_size must be in (0, chunk_size]")
    channels, total = audio.shape
    border = chunk_size - hop_size
    num_chunks = max(1, math.ceil((total + border) / hop_size))
    padded_len = (num_chunks - 1) * hop_size + chunk_size
    padded = F.pad(audio.float(), (border, padded_len - border - total))

    if window == "flat":
        weight = torch.ones(chunk_size)
    elif window == "hann":
        # strictly positive so that samples covered by a single chunk stay defined
        weight = torch.hann_window(chunk_size, periodic=False).clamp_min(1e-3)
    else:
        raise ValueError(f"unknown window '{window}'")

    starts = [index * hop_size for index in range(num_chunks)]
    out: Optional[torch.Tensor] = None
    norm = torch.zeros(padded_len)

    for batch_start in range(0, num_chunks, batch_size):
        batch_offsets = starts[batch_start:batch_start + batch_size]
        batch = torch.stack([padded[:, off:off + chunk_size] for off in batch_offsets])
        if device is not None:
            batch = batch.to(device)
        result = fn(batch).float().cpu()
        if result.ndim != 4 or result.shape[-1] != chunk_size:
            raise ValueError(f"fn must return (B, N, C, {chunk_size}), got {tuple(result.shape)}")
        if out is None:
            out = torch.zeros(result.shape[1], result.shape[2], padded_len)
        for item, off in zip(result, batch_offsets):
            out[..., off:off + chunk_size] += item * weight
            norm[off:off + chunk_size] += weight

    assert out is not None
    out = out / norm.clamp_min(1e-8)
    return out[..., border:border + total]


# --------------------------------------------------------------------------- #
# Projection of generated sub-stems onto the parent stem
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ratio_mask_projection(
    parent: torch.Tensor,
    estimates: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
    eps: float = 1e-8,
    segment: int = 44100 * 30,
) -> torch.Tensor:
    """Re-synthesises sub-stems as soft masks applied to the parent's STFT.

    ``parent`` is ``(C, T)`` (the Stage 1 stem), ``estimates`` is ``(S, C, T)``
    (decoded generative outputs, used only as magnitude guides). Each output is
    ``mask_i * STFT(parent)`` with ``mask_i = |E_i|^p / sum_j |E_j|^p``, so

    * the phase and the fine structure come from the real recording, and
    * the outputs sum to the parent (up to iSTFT rounding) wherever any guide
      has energy.

    ``power=2`` gives a Wiener-like mask, ``power=1`` a magnitude-ratio mask.
    The signal is processed in independent segments to bound memory.
    """
    if estimates.ndim != 3 or parent.ndim != 2:
        raise ValueError("expected parent (C, T) and estimates (S, C, T)")
    total = parent.shape[-1]
    if estimates.shape[-1] != total:
        length = min(estimates.shape[-1], total)
        estimates = F.pad(estimates[..., :length], (0, total - length))
    segment = max(segment - segment % hop_length, n_fft)
    out = torch.zeros_like(estimates, dtype=torch.float32)
    window = torch.hann_window(n_fft, device=parent.device)

    for start in range(0, total, segment):
        stop = min(start + segment, total)
        parent_seg = parent[..., start:stop].float()
        est_seg = estimates[..., start:stop].float()
        seg_len = stop - start
        if seg_len < n_fft:  # too short for a centred reflect-padded STFT
            parent_seg = F.pad(parent_seg, (0, n_fft - seg_len))
            est_seg = F.pad(est_seg, (0, n_fft - seg_len))
        parent_spec = stft(parent_seg, n_fft, hop_length, window=window)       # (C, F, N)
        est_mag = stft(est_seg, n_fft, hop_length, window=window).abs().pow(power)  # (S, C, F, N)
        denom = est_mag.sum(dim=0, keepdim=True)
        # where no guide has energy, share the parent equally so the sum is preserved
        mask = torch.where(denom > eps, est_mag / denom.clamp_min(eps), torch.full_like(est_mag, 1.0 / est_mag.shape[0]))
        projected = istft(mask * parent_spec.unsqueeze(0), n_fft, hop_length, window=window, length=parent_seg.shape[-1])
        out[..., start:stop] = projected[..., :seg_len]
    return out
