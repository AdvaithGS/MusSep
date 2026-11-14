"""Frozen audio VAE used as the latent space of Stage 2.

:class:`DACVAEWrapper` adapts Meta's continuous DAC-VAE
(``facebook/dacvae-watermarked``, package ``dacvae`` from
https://github.com/facebookresearch/dacvae) to MusSep's conventions:

* the VAE is mono, so each channel is encoded separately and the per-channel
  latents are concatenated: a stereo signal becomes ``(B, 2*D, F)`` laid out
  as ``[left (D), right (D)]``;
* audio is resampled to the VAE's rate on the way in and back on the way out;
* long signals are processed in overlapping chunks to bound memory;
* everything runs in float32 with autocast disabled, without gradients.

Sample rate, hop length and latent width are read from the loaded model.
Nothing in this repository hard-codes them.

:class:`ToyVAE` is a tiny linear frame codec with the same interface, for
tests and CPU smoke runs. It is not a usable audio codec.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.audio import resample

DEFAULT_VAE = "facebook/dacvae-watermarked"


def load_dacvae(model_name: str = DEFAULT_VAE) -> nn.Module:
    """Loads the DAC-VAE through its own package (weights come from the
    Hugging Face Hub for ``facebook/...`` names, or from a local path)."""
    try:
        import dacvae  # noqa: WPS433
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'dacvae' package is required for Stage 2. Install it with\n"
            "    pip install git+https://github.com/facebookresearch/dacvae\n"
            "(it is not the Hugging Face 'transformers' AutoModel API)."
        ) from error
    return dacvae.DACVAE.load(model_name)


class ToyVAE(nn.Module):
    """Orthonormal frame projection: ``hop_length`` samples -> ``latent_dim`` numbers.

    Lossless when ``latent_dim == hop_length``. Deterministic for a given seed.
    """

    def __init__(self, sample_rate: int = 16000, hop_length: int = 320, latent_dim: int = 8, seed: int = 0):
        super().__init__()
        if latent_dim > hop_length:
            raise ValueError("latent_dim must not exceed hop_length")
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.latent_dim = latent_dim
        generator = torch.Generator().manual_seed(seed)
        basis, _ = torch.linalg.qr(torch.randn(hop_length, hop_length, generator=generator))
        self.register_buffer("basis", basis[:latent_dim].contiguous())   # (D, hop), orthonormal rows

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T) -> (B, D, ceil(T / hop))``"""
        pad = (-audio.shape[-1]) % self.hop_length
        frames = F.pad(audio, (0, pad)).reshape(audio.shape[0], -1, self.hop_length)
        return (frames @ self.basis.t()).transpose(1, 2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, D, F) -> (B, 1, F * hop)``"""
        frames = latents.transpose(1, 2) @ self.basis
        return frames.reshape(latents.shape[0], 1, -1)


def build_vae(name: str = DEFAULT_VAE, **wrapper_kwargs) -> "DACVAEWrapper":
    """``"toy"`` or ``"toy:<sample_rate>:<hop>:<latent_dim>"`` builds a
    :class:`ToyVAE` (smoke tests); anything else is a DAC-VAE name or path."""
    if name == "toy" or name.startswith("toy:"):
        numbers = [int(part) for part in name.split(":")[1:]]
        return DACVAEWrapper(vae=ToyVAE(*numbers), **wrapper_kwargs)
    return DACVAEWrapper(model_name=name, **wrapper_kwargs)


class DACVAEWrapper(nn.Module):
    """See the module docstring.

    Args:
        model_name: DAC-VAE identifier, used when ``vae`` is not given.
        vae: an already constructed VAE. It must expose ``sample_rate``,
            ``hop_length``, ``latent_dim``, ``encode((B, 1, T)) -> (B, D, F)``
            and ``decode((B, D, F)) -> (B, 1, T')``.
        sample: if False (default) the posterior *mean* is used as the latent,
            which makes pre-encoded datasets deterministic. This needs the
            DAC-VAE internals (``encoder``, ``quantizer.in_proj``); VAEs that do
            not have them fall back to their own ``encode``.
        chunk_seconds / margin_seconds: long signals are encoded and decoded in
            pieces of ``chunk_seconds`` with ``margin_seconds`` of context on
            both sides that is discarded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VAE,
        vae: Optional[nn.Module] = None,
        sample: bool = False,
        chunk_seconds: float = 30.0,
        margin_seconds: float = 2.0,
    ):
        super().__init__()
        self.vae = vae if vae is not None else load_dacvae(model_name)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.sample = sample
        self.sample_rate = int(self.vae.sample_rate)
        self.hop_length = int(self.vae.hop_length)
        self.latent_dim = int(self.vae.latent_dim)
        self.chunk_frames = max(1, round(chunk_seconds * self.sample_rate / self.hop_length))
        self.margin_frames = max(0, round(margin_seconds * self.sample_rate / self.hop_length))
        self._warned = False

    # -- bookkeeping ------------------------------------------------------

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def device(self) -> torch.device:
        try:
            return next(self.vae.parameters()).device
        except StopIteration:
            return next(self.vae.buffers()).device

    def train(self, mode: bool = True) -> "DACVAEWrapper":
        return super().train(False)   # always frozen / eval

    def num_frames(self, num_samples: int, sample_rate: Optional[int] = None) -> int:
        """Latent frames produced for ``num_samples`` samples at ``sample_rate``."""
        if sample_rate is not None and sample_rate != self.sample_rate:
            num_samples = math.ceil(num_samples * self.sample_rate / sample_rate)
        return math.ceil(num_samples / self.hop_length)

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            warnings.warn(message, stacklevel=3)
            self._warned = True

    @staticmethod
    def _fit(tensor: torch.Tensor, length: int) -> torch.Tensor:
        """Crops or edge-pads the last dimension to ``length``."""
        if tensor.shape[-1] > length:
            return tensor[..., :length]
        if tensor.shape[-1] < length:
            return F.pad(tensor, (0, length - tensor.shape[-1]), mode="replicate")
        return tensor

    # -- mono primitives --------------------------------------------------

    def _encode_piece(self, audio: torch.Tensor) -> torch.Tensor:
        vae = self.vae
        if not self.sample and hasattr(vae, "quantizer") and hasattr(vae, "encoder"):
            padded = vae._pad(audio) if hasattr(vae, "_pad") else audio
            mean, _ = vae.quantizer.in_proj(vae.encoder(padded)).chunk(2, dim=1)
            return mean
        return vae.encode(audio)

    def _encode_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """``(N, 1, T)`` at the VAE rate, T a multiple of the hop -> ``(N, D, T / hop)``."""
        hop, core, margin = self.hop_length, self.chunk_frames, self.margin_frames
        total = audio.shape[-1] // hop
        device = self.device
        single = total <= core + 2 * margin
        pieces = []
        for start in ([0] if single else range(0, total, core)):
            stop = total if single else min(total, start + core)
            left = 0 if single else max(0, start - margin)
            right = total if single else min(total, stop + margin)
            piece = self._encode_piece(audio[..., left * hop: right * hop].to(device, torch.float32))
            if piece.shape[-1] != right - left:
                self._warn_once(
                    f"VAE returned {piece.shape[-1]} frames for {right - left} hops; cropping/padding. "
                    "Check hop_length against the model."
                )
                piece = self._fit(piece, right - left)
            pieces.append(piece[..., start - left: stop - left].to(audio.device))
        return torch.cat(pieces, dim=-1)

    def _decode_mono(self, latents: torch.Tensor) -> torch.Tensor:
        """``(N, D, F) -> (N, 1, F * hop)``"""
        hop, core, margin = self.hop_length, self.chunk_frames, self.margin_frames
        total = latents.shape[-1]
        device = self.device
        single = total <= core + 2 * margin
        pieces = []
        for start in ([0] if single else range(0, total, core)):
            stop = total if single else min(total, start + core)
            left = 0 if single else max(0, start - margin)
            right = total if single else min(total, stop + margin)
            piece = self.vae.decode(latents[..., left:right].to(device, torch.float32))
            if piece.shape[-1] != (right - left) * hop:
                self._warn_once(
                    f"VAE returned {piece.shape[-1]} samples for {right - left} frames "
                    f"(expected {(right - left) * hop}); cropping/padding."
                )
                piece = self._fit(piece, (right - left) * hop)
            pieces.append(piece[..., (start - left) * hop: (stop - left) * hop].to(latents.device))
        return torch.cat(pieces, dim=-1)

    # -- public API -------------------------------------------------------

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """``(B, C, T)`` or ``(C, T)`` waveform -> ``(B, C*D, F)`` / ``(C*D, F)`` latents.

        ``sample_rate`` is the rate of ``audio`` (default: the VAE's rate).
        ``F = ceil(T_vae / hop)`` where ``T_vae`` is the length after resampling.
        Resampling runs on the VAE's device; the result comes back on the input's.
        """
        squeeze = audio.dim() == 2
        if squeeze:
            audio = audio.unsqueeze(0)
        batch, channels, _ = audio.shape
        home = audio.device
        audio = audio.float().to(self.device)
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = resample(audio, sample_rate, self.sample_rate)
        audio = F.pad(audio, (0, (-audio.shape[-1]) % self.hop_length))
        with torch.autocast(device_type=self.device.type, enabled=False):
            latents = self._encode_mono(audio.reshape(batch * channels, 1, -1))
        latents = latents.reshape(batch, channels * latents.shape[1], latents.shape[-1]).to(home)
        return latents[0] if squeeze else latents

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        sample_rate: Optional[int] = None,
        length: Optional[int] = None,
        channels: int = 2,
    ) -> torch.Tensor:
        """``(B, C*D, F)`` or ``(C*D, F)`` latents -> ``(B, C, T)`` / ``(C, T)`` waveform.

        The result is resampled to ``sample_rate`` (if given) and cropped or
        zero-padded to ``length`` samples (if given).
        """
        squeeze = latents.dim() == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        batch, width, frames = latents.shape
        if width != channels * self.latent_dim:
            raise ValueError(f"expected {channels} x {self.latent_dim} latent channels, got {width}")
        home = latents.device
        latents = latents.float().to(self.device)                        # decode and resample on the VAE's device
        with torch.autocast(device_type=self.device.type, enabled=False):
            audio = self._decode_mono(latents.reshape(batch * channels, self.latent_dim, frames))
        audio = audio.reshape(batch, channels, -1)
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = resample(audio, self.sample_rate, sample_rate)
        if length is not None:
            audio = audio[..., :length] if audio.shape[-1] >= length else F.pad(audio, (0, length - audio.shape[-1]))
        audio = audio.to(home)
        return audio[0] if squeeze else audio

    @torch.no_grad()
    def silence_latent(self, channels: int = 2, seconds: float = 4.0) -> torch.Tensor:
        """Latent of digital silence, ``(C*D,)`` (mean over the middle frames)."""
        samples = max(self.hop_length * 8, int(seconds * self.sample_rate))
        latents = self.encode(torch.zeros(1, channels, samples))[0]
        quarter = latents.shape[-1] // 4
        return latents[:, quarter: latents.shape[-1] - quarter].mean(dim=-1)


class LatentNormalizer(nn.Module):
    """Per-channel standardisation of latents ``(..., C, T)``."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1).clamp_min(eps))

    @classmethod
    def identity(cls, channels: int) -> "LatentNormalizer":
        return cls(torch.zeros(channels), torch.ones(channels))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean.to(latents.dtype)) / self.std.to(latents.dtype)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.std.to(latents.dtype) + self.mean.to(latents.dtype)
