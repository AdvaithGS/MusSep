"""Stage 1 training loss: waveform L1 + multi-resolution complex STFT L1."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionComplexSTFTLoss(nn.Module):
    """Time-domain MAE plus complex-spectrogram MAE at several resolutions.

    Follows the BS-RoFormer objective: window sizes 4096/2048/1024/512/256 with
    a fixed hop of 147 samples. The spectral term is the L1 distance of the real
    and imaginary parts.

    Inputs may have any leading shape ``(..., T)`` (e.g. ``(B, C, T)`` or
    ``(B, N, C, T)``). Everything is computed in float32 with autocast disabled:
    half-precision STFTs produce NaN gradients.
    """

    def __init__(
        self,
        window_sizes: Sequence[int] = (4096, 2048, 1024, 512, 256),
        hop_length: int = 147,
        stft_weight: float = 1.0,
    ):
        super().__init__()
        self.window_sizes = tuple(int(size) for size in window_sizes)
        self.hop_length = int(hop_length)
        self.stft_weight = float(stft_weight)
        for size in self.window_sizes:
            self.register_buffer(f"window_{size}", torch.hann_window(size), persistent=False)

    def _stft(self, wave: torch.Tensor, size: int) -> torch.Tensor:
        window = getattr(self, f"window_{size}").to(device=wave.device, dtype=torch.float32)
        return torch.stft(wave, n_fft=size, hop_length=self.hop_length, window=window, return_complex=True)

    def forward(
        self,
        pred_wave: torch.Tensor,
        target_wave: torch.Tensor,
        return_breakdown: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        length = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred = pred_wave[..., :length].float()
        target = target_wave[..., :length].float()

        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss_wave = F.l1_loss(pred, target)

            pred_flat = pred.reshape(-1, length)
            target_flat = target.reshape(-1, length)
            loss_stft = pred.new_zeros(())
            for size in self.window_sizes:
                pred_spec = torch.view_as_real(self._stft(pred_flat, size))
                target_spec = torch.view_as_real(self._stft(target_flat, size))
                # mean over (real, imag) pairs * 2 == L1(real) + L1(imag)
                loss_stft = loss_stft + 2.0 * F.l1_loss(pred_spec, target_spec)

            total = loss_wave + self.stft_weight * loss_stft

        if return_breakdown:
            return total, {"wave": loss_wave.detach(), "stft": loss_stft.detach()}
        return total
