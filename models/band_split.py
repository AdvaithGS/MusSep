"""Band-split front end and band-wise mask estimator of BS-RoFormer.

Attribute names follow the public implementations so that community
checkpoints load unchanged (``band_split.to_features.{i}.{0,1}`` and
``mask_estimators.{n}.to_freqs.{i}.0.{0,2}``).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .roformer import RMSNorm

# Number of STFT bins in each band, for n_fft = 2048 at 44.1 kHz (1025 bins):
#   24 bands x  2 bins   (< ~1 kHz)
#   12 bands x  4 bins   (~1-2 kHz)
#    8 bands x 12 bins   (~2-4 kHz)
#    8 bands x 24 bins   (~4-8 kHz)
#    8 bands x 48 bins   (~8-16 kHz)
#    2 bands (128 + 129) (> ~16 kHz)
# = 62 bands, 1025 bins. Same table as the community implementations.
DEFAULT_FREQS_PER_BANDS: Tuple[int, ...] = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


def bands_from_widths(freqs_per_bands: Sequence[int]) -> List[Tuple[int, int]]:
    """``(start, end)`` bin ranges (end exclusive) for a list of band widths."""
    bands, start = [], 0
    for width in freqs_per_bands:
        bands.append((start, start + int(width)))
        start += int(width)
    return bands


def get_bs_roformer_bands() -> List[Tuple[int, int]]:
    """The 62-band split of BS-RoFormer as ``(start, end)`` bin ranges.

    Covers every bin from 0 (DC) to 1024 exactly once.
    """
    return bands_from_widths(DEFAULT_FREQS_PER_BANDS)


def validate_band_widths(freqs_per_bands: Sequence[int], num_bins: int) -> Tuple[int, ...]:
    widths = tuple(int(width) for width in freqs_per_bands)
    if len(widths) < 2:
        raise ValueError("at least two bands are required")
    if any(width <= 0 for width in widths):
        raise ValueError("band widths must be positive")
    if sum(widths) != num_bins:
        raise ValueError(
            f"band widths sum to {sum(widths)} but the STFT has {num_bins} bins "
            f"(n_fft // 2 + 1); adjust `freqs_per_bands` or `stft_n_fft`"
        )
    return widths


class BandSplit(nn.Module):
    """Projects each sub-band of the (real, imag) spectrogram to ``dim`` features.

    Input  ``(B, T, sum(dim_inputs))`` - bins flattened as (freq, channel, re/im).
    Output ``(B, T, num_bands, dim)``.
    """

    def __init__(self, dim: int, dim_inputs: Sequence[int]):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.to_features = nn.ModuleList([
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in self.dim_inputs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = x.split(self.dim_inputs, dim=-1)
        return torch.stack([net(piece) for piece, net in zip(pieces, self.to_features)], dim=-2)


def MLP(dim_in: int, dim_out: int, dim_hidden: int | None = None, depth: int = 1, activation=nn.Tanh) -> nn.Sequential:
    """Linear / activation chain with ``depth`` linear layers (no final activation)."""
    dim_hidden = dim_hidden if dim_hidden is not None else dim_in
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers: List[nn.Module] = []
    for index, (layer_in, layer_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(layer_in, layer_out))
        if index < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    """Per-band MLP + GLU producing the (real, imag) mask of every bin.

    Input  ``(B, T, num_bands, dim)``.
    Output ``(B, T, sum(dim_inputs))`` - same flattening as :class:`BandSplit`.
    """

    def __init__(self, dim: int, dim_inputs: Sequence[int], depth: int, mlp_expansion_factor: int = 4):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([
            nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1))
            for dim_in in self.dim_inputs
        ])

    def output_layers(self) -> List[nn.Linear]:
        """Last linear layer of every band (used for the optional near-zero init)."""
        return [branch[0][-1] for branch in self.to_freqs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = x.unbind(dim=-2)
        return torch.cat([net(band) for band, net in zip(bands, self.to_freqs)], dim=-1)
