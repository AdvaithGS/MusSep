"""Stage 1: Band-Split RoFormer (Lu et al., "Music Source Separation with
Band-Split RoPE Transformer", 2023).

This implementation is layout-compatible with the public community code
(lucidrains/BS-RoFormer as vendored by ZFTurbo's
Music-Source-Separation-Training): same constructor keywords, same module
tree, same tensor layouts. A state dict produced by either loads here with
``strict=True`` and yields the same output, see ``load_community_state_dict``
and ``tests/test_bs_roformer.py``.

Differences that do not affect the weights:

* the forward pass always returns ``(B, num_stems, channels, T)``;
* no built-in loss (see ``utils.loss.MultiResolutionComplexSTFTLoss``); the
  ``multi_stft_*`` keywords are accepted and ignored so that community configs
  can be passed through unchanged;
* STFT, mask multiplication and iSTFT run in float32 with autocast disabled;
* ``linear_transformer_depth > 0`` (rarely used) is not implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .band_split import DEFAULT_FREQS_PER_BANDS, BandSplit, MaskEstimator, validate_band_widths
from .roformer import RMSNorm, RotaryEmbedding, Transformer


class BSRoformer(nn.Module):
    """Waveform-in / waveform-out BS-RoFormer.

    Args mirror the community implementation. The ones that matter most:

        dim, depth                     hidden size and number of (time, freq) layer pairs
        time_transformer_depth,
        freq_transformer_depth         transformer blocks inside each layer pair
        dim_head, heads                attention geometry (need not multiply to ``dim``)
        stereo, num_stems              audio channels in/out and number of mask heads
        freqs_per_bands                band widths in STFT bins; must sum to n_fft // 2 + 1
        stft_n_fft / hop / win         STFT geometry
        mask_estimator_depth,
        mlp_expansion_factor           mask MLP shape
        mask_init_std                  MusSep extension: if set, the last linear layer of
                                       every mask MLP is re-initialised with N(0, std) and
                                       zero bias, so that an untrained model starts from a
                                       near-zero mask. Only meaningful for training from
                                       scratch; it does not change the architecture.
    """

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        linear_transformer_depth: int = 0,
        freqs_per_bands: Sequence[int] = DEFAULT_FREQS_PER_BANDS,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        flash_attn: bool = True,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        stft_window_fn: Optional[Callable[..., torch.Tensor]] = None,
        zero_dc: bool = True,
        mask_estimator_depth: int = 2,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Sequence[int] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        multi_stft_window_fn: Optional[Callable[..., torch.Tensor]] = None,
        mlp_expansion_factor: int = 4,
        use_torch_checkpoint: bool = False,
        skip_connection: bool = False,
        mask_init_std: Optional[float] = None,
        **unused: Any,
    ):
        super().__init__()
        if linear_transformer_depth > 0:
            raise NotImplementedError(
                "linear_transformer_depth > 0 is not supported by this implementation "
                "(none of the commonly distributed BS-RoFormer checkpoints use it)"
            )
        ignorable = {"sample_rate", "use_pope", "use_shared_bias"}
        unknown = sorted(set(unused) - ignorable)
        if unknown:
            raise TypeError(f"BSRoformer got unexpected keyword arguments: {unknown}")

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.zero_dc = zero_dc

        num_bins = stft_n_fft // 2 + 1
        widths = validate_band_widths(freqs_per_bands, num_bins)
        self.freqs_per_bands = widths

        # One RoPE table per axis, shared by every layer (as in the reference).
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout,
            ff_dropout=ff_dropout, flash_attn=flash_attn, norm_output=False,
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))
        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized,
        )
        self.stft_win_length = stft_win_length
        self.stft_window_fn = stft_window_fn if stft_window_fn is not None else torch.hann_window

        # features per band: bins * channels * (real, imag)
        dims_per_band = tuple(2 * width * self.audio_channels for width in widths)
        self.band_split = BandSplit(dim=dim, dim_inputs=dims_per_band)
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dims_per_band, depth=mask_estimator_depth,
                          mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(num_stems)
        ])

        self.mask_init_std = mask_init_std
        if mask_init_std is not None:
            for estimator in self.mask_estimators:
                for layer in estimator.output_layers():
                    nn.init.normal_(layer.weight, mean=0.0, std=mask_init_std)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ #

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_torch_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """``raw_audio``: ``(B, T)`` (mono only) or ``(B, channels, T)``.

        Returns the separated stems as ``(B, num_stems, channels, T)``.
        """
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        batch, channels, length = raw_audio.shape
        if channels != self.audio_channels:
            raise ValueError(
                f"model was built with stereo={self.stereo} and expects {self.audio_channels} "
                f"channel(s), got {channels}"
            )
        device = raw_audio.device

        # ---- STFT (float32) ------------------------------------------------
        with torch.autocast(device_type=device.type, enabled=False):
            window = self.stft_window_fn(self.stft_win_length, device=device, dtype=torch.float32)
            spec = torch.stft(
                raw_audio.reshape(batch * channels, length).float(),
                **self.stft_kwargs, window=window, return_complex=True,
            )                                                   # (B*S, F, N) complex
            num_bins, frames = spec.shape[-2], spec.shape[-1]
            spec = torch.view_as_real(spec).reshape(batch, channels, num_bins, frames, 2)
            # 'b s f t c -> b (f s) t c': frequency-major, channels interleaved
            spec = spec.permute(0, 2, 1, 3, 4).reshape(batch, num_bins * channels, frames, 2)

        # 'b f t c -> b t (f c)'
        x = spec.permute(0, 2, 1, 3).reshape(batch, frames, num_bins * channels * 2)
        x = self._run(self.band_split, x)                       # (B, N, K, D)
        num_bands, dim = x.shape[2], x.shape[3]

        # ---- axial transformer ---------------------------------------------
        store: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        for index, (time_transformer, freq_transformer) in enumerate(self.layers):
            if self.skip_connection:
                for previous in range(index):
                    x = x + store[previous]

            # along time, one sequence per (batch, band)
            x = x.transpose(1, 2).reshape(batch * num_bands, frames, dim)
            x = self._run(time_transformer, x)
            # along frequency, one sequence per (batch, frame)
            x = x.reshape(batch, num_bands, frames, dim).transpose(1, 2).reshape(batch * frames, num_bands, dim)
            x = self._run(freq_transformer, x)
            x = x.reshape(batch, frames, num_bands, dim)

            if self.skip_connection:
                store[index] = x

        x = self.final_norm(x)

        # ---- masks ---------------------------------------------------------
        mask = torch.stack([self._run(estimator, x) for estimator in self.mask_estimators], dim=1)
        num_stems = mask.shape[1]

        with torch.autocast(device_type=device.type, enabled=False):
            # 'b n t (f c) -> b n f t c'
            mask = mask.float().reshape(batch, num_stems, frames, num_bins * channels, 2).permute(0, 1, 3, 2, 4)
            mask = torch.view_as_complex(mask.contiguous())
            spec_c = torch.view_as_complex(spec.contiguous()).unsqueeze(1)   # (B, 1, F*S, N)
            est = spec_c * mask                                              # (B, n, F*S, N)

            # 'b n (f s) t -> (b n s) f t'
            est = est.reshape(batch, num_stems, num_bins, channels, frames).permute(0, 1, 3, 2, 4)
            est = est.reshape(batch * num_stems * channels, num_bins, frames)
            if self.zero_dc:
                est = est.index_fill(1, torch.zeros(1, dtype=torch.long, device=device), 0.0)

            audio = torch.istft(est, **self.stft_kwargs, window=window, return_complex=False, length=length)

        return audio.reshape(batch, num_stems, channels, length)


# Alias with the capitalisation used in the paper / in earlier revisions of this repo.
BSRoFormer = BSRoformer


# ---------------------------------------------------------------------- #
# Construction and checkpoint helpers
# ---------------------------------------------------------------------- #

def build_bs_roformer(model_config: Mapping[str, Any]) -> BSRoformer:
    """Builds a model from a mapping of constructor keywords.

    Accepts the ``model:`` section of this repo's configs (after tier
    resolution) as well as the ``model:`` section of a community YAML.
    """
    config = dict(model_config)
    config.pop("tier", None)
    config.pop("type", None)
    if "freqs_per_bands" in config and config["freqs_per_bands"] is not None:
        config["freqs_per_bands"] = tuple(config["freqs_per_bands"])
    else:
        config.pop("freqs_per_bands", None)
    return BSRoformer(**config)


_STATE_KEYS = ("state_dict", "state", "model_state_dict", "model", "ema")


def extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    """Finds the tensor dictionary inside the usual checkpoint wrappers and
    strips a ``module.`` (DataParallel) prefix."""
    state = checkpoint_obj
    for _ in range(3):
        if isinstance(state, Mapping) and state and not all(torch.is_tensor(v) for v in state.values()):
            for key in _STATE_KEYS:
                if key in state and isinstance(state[key], Mapping):
                    state = state[key]
                    break
            else:
                break
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("could not find a state dict in the checkpoint")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return dict(state)


def load_community_state_dict(
    model: BSRoformer,
    checkpoint_or_path: Any,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Loads a community BS-RoFormer checkpoint into ``model``.

    ``checkpoint_or_path`` is a path to a ``.ckpt`` / ``.pth`` / ``.safetensors``
    file or an already loaded object. Files are read with
    ``torch.load(weights_only=True)``: public checkpoints are untrusted pickles.

    Returns ``(missing_keys, unexpected_keys)``. With ``strict=True`` any
    mismatch raises, with a message that usually points at a wrong config
    (``dim``, ``depth``, ``num_stems`` ...).

    Scientific caveat: nearly all publicly distributed BS-RoFormer checkpoints
    were trained with data beyond MUSDB18HQ. Results obtained with them must
    not be reported as "MUSDB18HQ-only" numbers and are not comparable with
    the 9.80 dB average SDR of the paper's MUSDB-only model.
    """
    obj = checkpoint_or_path
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file  # optional dependency

            obj = load_file(str(path))
        else:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
    state = extract_state_dict(obj)

    own = model.state_dict()
    missing = [key for key in own if key not in state]
    unexpected = [key for key in state if key not in own]
    wrong = [f"{key} {tuple(state[key].shape)} vs {tuple(own[key].shape)}"
             for key in own if key in state and state[key].shape != own[key].shape]
    if strict and (missing or unexpected or wrong):
        raise RuntimeError(
            "checkpoint does not match the model "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys, {len(wrong)} shape mismatches). "
            f"First missing: {missing[:5]}; first unexpected: {unexpected[:5]}; first mismatch: {wrong[:3]}. "
            "Check that the model config (dim, depth, num_stems, stereo, dim_head, heads, "
            "freqs_per_bands, mask_estimator_depth, mlp_expansion_factor) is the one the "
            "checkpoint was trained with."
        )
    model.load_state_dict({key: value for key, value in state.items() if key in own and value.shape == own[key].shape},
                          strict=False)
    return missing, unexpected
