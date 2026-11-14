"""Joint sub-stem Diffusion Transformer (Stage 2).

The model predicts the flow-matching velocity for *all* sub-stems of one
parent stem at once. Tokens live on a (stem slot x time) grid and every block
attends along both axes:

* **time attention** inside each slot, with rotary position embedding;
* **stem attention** across slots at each time step, *without* positional
  encoding - slots are an unordered set, told apart only by their class
  embedding - and with padded slots masked out as keys.

Conditioning:

* the parent stem's latent is concatenated to every slot's noisy latent
  along the channel axis;
* the slot's class embedding is added to its tokens and, together with the
  flow time embedding, drives adaLN-Zero modulation in every block.

Label 0 is the null class (padding slots and classifier-free guidance).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .roformer import RotaryEmbedding


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the flow time ``t in [0, 1]`` followed by an MLP."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256, time_scale: float = 1000.0):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.time_scale = time_scale
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * self.time_scale * freqs[None]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class AxisAttention(nn.Module):
    """Multi-head self-attention over ``(N, L, H)`` sequences."""

    def __init__(self, hidden_dim: int, num_heads: int, rotary: Optional[RotaryEmbedding] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.rotary = rotary

    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``key_mask``: ``(N, L)`` bool, True = this position may be attended to."""
        n, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(n, length, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.rotary is not None:
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)
        mask = key_mask[:, None, None, :] if key_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(n, length, dim))


class JointAxialDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, time_rotary: RotaryEmbedding, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.time_attn = AxisAttention(hidden_dim, num_heads, rotary=time_rotary, dropout=dropout)
        self.stem_attn = AxisAttention(hidden_dim, num_heads, rotary=None, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # adaLN-Zero: (shift, scale, gate) x (time attention, stem attention, MLP)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``x``: (B, S, T, H); ``cond``: (B, S, H); ``key_mask``: (B, S) bool or None."""
        b, s, t, h = x.shape
        params = self.adaLN_modulation(cond).reshape(b, s, 1, 9, h).unbind(dim=3)
        shift_t, scale_t, gate_t, shift_s, scale_s, gate_s, shift_m, scale_m, gate_m = params

        y = self.norm1(x) * (1 + scale_t) + shift_t
        y = self.time_attn(y.reshape(b * s, t, h)).reshape(b, s, t, h)
        x = x + gate_t * y

        y = self.norm2(x) * (1 + scale_s) + shift_s
        y = y.transpose(1, 2).reshape(b * t, s, h)
        mask = key_mask[:, None, :].expand(b, t, s).reshape(b * t, s) if key_mask is not None else None
        y = self.stem_attn(y, mask).reshape(b, t, s, h).transpose(1, 2)
        x = x + gate_s * y

        y = self.norm3(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(y)


class JointSubStemDiT(nn.Module):
    """
    Args:
        in_channels: latent channels per slot (2 x VAE latent width for stereo).
        hidden_dim, num_heads, depth: transformer size.
        num_stem_classes: size of the label vocabulary *including* the null label 0.
        cond_channels: channels of the parent latent (default ``in_channels``).
        use_checkpoint: gradient checkpointing over blocks (saves memory when training).

    ``forward(x, t, stem_labels, parent_latent, valid_mask=None)``:
        x              (B, S, C, T)  noisy latents
        t              (B,)          flow time in [0, 1] (0 = noise, 1 = data)
        stem_labels    (B, S) long   class id per slot, 0 = null
        parent_latent  (B, C_cond, T)
        valid_mask     (B, S) bool   False for padding slots
    returns the velocity, (B, S, C, T).
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 12,
        num_stem_classes: int = 15,
        cond_channels: Optional[int] = None,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = in_channels if cond_channels is None else cond_channels
        self.num_stem_classes = num_stem_classes
        self.use_checkpoint = use_checkpoint

        self.in_proj = nn.Linear(in_channels + self.cond_channels, hidden_dim)
        self.stem_embedding = nn.Embedding(num_stem_classes, hidden_dim)        # added to tokens
        self.cond_stem_embedding = nn.Embedding(num_stem_classes, hidden_dim)   # added to adaLN conditioning
        self.time_embedder = TimestepEmbedder(hidden_dim)

        self.time_rotary = RotaryEmbedding(hidden_dim // num_heads)             # shared by all blocks
        self.blocks = nn.ModuleList(
            [JointAxialDiTBlock(hidden_dim, num_heads, self.time_rotary, ff_mult, dropout) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, in_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.stem_embedding.weight, std=0.02)
        nn.init.normal_(self.cond_stem_embedding.weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        for layer in (self.out_modulation[1], self.out_proj):   # zero output: v = 0 at init
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        stem_labels: torch.Tensor,
        parent_latent: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, c, frames = x.shape
        if t.dim() == 0:
            t = t.expand(b)
        parent = parent_latent[:, None].expand(b, s, parent_latent.shape[1], frames)
        tokens = torch.cat([x, parent.to(x.dtype)], dim=2).permute(0, 1, 3, 2)   # (B, S, T, C + C_cond)
        hidden = self.in_proj(tokens) + self.stem_embedding(stem_labels)[:, :, None]
        cond = self.time_embedder(t)[:, None] + self.cond_stem_embedding(stem_labels)   # (B, S, H)

        key_mask = None
        if valid_mask is not None and not bool(valid_mask.all()):
            key_mask = valid_mask.bool()
            # A row with no valid slot would attend to nothing (NaN); let it see everything.
            key_mask = key_mask | ~key_mask.any(dim=1, keepdim=True)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                hidden = checkpoint(block, hidden, cond, key_mask, use_reentrant=False)
            else:
                hidden = block(hidden, cond, key_mask)

        shift, scale = self.out_modulation(cond)[:, :, None].chunk(2, dim=-1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.out_proj(hidden).permute(0, 1, 3, 2)


def build_dit(model_config: Mapping[str, Any], in_channels: int, num_stem_classes: int) -> JointSubStemDiT:
    """Builds the DiT from a tier-resolved ``model:`` section. ``in_channels``
    and ``num_stem_classes`` come from the latent manifest, not the config."""
    config = dict(model_config)
    for key in ("tier", "type", "in_channels", "num_stem_classes"):
        config.pop(key, None)
    return JointSubStemDiT(in_channels=in_channels, num_stem_classes=num_stem_classes, **config)
