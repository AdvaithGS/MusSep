"""Transformer building blocks for BS-RoFormer (Stage 1) and the latent DiT (Stage 2).

The module / attribute names in this file deliberately mirror the public
BS-RoFormer implementations (lucidrains/BS-RoFormer and the copy used by
ZFTurbo's Music-Source-Separation-Training), so that their checkpoints load
with ``load_state_dict(strict=True)``:

    RMSNorm.gamma
    FeedForward.net.{0: RMSNorm, 1: Linear, 4: Linear}
    Attention.{norm, to_qkv, to_gates, to_out.0, rotary_embed.freqs}
    Transformer.{layers.{i}.{0: Attention, 1: FeedForward}, norm}

Do not rename attributes here without also updating
``models.bs_roformer.load_community_state_dict``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """``x / ||x|| * sqrt(dim) * gamma`` (no bias, no mean subtraction)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), numerically identical to
    ``rotary_embedding_torch.RotaryEmbedding(dim)`` with default arguments.

    * ``freqs = theta ** -(arange(0, dim, 2) / dim)`` is stored as a frozen
      ``nn.Parameter`` so that it appears in the state dict under the same key
      as in community checkpoints.
    * Feature pairs are *interleaved*: features ``(2i, 2i + 1)`` are rotated by
      ``pos * freqs[i]``. (This is not the "split in two halves" convention.)
    * The rotation is evaluated in float32 with autocast disabled and cast back
      to the input dtype.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding needs an even dimension, got {dim}")
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def _angles(self, seq_len: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32) + offset
        angles = torch.einsum("n,f->nf", positions, self.freqs.float())
        return angles.repeat_interleave(2, dim=-1)  # (n, dim): f0 f0 f1 f1 ...

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def rotate_queries_or_keys(self, t: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotates ``t`` of shape ``(..., seq, dim)`` along its second-to-last axis."""
        if t.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {t.shape[-1]}")
        dtype = t.dtype
        with torch.autocast(device_type=t.device.type, enabled=False):
            angles = self._angles(t.shape[-2], t.device, offset)
            t32 = t.float()
            out = t32 * angles.cos() + self._rotate_half(t32) * angles.sin()
        return out.to(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4, dropout: float = 0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Pre-norm multi-head self-attention with RoPE and per-head output gates.

    ``flash`` is accepted for configuration compatibility; attention always goes
    through ``F.scaled_dot_product_attention``, which picks the fastest kernel
    available (and is mathematically the same as the explicit softmax path).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.flash = flash
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: (batch, seq, dim). ``attn_mask``: optional boolean mask
        broadcastable to (batch, heads, seq, seq); True = may attend."""
        x = self.norm(x)
        batch, seq, _ = x.shape

        # 'b n (qkv h d) -> qkv b h n d'
        qkv = self.to_qkv(x).reshape(batch, seq, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )

        gates = self.to_gates(x)  # (b, n, h)
        out = out * gates.permute(0, 2, 1).unsqueeze(-1).sigmoid()

        out = out.transpose(1, 2).reshape(batch, seq, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: float = 4,
        norm_output: bool = True,
        rotary_embed: Optional[RotaryEmbedding] = None,
        flash_attn: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                          rotary_embed=rotary_embed, flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
