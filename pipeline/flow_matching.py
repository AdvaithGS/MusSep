"""Optimal-transport conditional flow matching (OT-CFM) for joint sub-stem
generation, and the samplers that integrate the learned field.

Convention: ``t = 0`` is noise, ``t = 1`` is data.

    x_t = (1 - (1 - sigma_min) t) x_0 + t x_1,      x_0 ~ N(0, I)
    u_t = x_1 - (1 - sigma_min) x_0                  (regression target)

Tensors: latents ``(B, S, C, T)`` (S = stem slots), labels ``(B, S)``,
``valid_mask (B, S)`` (False = padding slot, excluded from the loss and from
stem attention), parent latent ``(B, C, T)``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def drop_labels(stem_labels: torch.Tensor, prob: float, null_label: int = 0) -> torch.Tensor:
    """Classifier-free-guidance dropout: with probability ``prob`` *per example*
    all of its labels are replaced by the null label."""
    if prob <= 0:
        return stem_labels
    drop = torch.rand(stem_labels.shape[0], device=stem_labels.device) < prob
    return torch.where(drop[:, None], torch.full_like(stem_labels, null_label), stem_labels)


def masked_slot_mean(per_slot: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of a ``(B, S)`` quantity over valid slots (0 with grad if there are none)."""
    if valid_mask is None:
        return per_slot.mean()
    mask = valid_mask.to(per_slot.dtype)
    return (per_slot * mask).sum() / mask.sum().clamp_min(1.0)


def predict_x1(x_t: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Data estimate from a point on the path and its velocity:
    ``x_1 ~= x_t + (1 - t) v`` (exact up to ``sigma_min * x_0``)."""
    return x_t + (1.0 - t.reshape(-1, 1, 1, 1)).to(x_t.dtype) * velocity


class MaskedOTFlowMatcher(nn.Module):
    """OT-CFM objective with a binary validity mask over stem slots.

    Args:
        sigma_min: width of the path at ``t = 1``.
        time_sampling: ``"uniform"`` or ``"logit_normal"``
            (``t = sigmoid(N(logit_mean, logit_std))``).
    """

    def __init__(
        self,
        sigma_min: float = 1e-5,
        time_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ):
        super().__init__()
        if time_sampling not in ("uniform", "logit_normal"):
            raise ValueError("time_sampling must be 'uniform' or 'logit_normal'")
        self.sigma_min = sigma_min
        self.time_sampling = time_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_time(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "uniform":
            return torch.rand(batch, device=device)
        return torch.sigmoid(torch.randn(batch, device=device) * self.logit_std + self.logit_mean)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_t, target velocity)``."""
        t = t.reshape(-1, 1, 1, 1).to(x1.dtype)
        x_t = (1.0 - (1.0 - self.sigma_min) * t) * x0 + t * x1
        return x_t, x1 - (1.0 - self.sigma_min) * x0

    def forward(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        parent_latent: torch.Tensor,
        stem_labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x0 = torch.randn_like(x1) if noise is None else noise
        if t is None:
            t = self.sample_time(x1.shape[0], x1.device)
        x_t, target = self.interpolate(x0, x1, t)
        pred = model(x_t, t, stem_labels, parent_latent, valid_mask)
        per_slot = (pred.float() - target.float()).pow(2).mean(dim=(2, 3))
        return masked_slot_mean(per_slot, valid_mask)


class ODESolver:
    """Samplers. ``model`` has the :class:`models.dit.JointSubStemDiT` call signature."""

    @staticmethod
    def _noise(shape: Sequence[int], like: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(tuple(shape), device=like.device, dtype=like.dtype, generator=generator)

    @staticmethod
    def _shape(shape, stem_labels: torch.Tensor, parent_latent: torch.Tensor) -> Tuple[int, ...]:
        if shape is not None:
            return tuple(shape)
        return (stem_labels.shape[0], stem_labels.shape[1], parent_latent.shape[1], parent_latent.shape[-1])

    @staticmethod
    def velocity(
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stem_labels: torch.Tensor,
        parent_latent: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        null_label: int = 0,
    ) -> torch.Tensor:
        """Velocity with classifier-free guidance over the class labels:
        ``v = v_null + cfg_scale * (v_cond - v_null)``. ``cfg_scale = 1`` is
        the plain conditional model (one network call instead of two)."""
        v_cond = model(x_t, t, stem_labels, parent_latent, valid_mask).float()
        if cfg_scale == 1.0:
            return v_cond
        v_null = model(x_t, t, torch.full_like(stem_labels, null_label), parent_latent, valid_mask).float()
        return v_null + cfg_scale * (v_cond - v_null)

    @staticmethod
    @torch.no_grad()
    def sample_euler(
        model: nn.Module,
        stem_labels: torch.Tensor,
        parent_latent: torch.Tensor,
        shape: Optional[Sequence[int]] = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        valid_mask: Optional[torch.Tensor] = None,
        null_label: int = 0,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Fixed-step Euler integration from ``t = 0`` to ``t = 1``. Returns ``(B, S, C, T)``."""
        shape = ODESolver._shape(shape, stem_labels, parent_latent)
        parent = parent_latent.float()
        x = ODESolver._noise(shape, parent, generator) if noise is None else noise.float()
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=parent.device)
        for i in range(num_steps):
            t = times[i].expand(shape[0])
            v = ODESolver.velocity(model, x, t, stem_labels, parent, valid_mask, cfg_scale, null_label)
            x = x + (times[i + 1] - times[i]) * v
        return x

    @staticmethod
    @torch.no_grad()
    def sample_consistency(
        model: nn.Module,
        stem_labels: torch.Tensor,
        parent_latent: torch.Tensor,
        shape: Optional[Sequence[int]] = None,
        num_steps: int = 2,
        valid_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_min: float = 1e-5,
    ) -> torch.Tensor:
        """Multi-step consistency sampling for a distilled student (1-4 steps).

        Each step jumps to a data estimate ``x_1 = x_t + (1 - t) v``; between
        steps the estimate is re-noised to the next time with fresh noise.
        """
        shape = ODESolver._shape(shape, stem_labels, parent_latent)
        parent = parent_latent.float()
        x = ODESolver._noise(shape, parent, generator) if noise is None else noise.float()
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=parent.device)
        x1 = x
        for i in range(num_steps):
            t = times[i].expand(shape[0])
            x1 = predict_x1(x, t, model(x, t, stem_labels, parent, valid_mask).float())
            if i + 1 < num_steps:
                t_next = times[i + 1]
                fresh = ODESolver._noise(shape, parent, generator)
                x = (1.0 - (1.0 - sigma_min) * t_next) * fresh + t_next * x1
        return x1
