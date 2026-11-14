"""Consistency distillation of the flow-matching teacher (Stage 4).

The student keeps the teacher's velocity parameterisation; its data estimate
``f(x_t, t) = x_t + (1 - t) v(x_t, t)`` satisfies the boundary condition
``f(x_1, 1) = x_1`` by construction. Training enforces self-consistency along
the teacher's probability-flow ODE, discretised into ``num_scales`` steps:

    t_n = n / N,   n ~ U{0, ..., N-1}
    x_{t_n}      on the OT path between noise and data
    x_{t_{n+1}}  = x_{t_n} + (t_{n+1} - t_n) * v_teacher(x_{t_n}, t_n)     (one Euler step)
    loss         = PseudoHuber( f_student(x_{t_n}, t_n) - f_ema(x_{t_{n+1}}, t_{n+1}) )

with the Pseudo-Huber metric ``sqrt(||d||^2 + c^2) - c`` computed per stem
slot in latent space and averaged over valid slots.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .flow_matching import ODESolver, masked_slot_mean, predict_x1


def pseudo_huber(difference: torch.Tensor, c: Optional[float] = None) -> torch.Tensor:
    """Pseudo-Huber distance per slot: ``(B, S, C, T) -> (B, S)``.

    ``c`` defaults to ``0.00054 * sqrt(C * T)`` (Song & Dhariwal, "Improved
    Techniques for Training Consistency Models").
    """
    difference = difference.float()
    if c is None:
        c = 0.00054 * math.sqrt(difference.shape[2] * difference.shape[3])
    return torch.sqrt(difference.pow(2).sum(dim=(2, 3)) + c * c) - c


class ConsistencyFlowDistiller(nn.Module):
    """
    Args:
        teacher_model: trained flow-matching DiT (frozen here).
        student_model: trainable DiT, normally initialised from the teacher.
        num_scales: discretisation ``N`` of the teacher trajectory.
        teacher_cfg_scale: classifier-free-guidance scale baked into the
            teacher's step. With a value other than 1 the student reproduces
            guided sampling without needing guidance itself.
        huber_c: Pseudo-Huber constant (``None`` = automatic).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        num_scales: int = 50,
        sigma_min: float = 1e-5,
        teacher_cfg_scale: float = 1.0,
        huber_c: Optional[float] = None,
        null_label: int = 0,
    ):
        super().__init__()
        self.teacher = teacher_model.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.student = student_model
        self.ema_student = copy.deepcopy(student_model).eval()
        for param in self.ema_student.parameters():
            param.requires_grad_(False)
        self.num_scales = int(num_scales)
        self.sigma_min = sigma_min
        self.teacher_cfg_scale = teacher_cfg_scale
        self.huber_c = huber_c
        self.null_label = null_label

    def train(self, mode: bool = True) -> "ConsistencyFlowDistiller":
        super().train(mode)
        self.teacher.eval()
        self.ema_student.eval()
        return self

    @torch.no_grad()
    def update_ema(self, decay: float = 0.999) -> None:
        for ema_param, param in zip(self.ema_student.parameters(), self.student.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_student.buffers(), self.student.buffers()):
            ema_buffer.copy_(buffer)

    @staticmethod
    def predict_x1(model, x_t, t, stem_labels, parent_latent, valid_mask=None) -> torch.Tensor:
        return predict_x1(x_t, t, model(x_t, t, stem_labels, parent_latent, valid_mask).float())

    def forward(
        self,
        x1: torch.Tensor,
        parent_latent: torch.Tensor,
        stem_labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = x1.shape[0]
        x0 = torch.randn_like(x1)
        index = torch.randint(0, self.num_scales, (batch,), device=x1.device)
        t_now = index.float() / self.num_scales
        t_next = (index + 1).float() / self.num_scales

        t_b = t_now.reshape(-1, 1, 1, 1).to(x1.dtype)
        x_now = (1.0 - (1.0 - self.sigma_min) * t_b) * x0 + t_b * x1

        with torch.no_grad():
            v_teacher = ODESolver.velocity(
                self.teacher, x_now, t_now, stem_labels, parent_latent, valid_mask,
                self.teacher_cfg_scale, self.null_label,
            )
            x_next = x_now.float() + (t_next - t_now).reshape(-1, 1, 1, 1) * v_teacher
            target = self.predict_x1(self.ema_student, x_next, t_next, stem_labels, parent_latent, valid_mask)

        prediction = self.predict_x1(self.student, x_now, t_now, stem_labels, parent_latent, valid_mask)
        return masked_slot_mean(pseudo_huber(prediction - target, self.huber_c), valid_mask)
