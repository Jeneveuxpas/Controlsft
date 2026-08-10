from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RepresentationProjectionHead(nn.Module):
    """Lightweight per-token MLP used on the student representation only."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states)


def compute_representation_distillation_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    loss_mask: Tensor,
    *,
    loss_weight: float,
    warmup_steps: int,
    global_step: int,
    loss_type: Literal["cosine", "l1", "l2"] = "cosine",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute a masked per-sample feature loss against detached teacher features."""
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "Student and teacher hidden states must have identical shapes after target slicing, "
            f"got student={tuple(student_hidden.shape)}, teacher={tuple(teacher_hidden.shape)}"
        )
    if loss_mask.shape != student_hidden.shape[:2]:
        raise ValueError(
            f"Distillation mask shape {tuple(loss_mask.shape)} does not match hidden states "
            f"{tuple(student_hidden.shape[:2])}"
        )

    teacher_hidden = teacher_hidden.detach()
    student_float = student_hidden.float()
    teacher_float = teacher_hidden.float()
    if loss_type == "cosine":
        token_loss = 1.0 - F.cosine_similarity(student_float, teacher_float, dim=-1)
    elif loss_type == "l1":
        token_loss = (student_float - teacher_float).abs().mean(dim=-1)
    elif loss_type == "l2":
        token_loss = (student_float - teacher_float).square().mean(dim=-1)
    else:
        raise ValueError(f"Unknown representation distillation loss type: {loss_type}")

    mask = loss_mask.to(device=token_loss.device, dtype=token_loss.dtype)
    raw_loss = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    warmup_scale = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
    effective_weight = float(loss_weight) * warmup_scale
    weight = torch.tensor(effective_weight, device=student_hidden.device, dtype=torch.float32)
    weighted_loss = raw_loss * weight

    valid = loss_mask.bool()
    student_norm = student_hidden.detach().float().norm(dim=-1)
    teacher_norm = teacher_hidden.float().norm(dim=-1)
    normalizer = valid.sum().clamp(min=1)
    return weighted_loss, {
        "train/representation_distillation_raw": raw_loss.detach().mean(),
        "train/representation_distillation_loss": weighted_loss.detach().mean(),
        "train/representation_distillation_weight": weight.detach(),
        "train/representation_student_norm": (student_norm * valid).sum() / normalizer,
        "train/representation_teacher_norm": (teacher_norm * valid).sum() / normalizer,
    }
