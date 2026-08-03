import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


def extract_clean_rgb_sra_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Extract an SRA head from a full model state dict and remove its model prefix."""
    marker = "clean_rgb_sra_head."
    extracted = {key.split(marker, 1)[1]: value for key, value in state_dict.items() if marker in key}
    if not extracted:
        raise ValueError("Full model state dict contains no Clean RGB SRA head")
    return extracted


class _ResidualLayer(nn.Module):
    def __init__(self, dim: int, residual_scale: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU(approximate="tanh")
        # Depth-aware initialization stabilizes deeper heads while allowing
        # each residual block to learn its own contribution.
        # FSDP v1 cannot manage scalar parameters. A single-element vector has
        # identical broadcasting semantics while remaining shardable.
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale]))

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.proj(self.act(self.norm(hidden_states)))
        return hidden_states + self.residual_scale * residual


class CleanRGBSRAHead(nn.Module):
    """Configurable-depth per-token projector from transformer hidden states to VAE latents."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 128, num_layers: int = 5) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"Clean RGB SRA projector requires at least 2 linear layers, got {num_layers}")
        num_residual_blocks = num_layers - 2
        residual_scale = 1.0 / math.sqrt(num_residual_blocks) if num_residual_blocks > 0 else 1.0

        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU(approximate="tanh")
        self.blocks = nn.ModuleList(
            [_ResidualLayer(hidden_dim, residual_scale=residual_scale) for _ in range(num_residual_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.input_act(self.input_proj(self.input_norm(hidden_states)))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_proj(self.output_norm(hidden_states))
