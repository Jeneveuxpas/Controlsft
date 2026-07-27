import torch
from torch import Tensor, nn


class _ResidualLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU(approximate="tanh")
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.proj(self.act(self.norm(hidden_states)))
        return hidden_states + self.residual_scale * residual


class CleanRGBSRAHead(nn.Module):
    """Five-linear-layer per-token projector from transformer hidden states to VAE latents."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU(approximate="tanh")
        self.blocks = nn.ModuleList([_ResidualLayer(hidden_dim) for _ in range(3)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.input_act(self.input_proj(self.input_norm(hidden_states)))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_proj(self.output_norm(hidden_states))
