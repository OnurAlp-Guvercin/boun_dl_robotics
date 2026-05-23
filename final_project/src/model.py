"""
Navigation policy: predicts bounded EE deltas from (fixed bbox, ee_pos).

Input  : (7,)         – [bbox_cx, bbox_cy, bbox_w, bbox_h,  ← 4, normalised [0,1]
                          ee_x_norm, ee_y_norm, ee_z_norm]  ← 3, normalised [0,1]
Output : (3*HORIZON,) – normalised EE deltas in [-1, 1], HORIZON steps ahead
"""
from __future__ import annotations

import torch
import torch.nn as nn


INPUT_DIM  = 7
ACTION_SCALE = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32)


class ResidualBlock(nn.Module):
    """LayerNorm + SiLU residual block for stable MLP policy learning."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class NavigationMLP(nn.Module):
    """
    Residual MLP for closed-loop delta-action prediction with configurable horizon.

    Parameters
    ----------
    horizon     : number of steps to predict ahead (1-5)
    width       : residual trunk width
    n_blocks    : number of residual blocks
    dropout     : dropout probability inside residual blocks
    """

    def __init__(
        self,
        horizon: int = 1,
        width: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        output_dim = horizon * 3

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, width),
            nn.SiLU(),
            *[ResidualBlock(width, dropout) for _ in range(n_blocks)],
            nn.LayerNorm(width),
            nn.Linear(width, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 7) float32

        Returns
        -------
        (B, 3*HORIZON) float32, normalised delta values in [-1,1]
        """
        return self.net(x)

    def predict_delta_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalised delta actions (B, 3*HORIZON), clipped by tanh to [-1, 1]."""
        return self.forward(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

