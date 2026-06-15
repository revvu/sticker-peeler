from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cube_encoder import CubeEncoder


class DistanceModel(nn.Module):
    def __init__(self, latent_dim: int, sticker_embed_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.encoder = CubeEncoder(
            latent_dim=latent_dim,
            sticker_embed_dim=sticker_embed_dim,
            hidden_dim=hidden_dim,
        )
        self.head = nn.Linear(latent_dim, 1)

    def encode(self, state_indices: torch.Tensor) -> torch.Tensor:
        return self.encoder(state_indices)

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        latent = self.encode(state_indices)
        return F.softplus(self.head(latent).squeeze(-1))
