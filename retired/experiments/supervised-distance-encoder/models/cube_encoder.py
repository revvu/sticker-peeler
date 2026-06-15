from __future__ import annotations

import torch
import torch.nn as nn

from featurize import NUM_FACE_TYPES, STICKER_COUNT


class CubeEncoder(nn.Module):
    def __init__(self, latent_dim: int, sticker_embed_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.sticker_embed = nn.Embedding(NUM_FACE_TYPES, sticker_embed_dim)
        input_dim = STICKER_COUNT * sticker_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        embedded = self.sticker_embed(state_indices)
        flattened = embedded.reshape(state_indices.shape[0], -1)
        return self.mlp(flattened)
