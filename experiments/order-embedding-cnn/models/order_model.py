from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_encoder import FaceGridCNNEncoder


class OrderModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        sticker_embed_dim: int = 16,
        conv_channels: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = FaceGridCNNEncoder(
            latent_dim=latent_dim,
            sticker_embed_dim=sticker_embed_dim,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )

    def encode(self, face_grid: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.encoder(face_grid))

    def order_violation(
        self,
        closer_emb: torch.Tensor,
        farther_emb: torch.Tensor,
    ) -> torch.Tensor:
        return F.relu(farther_emb - closer_emb).pow(2).sum(dim=-1)

    def distance_to_solved(
        self,
        state_emb: torch.Tensor,
        solved_emb: torch.Tensor,
    ) -> torch.Tensor:
        return F.relu(state_emb - solved_emb).sum(dim=-1)

    def forward_embeddings(self, face_grid: torch.Tensor) -> torch.Tensor:
        return self.encode(face_grid)

    def forward_distance(
        self,
        face_grid: torch.Tensor,
        solved_emb: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = self.encode(face_grid)
        return self.distance_to_solved(state_emb, solved_emb)
