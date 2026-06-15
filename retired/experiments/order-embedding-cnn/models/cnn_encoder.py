from __future__ import annotations

import torch
import torch.nn as nn

from featurize import NUM_FACE_TYPES

NUM_FACES = 6
FACE_NEIGHBORS = [
    [2, 4, 3, 5],  # F: U, R, D, L
    [2, 4, 3, 5],  # B
    [0, 4, 1, 5],  # U
    [0, 4, 1, 5],  # D
    [2, 1, 3, 0],  # R
    [2, 1, 3, 0],  # L
]


class FaceGridCNNEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        sticker_embed_dim: int = 16,
        conv_channels: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sticker_embed = nn.Embedding(NUM_FACE_TYPES, sticker_embed_dim)
        self.face_conv = nn.Sequential(
            nn.Conv2d(sticker_embed_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.face_feat_dim = conv_channels * 9
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.face_feat_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * NUM_FACES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.register_buffer(
            "neighbor_indices",
            torch.tensor(FACE_NEIGHBORS, dtype=torch.long),
            persistent=False,
        )

    def _encode_faces(self, face_grid: torch.Tensor) -> torch.Tensor:
        batch_size, num_faces, height, width = face_grid.shape
        embedded = self.sticker_embed(face_grid)
        packed = embedded.permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_faces,
            embedded.shape[-1],
            height,
            width,
        )
        conv_out = self.face_conv(packed)
        return conv_out.reshape(batch_size, num_faces, -1)

    def forward(self, face_grid: torch.Tensor) -> torch.Tensor:
        face_features = self._encode_faces(face_grid)
        neighbor_features = face_features[:, self.neighbor_indices]
        neighbor_mean = neighbor_features.mean(dim=2)
        fusion_input = torch.cat([face_features, neighbor_mean], dim=-1)
        batch_size = face_features.shape[0]
        fused = self.fusion_mlp(fusion_input.reshape(batch_size * NUM_FACES, -1))
        return self.output_mlp(fused.reshape(batch_size, -1))
