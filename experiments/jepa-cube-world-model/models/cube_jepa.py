from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import EMPTY_MOVE_TOKEN, MAX_MOVE_LENGTH, NUM_FACE_TYPES, NUM_MOVE_TYPES, STICKER_COUNT


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


class MoveSequenceEncoder(nn.Module):
    def __init__(self, move_embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.move_embed = nn.Embedding(NUM_MOVE_TYPES + 1, move_embed_dim)
        self.null_move = nn.Parameter(torch.zeros(move_embed_dim))
        self.gru = nn.GRU(
            input_size=move_embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, move_indices: torch.Tensor, move_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.move_embed(move_indices)
        empty_rows = move_lengths == 0
        if empty_rows.any():
            embedded[empty_rows] = self.null_move

        packed_lengths = move_lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            packed_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.proj(hidden[-1])


class CubePredictor(nn.Module):
    def __init__(self, latent_dim: int, move_context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + move_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state_embedding: torch.Tensor, move_context: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([state_embedding, move_context], dim=-1))


class CubeJEPA(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        sticker_embed_dim: int = 16,
        move_embed_dim: int = 32,
        hidden_dim: int = 256,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum

        self.context_encoder = CubeEncoder(latent_dim, sticker_embed_dim, hidden_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

        self.move_encoder = MoveSequenceEncoder(move_embed_dim, hidden_dim, move_embed_dim)
        self.predictor = CubePredictor(latent_dim, move_embed_dim, hidden_dim)

    def encode_context(self, start_states: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(start_states)

    def encode_target(self, end_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_encoder(end_states)

    def predict(self, start_states: torch.Tensor, move_indices: torch.Tensor, move_lengths: torch.Tensor) -> torch.Tensor:
        state_embedding = self.encode_context(start_states)
        move_context = self.move_encoder(move_indices, move_lengths)
        return self.predictor(state_embedding, move_context)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online_parameter, target_parameter in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target_parameter.data.mul_(self.ema_momentum).add_(
                online_parameter.data,
                alpha=1 - self.ema_momentum,
            )


def vicreg_loss(
    embeddings: torch.Tensor,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, feature_dim = embeddings.shape
    if batch_size <= 1:
        zero = embeddings.new_tensor(0.0)
        return zero, {"variance": 0.0, "covariance": 0.0}

    centered = embeddings - embeddings.mean(dim=0)
    variance = F.relu(1.0 - centered.std(dim=0)).mean()
    covariance = (centered.T @ centered) / (batch_size - 1)
    off_diagonal = covariance.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:].flatten()
    covariance_penalty = off_diagonal.pow(2).mean()

    total = variance_weight * variance + covariance_weight * covariance_penalty
    return total, {
        "variance": float(variance.detach().cpu()),
        "covariance": float(covariance_penalty.detach().cpu()),
    }


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    vicreg_weight: float = 0.1,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(predicted, target)
    vicreg, vicreg_parts = vicreg_loss(
        torch.cat([predicted, target], dim=0),
        variance_weight=variance_weight,
        covariance_weight=covariance_weight,
    )
    total = l1 + vicreg_weight * vicreg
    return total, {
        "l1": float(l1.detach().cpu()),
        "vicreg": float(vicreg.detach().cpu()),
        "vicreg_variance": vicreg_parts["variance"],
        "vicreg_covariance": vicreg_parts["covariance"],
        "total": float(total.detach().cpu()),
    }


def pad_move_sequences(move_sequences: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in move_sequences], dtype=torch.long, device=device)
    padded = torch.full((len(move_sequences), MAX_MOVE_LENGTH), EMPTY_MOVE_TOKEN, dtype=torch.long, device=device)

    for row_index, sequence in enumerate(move_sequences):
        if sequence:
            padded[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)

    return padded, lengths
