from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from featurize import FACE_ORDER, state_key_to_face_grid
from models.cnn_encoder import FACE_NEIGHBORS, NUM_FACES
from models.order_model import OrderModel

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "best_greedy_step.pt",
    ROOT / "checkpoints" / "best_order_acc.pt",
    ROOT / "checkpoints" / "best_mae_bfs.pt",
]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "order-embedding" / "encoder_weights.json"
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def export_conv_layer(layer: nn.Conv2d) -> dict:
    return {
        "weight": tensor_to_list(layer.weight),
        "bias": tensor_to_list(layer.bias),
        "in_channels": layer.in_channels,
        "out_channels": layer.out_channels,
        "kernel_size": layer.kernel_size[0],
        "stride": layer.stride[0],
        "padding": layer.padding[0],
    }


def export_linear_layer(layer: nn.Linear) -> dict:
    return {
        "weight": tensor_to_list(layer.weight),
        "bias": tensor_to_list(layer.bias),
    }


def export_encoder(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]

    model = OrderModel(
        latent_dim=config["latent_dim"],
        sticker_embed_dim=config["sticker_embed_dim"],
        conv_channels=config["conv_channels"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    encoder = model.encoder
    conv_layers = []
    for layer in encoder.face_conv:
        if isinstance(layer, nn.Conv2d):
            conv_layers.append(export_conv_layer(layer))

    fusion_layers = []
    for layer in encoder.fusion_mlp:
        if isinstance(layer, nn.Linear):
            fusion_layers.append(export_linear_layer(layer))

    output_layers = []
    for layer in encoder.output_mlp:
        if isinstance(layer, nn.Linear):
            output_layers.append(export_linear_layer(layer))

    solved_grid = torch.tensor([state_key_to_face_grid(SOLVED_STATE_KEY)], dtype=torch.long)
    with torch.no_grad():
        solved_embedding = model.encode(solved_grid)
        distance_at_solved = model.distance_to_solved(solved_embedding, solved_embedding).item()

    payload = {
        "version": 1,
        "model_type": "order_embedding_cnn",
        "face_order": FACE_ORDER,
        "face_neighbors": {str(index): neighbors for index, neighbors in enumerate(FACE_NEIGHBORS)},
        "num_faces": NUM_FACES,
        "latent_dim": config["latent_dim"],
        "sticker_embed_dim": config["sticker_embed_dim"],
        "conv_channels": config["conv_channels"],
        "hidden_dim": config["hidden_dim"],
        "sticker_embed": tensor_to_list(encoder.sticker_embed.weight),
        "face_conv": conv_layers,
        "fusion_mlp": fusion_layers,
        "output_mlp": output_layers,
        "solved_embedding": tensor_to_list(solved_embedding.squeeze(0)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_metrics": checkpoint.get("val_metrics"),
        "distance_at_solved": distance_at_solved,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(f"Exported order-embedding CNN from {checkpoint_path} to {output_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Distance at solved: {distance_at_solved:.4f}")


def resolve_checkpoint_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No checkpoint found. Train the model first or pass --checkpoint.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    export_encoder(checkpoint_path, args.output)


if __name__ == "__main__":
    main()
