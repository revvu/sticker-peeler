from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import FACE_ORDER, state_key_to_indices
from models.cube_encoder import CubeEncoder

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "best_solved_cosine.pt",
    ROOT / "checkpoints" / "best_triplet_acc.pt",
]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "triplet" / "encoder_weights.json"
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def export_encoder(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]

    model = CubeEncoder(
        latent_dim=config["latent_dim"],
        sticker_embed_dim=config["sticker_embed_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mlp_layers = []
    for layer in model.mlp:
        if isinstance(layer, nn.Linear):
            mlp_layers.append(
                {
                    "weight": tensor_to_list(layer.weight),
                    "bias": tensor_to_list(layer.bias),
                }
            )

    payload = {
        "version": 1,
        "face_order": FACE_ORDER,
        "latent_dim": config["latent_dim"],
        "sticker_embed_dim": config["sticker_embed_dim"],
        "hidden_dim": config["hidden_dim"],
        "sticker_embed": tensor_to_list(model.sticker_embed.weight),
        "mlp": mlp_layers,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_metrics": checkpoint.get("val_metrics"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    solved_indices = torch.tensor([state_key_to_indices(SOLVED_STATE_KEY)], dtype=torch.long)
    with torch.no_grad():
        embedding = model(solved_indices)
        norm = float(embedding.norm().item())
        self_similarity = float(F.cosine_similarity(embedding, embedding).item())

    print(f"Exported triplet encoder from {checkpoint_path} to {output_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Solved embedding norm: {norm:.4f}")
    print(f"Solved self-similarity: {self_similarity:.4f}")


def resolve_checkpoint_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_encoder(resolve_checkpoint_path(args.checkpoint), args.output)


if __name__ == "__main__":
    main()
