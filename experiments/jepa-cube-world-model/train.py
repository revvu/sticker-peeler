from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from featurize import moves_to_indices, state_key_to_indices
from models.cube_jepa import CubeJEPA, jepa_loss, pad_move_sequences

ROOT = Path(__file__).resolve().parent


class CubeTransitionDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "start": torch.tensor(state_key_to_indices(example["startKey"]), dtype=torch.long),
            "end": torch.tensor(state_key_to_indices(example["endKey"]), dtype=torch.long),
            "moves": moves_to_indices(example["moves"]),
            "end_key": example["endKey"],
        }


def collate_batch(batch: list[dict], device: torch.device) -> dict:
    move_sequences = [item["moves"] for item in batch]
    move_indices, move_lengths = pad_move_sequences(move_sequences, device)

    return {
        "start": torch.stack([item["start"] for item in batch]).to(device),
        "end": torch.stack([item["end"] for item in batch]).to(device),
        "move_indices": move_indices,
        "move_lengths": move_lengths,
        "end_keys": [item["end_key"] for item in batch],
    }


def load_examples(dataset_path: Path) -> list[dict]:
    examples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_examples(examples: list[dict], train_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * train_split)
    return shuffled[:split_index], shuffled[split_index:]


@torch.no_grad()
def evaluate(
    model: CubeJEPA,
    loader: DataLoader,
    device: torch.device,
    vicreg_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "l1": 0.0, "vicreg": 0.0, "cosine": 0.0}
    batches = 0
    nn_hits = 0
    nn_total = 0
    bank_embeddings: list[torch.Tensor] = []
    bank_keys: list[str] = []

    for batch in loader:
        predicted = model.predict(batch["start"], batch["move_indices"], batch["move_lengths"])
        target = model.encode_target(batch["end"])
        _, parts = jepa_loss(predicted, target, vicreg_weight=vicreg_weight)

        totals["total"] += parts["total"]
        totals["l1"] += parts["l1"]
        totals["vicreg"] += parts["vicreg"]
        totals["cosine"] += torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean().item()
        batches += 1

        bank_embeddings.append(target)
        bank_keys.extend(batch["end_keys"])

    if batches == 0:
        return totals

    bank = torch.cat(bank_embeddings, dim=0)
    bank = torch.nn.functional.normalize(bank, dim=-1)

    for batch in loader:
        predicted = model.predict(batch["start"], batch["move_indices"], batch["move_lengths"])
        predicted = torch.nn.functional.normalize(predicted, dim=-1)

        for row_index, end_key in enumerate(batch["end_keys"]):
            similarity = predicted[row_index] @ bank.T
            nearest_index = int(similarity.argmax().item())
            if bank_keys[nearest_index] == end_key:
                nn_hits += 1
            nn_total += 1

    return {
        "total": totals["total"] / batches,
        "l1": totals["l1"] / batches,
        "vicreg": totals["vicreg"] / batches,
        "cosine": totals["cosine"] / batches,
        "nn_accuracy": nn_hits / max(nn_total, 1),
    }


def maybe_init_wandb(config: dict, run_name: str):
    wandb_config = config.get("logging", {}).get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    import wandb

    wandb.init(project=wandb_config.get("project", "sticker-peeler-jepa"), name=run_name)
    return wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--run-name", default="cube-10k-v1")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset_path = ROOT / config["paths"]["dataset"]
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(dataset_path)
    train_examples, val_examples = split_examples(
        examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"],
    )

    train_loader = DataLoader(
        CubeTransitionDataset(train_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_batch(batch, device),
    )
    val_loader = DataLoader(
        CubeTransitionDataset(val_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, device),
    )

    model = CubeJEPA(
        latent_dim=config["model"]["latent_dim"],
        sticker_embed_dim=config["model"]["sticker_embed_dim"],
        move_embed_dim=config["model"]["move_embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        ema_momentum=config["model"]["ema_momentum"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/cube-jepa") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")

    wandb = maybe_init_wandb(config, args.run_name)

    best_val_loss = float("inf")
    patience = 0
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_totals = {"total": 0.0, "l1": 0.0, "vicreg": 0.0}
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")

        for batch in progress:
            predicted = model.predict(batch["start"], batch["move_indices"], batch["move_lengths"])
            target = model.encode_target(batch["end"])
            loss, parts = jepa_loss(predicted, target, vicreg_weight=config["training"]["vicreg_weight"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

            train_totals["total"] += parts["total"]
            train_totals["l1"] += parts["l1"]
            train_totals["vicreg"] += parts["vicreg"]
            global_step += 1

            if writer and global_step % tensorboard_config.get("log_every_steps", 10) == 0:
                writer.add_scalar("loss/total", parts["total"], global_step)
                writer.add_scalar("loss/l1", parts["l1"], global_step)
                writer.add_scalar("loss/vicreg", parts["vicreg"], global_step)
                writer.add_scalar("loss/vicreg_variance", parts["vicreg_variance"], global_step)

            progress.set_postfix(
                total=f"{parts['total']:.4f}",
                l1=f"{parts['l1']:.4f}",
            )

        train_metrics = {key: value / len(train_loader) for key, value in train_totals.items()}
        val_metrics = evaluate(model, val_loader, device, config["training"]["vicreg_weight"])

        print(
            f"epoch {epoch + 1}: "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_cosine={val_metrics['cosine']:.4f} "
            f"val_nn_accuracy={val_metrics['nn_accuracy']:.4f}"
        )

        if writer:
            writer.add_scalar("epoch/train_total", train_metrics["total"], epoch)
            writer.add_scalar("epoch/val_total", val_metrics["total"], epoch)
            writer.add_scalar("metrics/val_cosine", val_metrics["cosine"], epoch)
            writer.add_scalar("metrics/val_nn_accuracy", val_metrics["nn_accuracy"], epoch)

        if wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                }
            )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= config["training"]["early_stop_patience"]:
                print("Early stopping triggered.")
                break

    if writer:
        writer.close()
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
