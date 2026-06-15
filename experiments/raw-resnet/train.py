from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from featurize import state_key_to_one_hot_image
from models.raw_resnet import RawResNet

ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 27
CLASS_WEIGHTS = torch.arange(NUM_CLASSES, dtype=torch.float32)


class DistanceDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "image": state_key_to_one_hot_image(example["stateKey"]),
            "distance": torch.tensor(example["distance"], dtype=torch.long),
        }


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_checkpoint(
    model: RawResNet,
    config: dict,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_dir: Path,
    filename: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_dir / filename,
    )


def is_metric_improved(current: float, best: float, mode: str) -> bool:
    if mode == "max":
        return current > best
    return current < best


def compute_batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = F.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    expected_distance = (probabilities * CLASS_WEIGHTS.to(probabilities.device)).sum(dim=-1)
    abs_errors = (expected_distance - labels.float()).abs()

    return {
        "loss": F.cross_entropy(logits, labels).item(),
        "accuracy": (predictions == labels).float().mean().item(),
        "mae": abs_errors.mean().item(),
        "top3_accuracy": (
            torch.topk(probabilities, k=3, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
        ),
    }


@torch.no_grad()
def evaluate(model: RawResNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()

    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mae": 0.0,
        "top3_accuracy": 0.0,
    }
    per_distance_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    per_distance_total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["distance"].to(device)
        logits = model(images)
        batch_metrics = compute_batch_metrics(logits, labels)

        for key in totals:
            totals[key] += batch_metrics[key]
        batches += 1

        predictions = logits.argmax(dim=-1)
        for distance in range(NUM_CLASSES):
            mask = labels == distance
            per_distance_total[distance] += mask.sum().item()
            per_distance_correct[distance] += (predictions[mask] == distance).sum().item()

    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    metrics["per_distance_accuracy"] = (
        per_distance_correct.float() / per_distance_total.clamp_min(1).float()
    ).tolist()
    return metrics


def one_hot_batch_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """Map 6-channel one-hot cube images to RGB for TensorBoard preview."""
    color_map = torch.tensor(
        [
            [0.85, 0.16, 0.18],  # F red
            [1.00, 0.54, 0.00],  # B orange
            [0.97, 0.98, 0.99],  # U white
            [1.00, 0.82, 0.00],  # D yellow
            [0.12, 0.75, 0.36],  # R green
            [0.15, 0.39, 0.92],  # L blue
        ],
        dtype=images.dtype,
        device=images.device,
    )
    indices = images.argmax(dim=1)
    return color_map[indices].permute(0, 3, 1, 2)


def log_image_preview(writer: SummaryWriter, examples: list[dict], global_step: int) -> None:
    if not examples:
        return

    preview = torch.stack([state_key_to_one_hot_image(example["stateKey"]) for example in examples[:8]])
    writer.add_images("samples/cube_faces_rgb", one_hot_batch_to_rgb(preview), global_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--run-name", default="raw-resnet-v1")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_jsonl(ROOT / config["paths"]["train_dataset"])
    val_examples = load_jsonl(ROOT / config["paths"]["val_dataset"])
    test_examples = load_jsonl(ROOT / config["paths"]["test_dataset"])

    print(
        f"Loaded {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples on {device}"
    )

    train_loader = DataLoader(
        DistanceDataset(train_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        DistanceDataset(val_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        DistanceDataset(test_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = RawResNet(
        num_colors=config["model"]["num_colors"],
        num_classes=config["model"]["num_classes"],
        channels=config["model"]["channels"],
        num_blocks=config["model"]["num_blocks"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = None
    if config["training"].get("lr_schedule") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/raw-resnet") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")
        log_image_preview(writer, train_examples, global_step=0)

    early_stop_metric = config["training"].get("early_stop_metric", "mae")
    early_stop_mode = config["training"].get("early_stop_mode", "min")
    early_stop_patience = config["training"].get("early_stop_patience", 15)
    grad_clip_norm = config["training"].get("grad_clip_norm")
    log_every_steps = tensorboard_config.get("log_every_steps", 20)

    best_val_accuracy = float("-inf")
    best_val_mae = float("inf")
    best_early_stop_value = float("inf") if early_stop_mode == "min" else float("-inf")
    patience = 0
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_totals = {"loss": 0.0, "accuracy": 0.0, "mae": 0.0, "top3_accuracy": 0.0}
        train_batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")

        for batch in progress:
            images = batch["image"].to(device)
            labels = batch["distance"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            batch_metrics = compute_batch_metrics(logits.detach(), labels)
            train_batches += 1
            for key in train_totals:
                train_totals[key] += batch_metrics[key]

            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                acc=f"{batch_metrics['accuracy']:.3f}",
                mae=f"{batch_metrics['mae']:.2f}",
            )

            if writer and global_step % log_every_steps == 0:
                writer.add_scalar("train/loss", batch_metrics["loss"], global_step)
                writer.add_scalar("train/accuracy", batch_metrics["accuracy"], global_step)
                writer.add_scalar("train/mae", batch_metrics["mae"], global_step)
                writer.add_scalar("train/top3_accuracy", batch_metrics["top3_accuracy"], global_step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        if scheduler:
            scheduler.step()

        train_metrics = {key: value / max(train_batches, 1) for key, value in train_totals.items()}
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_mae={val_metrics['mae']:.3f}"
        )

        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f"epoch/train_{key}", value, epoch + 1)
            for key, value in val_metrics.items():
                if key == "per_distance_accuracy":
                    continue
                writer.add_scalar(f"epoch/val_{key}", value, epoch + 1)

            for distance, accuracy in enumerate(val_metrics["per_distance_accuracy"]):
                writer.add_scalar(f"metrics/val_distance_{distance}_accuracy", accuracy, epoch + 1)

            writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_accuracy.pt")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_mae.pt")

        monitor_value = val_metrics["mae" if early_stop_metric == "val_mae" else "accuracy"]
        if is_metric_improved(monitor_value, best_early_stop_value, early_stop_mode):
            best_early_stop_value = monitor_value
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1} on {early_stop_metric}.")
                break

    best_checkpoint = checkpoint_dir / "best_mae.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test: loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"mae={test_metrics['mae']:.3f} "
        f"top3_acc={test_metrics['top3_accuracy']:.4f}"
    )

    if writer:
        for key, value in test_metrics.items():
            if key == "per_distance_accuracy":
                continue
            writer.add_scalar(f"test/{key}", value, 0)
        writer.close()


if __name__ == "__main__":
    main()
