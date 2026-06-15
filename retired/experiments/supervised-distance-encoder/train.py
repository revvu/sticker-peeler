from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from featurize import FACE_ORDER, state_key_to_indices
from models.distance_model import DistanceModel

ROOT = Path(__file__).resolve().parent
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


class StateDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "state": torch.tensor(state_key_to_indices(example["stateKey"]), dtype=torch.long),
            "distance": torch.tensor(example["distance"], dtype=torch.float32),
            "loss_weight": torch.tensor(example["lossWeight"], dtype=torch.float32),
            "label_type": example["labelType"],
        }


class OrdinalDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "earlier": torch.tensor(state_key_to_indices(example["earlierKey"]), dtype=torch.long),
            "later": torch.tensor(state_key_to_indices(example["laterKey"]), dtype=torch.long),
            "gap": torch.tensor(example["gap"], dtype=torch.float32),
            "loss_weight": torch.tensor(example["lossWeight"], dtype=torch.float32),
            "label_type": example["labelType"],
        }


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
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


def weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    raw = F.huber_loss(predictions, targets, delta=delta, reduction="none")
    normalized = weights / weights.sum().clamp_min(1e-6)
    return (raw * normalized).sum()


def ordinal_loss(
    earlier_pred: torch.Tensor,
    later_pred: torch.Tensor,
    gap: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    raw = F.relu(margin * gap - (later_pred - earlier_pred))
    normalized = weights / weights.sum().clamp_min(1e-6)
    return (raw * normalized).sum()


def save_checkpoint(
    model: DistanceModel,
    config: dict,
    epoch: int,
    val_metrics: dict[str, float],
    checkpoint_dir: Path,
    filename: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        checkpoint_dir / filename,
    )


def is_metric_improved(current: float, best: float, mode: str) -> bool:
    if mode == "max":
        return current > best
    return current < best


@torch.no_grad()
def predict_distances(model: DistanceModel, states: torch.Tensor, device: torch.device) -> torch.Tensor:
    return model(states.to(device))


@torch.no_grad()
def evaluate(
    model: DistanceModel,
    state_loader: DataLoader,
    ordinal_loader: DataLoader,
    extrapolation_examples: list[dict],
    device: torch.device,
    huber_delta: float,
    ordinal_margin: float,
) -> dict[str, float]:
    model.eval()

    bfs_abs_errors = []
    mid_labels = []
    mid_preds = []
    long_labels = []
    long_preds = []
    greedy_correct = 0
    greedy_total = 0

    for batch in state_loader:
        predictions = predict_distances(model, batch["state"], device)
        labels = batch["distance"].to(device)
        abs_errors = (predictions - labels).abs()

        for index, label_type in enumerate(batch["label_type"]):
            if label_type == "bfs":
                bfs_abs_errors.append(abs_errors[index].item())
            elif label_type == "scramble_mid":
                mid_labels.append(labels[index].item())
                mid_preds.append(predictions[index].item())
            elif label_type == "scramble_long":
                long_labels.append(labels[index].item())
                long_preds.append(predictions[index].item())

    for batch in ordinal_loader:
        earlier_pred = predict_distances(model, batch["earlier"], device)
        later_pred = predict_distances(model, batch["later"], device)
        greedy_total += earlier_pred.shape[0]
        greedy_correct += int((later_pred < earlier_pred).sum().item())

    extrapolation_labels = []
    extrapolation_preds = []
    if extrapolation_examples:
        for example in extrapolation_examples:
            state = torch.tensor([state_key_to_indices(example["stateKey"])], dtype=torch.long)
            prediction = predict_distances(model, state, device).item()
            extrapolation_labels.append(example["distance"])
            extrapolation_preds.append(prediction)

    solved_state = torch.tensor([state_key_to_indices(SOLVED_STATE_KEY)], dtype=torch.long)
    pred_at_solved = predict_distances(model, solved_state, device).item()

    def spearman(labels: list[float], preds: list[float]) -> float:
        if len(labels) < 2 or len(set(labels)) < 2:
            return 0.0
        correlation = spearmanr(labels, preds)
        return float(correlation.correlation or 0.0)

    return {
        "mae_bfs": sum(bfs_abs_errors) / max(len(bfs_abs_errors), 1),
        "spearman_6_15": spearman(mid_labels, mid_preds),
        "spearman_16_40": spearman(long_labels, long_preds),
        "spearman_extrapolation": spearman(extrapolation_labels, extrapolation_preds),
        "greedy_step_acc": greedy_correct / max(greedy_total, 1),
        "pred_at_solved": pred_at_solved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--run-name", default="sdl-v1")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_examples = load_jsonl(ROOT / config["paths"]["states_dataset"])
    ordinal_examples = load_jsonl(ROOT / config["paths"]["ordinal_dataset"])
    extrapolation_examples = load_jsonl(ROOT / config["paths"]["extrapolation_dataset"])

    train_states, val_states = split_examples(
        state_examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"],
    )
    train_ordinals, val_ordinals = split_examples(
        ordinal_examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"] + 1,
    )

    print(
        f"Loaded {len(state_examples)} states "
        f"({len(train_states)} train, {len(val_states)} val), "
        f"{len(ordinal_examples)} ordinal pairs, "
        f"{len(extrapolation_examples)} extrapolation states"
    )

    train_state_loader = DataLoader(
        StateDataset(train_states),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_state_loader = DataLoader(
        StateDataset(val_states),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    train_ordinal_loader = DataLoader(
        OrdinalDataset(train_ordinals),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_ordinal_loader = DataLoader(
        OrdinalDataset(val_ordinals),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = DistanceModel(
        latent_dim=config["model"]["latent_dim"],
        sticker_embed_dim=config["model"]["sticker_embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = None
    if config["training"].get("lr_schedule") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    huber_delta = config["training"]["huber_delta"]
    ordinal_weight = config["training"]["ordinal_weight"]
    ordinal_margin = config["training"]["ordinal_margin"]
    grad_clip_norm = config["training"].get("grad_clip_norm")

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/supervised-distance") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")

    early_stop_metric = config["training"].get("early_stop_metric", "mae_bfs")
    early_stop_mode = config["training"].get("early_stop_mode", "min")
    best_mae_bfs = float("inf")
    best_greedy_step = float("-inf")
    best_early_stop_value = float("inf") if early_stop_mode == "min" else float("-inf")
    patience = 0
    global_step = 0
    ordinal_iter = iter(train_ordinal_loader)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_ord_loss = 0.0
        progress = tqdm(train_state_loader, desc=f"epoch {epoch + 1}")

        for batch in progress:
            predictions = model(batch["state"].to(device))
            reg_component = weighted_huber_loss(
                predictions,
                batch["distance"].to(device),
                batch["loss_weight"].to(device),
                huber_delta,
            )

            try:
                ordinal_batch = next(ordinal_iter)
            except StopIteration:
                ordinal_iter = iter(train_ordinal_loader)
                ordinal_batch = next(ordinal_iter)

            earlier_pred = model(ordinal_batch["earlier"].to(device))
            later_pred = model(ordinal_batch["later"].to(device))
            ord_component = ordinal_loss(
                earlier_pred,
                later_pred,
                ordinal_batch["gap"].to(device),
                ordinal_batch["loss_weight"].to(device),
                ordinal_margin,
            )

            loss = reg_component + ordinal_weight * ord_component

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_loss += loss.item()
            train_reg_loss += reg_component.item()
            train_ord_loss += ord_component.item()
            global_step += 1

            if writer and global_step % tensorboard_config.get("log_every_steps", 10) == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/regression", reg_component.item(), global_step)
                writer.add_scalar("loss/ordinal", ord_component.item(), global_step)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                reg=f"{reg_component.item():.4f}",
            )

        if scheduler:
            scheduler.step()

        train_metrics = {
            "loss": train_loss / len(train_state_loader),
            "regression_loss": train_reg_loss / len(train_state_loader),
            "ordinal_loss": train_ord_loss / len(train_state_loader),
        }
        val_metrics = evaluate(
            model,
            val_state_loader,
            val_ordinal_loader,
            extrapolation_examples,
            device,
            huber_delta,
            ordinal_margin,
        )

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_mae_bfs={val_metrics['mae_bfs']:.4f} "
            f"val_spearman_6_15={val_metrics['spearman_6_15']:.4f} "
            f"val_spearman_16_40={val_metrics['spearman_16_40']:.4f} "
            f"val_spearman_extrapolation={val_metrics['spearman_extrapolation']:.4f} "
            f"val_greedy_step_acc={val_metrics['greedy_step_acc']:.4f} "
            f"val_pred_at_solved={val_metrics['pred_at_solved']:.4f}"
        )

        if writer:
            writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"metrics/val_{key}", value, epoch)
            if scheduler:
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_metrics["mae_bfs"] < best_mae_bfs:
            best_mae_bfs = val_metrics["mae_bfs"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_mae_bfs.pt")

        if val_metrics["greedy_step_acc"] > best_greedy_step:
            best_greedy_step = val_metrics["greedy_step_acc"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_greedy_step.pt")

        current_early_stop_value = val_metrics.get(early_stop_metric)
        if current_early_stop_value is None:
            raise ValueError(f"Unknown early stop metric: {early_stop_metric}")

        if is_metric_improved(current_early_stop_value, best_early_stop_value, early_stop_mode):
            best_early_stop_value = current_early_stop_value
            patience = 0
        else:
            patience += 1
            if patience >= config["training"]["early_stop_patience"]:
                print("Early stopping triggered.")
                break

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
