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
from models.cube_encoder import CubeEncoder

ROOT = Path(__file__).resolve().parent
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


class TripletDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "anchor": torch.tensor(state_key_to_indices(example["anchorKey"]), dtype=torch.long),
            "positive": torch.tensor(state_key_to_indices(example["positiveKey"]), dtype=torch.long),
            "negative": torch.tensor(state_key_to_indices(example["negativeKey"]), dtype=torch.long),
            "anchor_key": example["anchorKey"],
            "positive_len": example["positiveLen"],
            "negative_len": example["negativeLen"],
        }


def load_examples(dataset_path: Path) -> list[dict]:
    examples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_examples_by_anchor(examples: list[dict], train_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    anchors = sorted({example["anchorKey"] for example in examples})
    rng = random.Random(seed)
    rng.shuffle(anchors)
    split_index = max(1, int(len(anchors) * train_split))
    if split_index >= len(anchors):
        split_index = len(anchors) - 1

    train_anchors = set(anchors[:split_index])
    train_examples = [example for example in examples if example["anchorKey"] in train_anchors]
    val_examples = [example for example in examples if example["anchorKey"] not in train_anchors]
    return train_examples, val_examples


def encode_normalized(model: CubeEncoder, states: torch.Tensor) -> torch.Tensor:
    return F.normalize(model(states), dim=-1)


def pairwise_squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).pow(2).sum(dim=-1)


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    distance_positive = pairwise_squared_distance(anchor, positive)
    distance_negative = pairwise_squared_distance(anchor, negative)
    return F.relu(distance_positive - distance_negative + margin).mean()


def select_semi_hard_negatives(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    batch_size = anchors.shape[0]
    distance_positive = pairwise_squared_distance(anchors, positives)
    distance_negative = pairwise_squared_distance(anchors, negatives).unsqueeze(1)
    all_negatives = negatives.unsqueeze(0).expand(batch_size, batch_size, -1)
    all_anchor = anchors.unsqueeze(1).expand(batch_size, batch_size, -1)
    batch_distances = pairwise_squared_distance(
        all_anchor.reshape(-1, all_anchor.shape[-1]),
        all_negatives.reshape(-1, all_negatives.shape[-1]),
    ).reshape(batch_size, batch_size)

    selected = []
    for row in range(batch_size):
        positive_distance = distance_positive[row]
        candidate_mask = batch_distances[row] > positive_distance
        semi_hard_mask = (batch_distances[row] > positive_distance) & (batch_distances[row] < positive_distance + margin)
        candidates = torch.where(semi_hard_mask)[0]

        if candidates.numel() > 0:
            pick = candidates[torch.randint(0, candidates.numel(), (1,)).item()]
        else:
            hard_candidates = torch.where(candidate_mask)[0]
            if hard_candidates.numel() > 0:
                pick = hard_candidates[torch.argmin(batch_distances[row, hard_candidates])]
            else:
                pick = row

        selected.append(negatives[pick])

    return torch.stack(selected, dim=0)


def save_checkpoint(
    model: CubeEncoder,
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
def evaluate(
    model: CubeEncoder,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    solved_indices: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0
    solved_cosines = []
    length_proxies = []
    solved_distances = []

    solved_embedding = encode_normalized(model, solved_indices)

    for batch in loader:
        anchor = encode_normalized(model, batch["anchor"].to(device))
        positive = encode_normalized(model, batch["positive"].to(device))
        negative = encode_normalized(model, batch["negative"].to(device))

        distance_positive = pairwise_squared_distance(anchor, positive)
        distance_negative = pairwise_squared_distance(anchor, negative)
        loss = F.relu(distance_positive - distance_negative + margin)
        total_loss += loss.mean().item()
        batches += 1

        correct += int((distance_positive + margin < distance_negative).sum().item())
        total += anchor.shape[0]

        anchor_solved_cosine = F.cosine_similarity(anchor, solved_embedding.expand_as(anchor), dim=-1)
        solved_cosines.extend(anchor_solved_cosine.cpu().tolist())
        solved_distances.extend(pairwise_squared_distance(anchor, solved_embedding.expand_as(anchor)).sqrt().cpu().tolist())
        length_proxies.extend(
            (batch["negative_len"].float() - batch["positive_len"].float()).abs().cpu().tolist()
        )

    length_correlation = 0.0
    if len(length_proxies) > 1 and len(set(length_proxies)) > 1:
        correlation = spearmanr(length_proxies, solved_distances)
        length_correlation = float(correlation.correlation or 0.0)

    return {
        "loss": total_loss / max(batches, 1),
        "triplet_accuracy": correct / max(total, 1),
        "solved_cosine": sum(solved_cosines) / max(len(solved_cosines), 1),
        "length_correlation": length_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--run-name", default="cube-triplet-v1")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset_path = ROOT / config["paths"]["dataset"]
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(dataset_path)
    train_examples, val_examples = split_examples_by_anchor(
        examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"],
    )
    print(f"Loaded {len(examples)} triplets ({len(train_examples)} train, {len(val_examples)} val)")

    train_loader = DataLoader(
        TripletDataset(train_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TripletDataset(val_examples),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = CubeEncoder(
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

    solved_indices = torch.tensor([state_key_to_indices(SOLVED_STATE_KEY)], dtype=torch.long, device=device)
    margin = config["training"]["margin"]
    grad_clip_norm = config["training"].get("grad_clip_norm")

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/cube-triplet") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")

    early_stop_metric = config["training"].get("early_stop_metric", "triplet_accuracy")
    early_stop_mode = config["training"].get("early_stop_mode", "max")
    best_triplet_acc = float("-inf")
    best_solved_cosine = float("-inf")
    best_early_stop_value = float("-inf") if early_stop_mode == "max" else float("inf")
    patience = 0
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")

        for batch in progress:
            anchor_states = batch["anchor"].to(device)
            positive_states = batch["positive"].to(device)
            negative_states = batch["negative"].to(device)

            anchors = encode_normalized(model, anchor_states)
            positives = encode_normalized(model, positive_states)
            negatives = encode_normalized(model, negative_states)
            mined_negatives = select_semi_hard_negatives(anchors, positives, negatives, margin)

            loss = triplet_loss(anchors, positives, mined_negatives, margin)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            if writer and global_step % tensorboard_config.get("log_every_steps", 10) == 0:
                writer.add_scalar("loss/triplet", loss.item(), global_step)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler:
            scheduler.step()

        train_metrics = {"loss": train_loss / len(train_loader)}
        val_metrics = evaluate(model, val_loader, device, margin, solved_indices)

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_triplet_accuracy={val_metrics['triplet_accuracy']:.4f} "
            f"val_solved_cosine={val_metrics['solved_cosine']:.4f} "
            f"val_length_correlation={val_metrics['length_correlation']:.4f}"
        )

        if writer:
            writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar("metrics/val_triplet_accuracy", val_metrics["triplet_accuracy"], epoch)
            writer.add_scalar("metrics/val_solved_cosine", val_metrics["solved_cosine"], epoch)
            writer.add_scalar("metrics/val_length_correlation", val_metrics["length_correlation"], epoch)
            if scheduler:
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_metrics["triplet_accuracy"] > best_triplet_acc:
            best_triplet_acc = val_metrics["triplet_accuracy"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_triplet_acc.pt")

        if val_metrics["solved_cosine"] > best_solved_cosine:
            best_solved_cosine = val_metrics["solved_cosine"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_solved_cosine.pt")

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
