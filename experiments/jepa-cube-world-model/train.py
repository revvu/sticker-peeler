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

from featurize import FACE_ORDER, moves_to_indices, state_key_to_indices
from models.cube_jepa import CubeJEPA, jepa_loss, pad_move_sequences

ROOT = Path(__file__).resolve().parent
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


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
            "start_key": example["startKey"],
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
        "start_keys": [item["start_key"] for item in batch],
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


def split_examples_random(examples: list[dict], train_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * train_split)
    return shuffled[:split_index], shuffled[split_index:]


def split_examples_by_start_key(examples: list[dict], train_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    start_keys = sorted({example["startKey"] for example in examples})
    rng = random.Random(seed)
    rng.shuffle(start_keys)
    split_index = max(1, int(len(start_keys) * train_split))
    if split_index >= len(start_keys):
        split_index = len(start_keys) - 1

    train_keys = set(start_keys[:split_index])
    train_examples = [example for example in examples if example["startKey"] in train_keys]
    val_examples = [example for example in examples if example["startKey"] not in train_keys]
    return train_examples, val_examples


def split_examples(
    examples: list[dict],
    train_split: float,
    seed: int,
    state_disjoint: bool,
) -> tuple[list[dict], list[dict]]:
    if state_disjoint:
        return split_examples_by_start_key(examples, train_split, seed)
    return split_examples_random(examples, train_split, seed)


def save_checkpoint(
    model: CubeJEPA,
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


def is_metric_improved(
    metric_name: str,
    current: float,
    best: float,
    mode: str,
) -> bool:
    if mode == "max":
        return current > best
    return current < best


@torch.no_grad()
def evaluate(
    model: CubeJEPA,
    loader: DataLoader,
    device: torch.device,
    vicreg_weight: float,
    solved_indices: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "l1": 0.0, "vicreg": 0.0, "cosine": 0.0, "solved_cosine": 0.0}
    batches = 0
    nn_hits = 0
    nn_total = 0
    encoder_nn_hits = 0
    encoder_nn_total = 0
    bank_embeddings: list[torch.Tensor] = []
    bank_keys: list[str] = []
    encoder_bank_embeddings: list[torch.Tensor] = []
    encoder_bank_keys: list[str] = []

    solved_embedding = F.normalize(model.encode_context(solved_indices), dim=-1)

    for batch in loader:
        predicted = model.predict(batch["start"], batch["move_indices"], batch["move_lengths"])
        target = model.encode_target(batch["end"])
        _, parts = jepa_loss(predicted, target, vicreg_weight=vicreg_weight)

        totals["total"] += parts["total"]
        totals["l1"] += parts["l1"]
        totals["vicreg"] += parts["vicreg"]
        totals["cosine"] += F.cosine_similarity(predicted, target, dim=-1).mean().item()

        start_embeddings = F.normalize(model.encode_context(batch["start"]), dim=-1)
        totals["solved_cosine"] += F.cosine_similarity(start_embeddings, solved_embedding.expand_as(start_embeddings), dim=-1).mean().item()

        batches += 1

        bank_embeddings.append(target)
        bank_keys.extend(batch["end_keys"])
        encoder_bank_embeddings.append(target)
        encoder_bank_keys.extend(batch["end_keys"])

    if batches == 0:
        return totals

    bank = F.normalize(torch.cat(bank_embeddings, dim=0), dim=-1)
    encoder_bank = F.normalize(torch.cat(encoder_bank_embeddings, dim=0), dim=-1)

    for batch in loader:
        predicted = model.predict(batch["start"], batch["move_indices"], batch["move_lengths"])
        predicted = F.normalize(predicted, dim=-1)

        for row_index, end_key in enumerate(batch["end_keys"]):
            similarity = predicted[row_index] @ bank.T
            nearest_index = int(similarity.argmax().item())
            if bank_keys[nearest_index] == end_key:
                nn_hits += 1
            nn_total += 1

        context_end = F.normalize(model.encode_context(batch["end"]), dim=-1)
        for row_index, end_key in enumerate(batch["end_keys"]):
            similarity = context_end[row_index] @ encoder_bank.T
            nearest_index = int(similarity.argmax().item())
            if encoder_bank_keys[nearest_index] == end_key:
                encoder_nn_hits += 1
            encoder_nn_total += 1

    return {
        "total": totals["total"] / batches,
        "l1": totals["l1"] / batches,
        "vicreg": totals["vicreg"] / batches,
        "cosine": totals["cosine"] / batches,
        "solved_cosine": totals["solved_cosine"] / batches,
        "nn_accuracy": nn_hits / max(nn_total, 1),
        "encoder_nn_accuracy": encoder_nn_hits / max(encoder_nn_total, 1),
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
    parser.add_argument("--run-name", default="cube-200k-v2")
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
        state_disjoint=config["training"].get("state_disjoint_split", True),
    )
    print(f"Loaded {len(examples)} examples ({len(train_examples)} train, {len(val_examples)} val)")

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

    scheduler = None
    if config["training"].get("lr_schedule") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    solved_indices = torch.tensor([state_key_to_indices(SOLVED_STATE_KEY)], dtype=torch.long, device=device)

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/cube-jepa") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")

    wandb = maybe_init_wandb(config, args.run_name)

    early_stop_metric = config["training"].get("early_stop_metric", "val_cosine")
    early_stop_mode = config["training"].get("early_stop_mode", "max")
    metric_key = early_stop_metric.removeprefix("val_")

    best_loss = float("inf")
    best_cosine = float("-inf")
    best_early_stop_value = float("-inf") if early_stop_mode == "max" else float("inf")
    patience = 0
    global_step = 0
    grad_clip_norm = config["training"].get("grad_clip_norm")

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
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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

        if scheduler:
            scheduler.step()

        train_metrics = {key: value / len(train_loader) for key, value in train_totals.items()}
        val_metrics = evaluate(model, val_loader, device, config["training"]["vicreg_weight"], solved_indices)

        print(
            f"epoch {epoch + 1}: "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_l1={val_metrics['l1']:.4f} "
            f"val_cosine={val_metrics['cosine']:.4f} "
            f"val_solved_cosine={val_metrics['solved_cosine']:.4f} "
            f"val_nn_accuracy={val_metrics['nn_accuracy']:.4f} "
            f"val_encoder_nn_accuracy={val_metrics['encoder_nn_accuracy']:.4f}"
        )

        if writer:
            writer.add_scalar("epoch/train_total", train_metrics["total"], epoch)
            writer.add_scalar("epoch/val_total", val_metrics["total"], epoch)
            writer.add_scalar("epoch/val_l1", val_metrics["l1"], epoch)
            writer.add_scalar("metrics/val_cosine", val_metrics["cosine"], epoch)
            writer.add_scalar("metrics/val_solved_cosine", val_metrics["solved_cosine"], epoch)
            writer.add_scalar("metrics/val_nn_accuracy", val_metrics["nn_accuracy"], epoch)
            writer.add_scalar("metrics/val_encoder_nn_accuracy", val_metrics["encoder_nn_accuracy"], epoch)
            if scheduler:
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                }
            )

        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_loss.pt")
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best.pt")

        if val_metrics["cosine"] > best_cosine:
            best_cosine = val_metrics["cosine"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_cosine.pt")

        current_early_stop_value = val_metrics.get(metric_key)
        if current_early_stop_value is None:
            raise ValueError(f"Unknown early stop metric: {early_stop_metric}")

        if is_metric_improved(metric_key, current_early_stop_value, best_early_stop_value, early_stop_mode):
            best_early_stop_value = current_early_stop_value
            patience = 0
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
