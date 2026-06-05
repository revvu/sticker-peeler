from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from featurize import FACE_ORDER, state_key_to_face_grid
from models.order_model import OrderModel

ROOT = Path(__file__).resolve().parent
SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)
STATE_KEY_FIELDS = ("stateKey", "closerKey", "fartherKey", "aKey", "bKey", "cKey", "earlierKey", "laterKey")


def face_grid_tensor(state_key: str) -> torch.Tensor:
    return torch.tensor(state_key_to_face_grid(state_key), dtype=torch.long)


class FaceGridCache:
    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, state_key: str) -> torch.Tensor:
        cached = self._cache.get(state_key)
        if cached is None:
            cached = face_grid_tensor(state_key)
            self._cache[state_key] = cached
        return cached

    def prime(self, state_keys: Iterable[str]) -> None:
        for state_key in state_keys:
            self.get(state_key)

    def __len__(self) -> int:
        return len(self._cache)


def collect_state_keys(*example_lists: list[dict]) -> set[str]:
    keys = {SOLVED_STATE_KEY}
    for examples in example_lists:
        for example in examples:
            for field in STATE_KEY_FIELDS:
                value = example.get(field)
                if value:
                    keys.add(value)
    return keys


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available on this Mac.")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(device, non_blocking=device.type == "cuda")


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


class StateDataset(Dataset):
    def __init__(self, examples: list[dict], cache: FaceGridCache):
        self.examples = examples
        self.cache = cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "state": self.cache.get(example["stateKey"]),
            "distance": torch.tensor(example["distance"], dtype=torch.float32),
            "loss_weight": torch.tensor(example["lossWeight"], dtype=torch.float32),
            "label_type": example["labelType"],
        }


class OrderPairDataset(Dataset):
    def __init__(self, examples: list[dict], cache: FaceGridCache):
        self.examples = examples
        self.cache = cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "closer": self.cache.get(example["closerKey"]),
            "farther": self.cache.get(example["fartherKey"]),
            "gap": torch.tensor(example["gap"], dtype=torch.float32),
            "loss_weight": torch.tensor(example["lossWeight"], dtype=torch.float32),
            "label_type": example["labelType"],
        }


class RankingTripletDataset(Dataset):
    def __init__(self, examples: list[dict], cache: FaceGridCache):
        self.examples = examples
        self.cache = cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "a": self.cache.get(example["aKey"]),
            "b": self.cache.get(example["bKey"]),
            "c": self.cache.get(example["cKey"]),
            "gap_ab": torch.tensor(example["gapAb"], dtype=torch.float32),
            "gap_bc": torch.tensor(example["gapBc"], dtype=torch.float32),
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


def subsample_solved_anchored(
    examples: list[dict],
    target_count: int,
    seed: int,
) -> list[dict]:
    solved_pairs = [row for row in examples if row["closerKey"] == SOLVED_STATE_KEY]
    other_pairs = [row for row in examples if row["closerKey"] != SOLVED_STATE_KEY]

    if len(solved_pairs) <= target_count:
        return examples

    rng = random.Random(seed)
    rng.shuffle(solved_pairs)
    return other_pairs + solved_pairs[:target_count]


def weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    raw = F.huber_loss(predictions, targets, delta=delta, reduction="none")
    normalized = weights / weights.sum().clamp_min(1e-6)
    return (raw * normalized).sum()


def weighted_order_loss(
    violations: torch.Tensor,
    gaps: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    raw = violations / gaps.clamp_min(1.0)
    normalized = weights / weights.sum().clamp_min(1e-6)
    return (raw * normalized).sum()


def save_checkpoint(
    model: OrderModel,
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
def get_solved_embedding(
    model: OrderModel,
    device: torch.device,
    cache: FaceGridCache,
) -> torch.Tensor:
    solved_grid = to_device(cache.get(SOLVED_STATE_KEY).unsqueeze(0), device)
    return model.encode(solved_grid).squeeze(0)


@torch.no_grad()
def predict_distances(
    model: OrderModel,
    states: torch.Tensor,
    solved_emb: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    state_emb = model.encode(states.to(device))
    return model.distance_to_solved(state_emb, solved_emb)


@torch.no_grad()
def evaluate(
    model: OrderModel,
    state_loader: DataLoader,
    order_loader: DataLoader,
    extrapolation_examples: list[dict],
    device: torch.device,
    order_violation_eps: float,
    cache: FaceGridCache,
) -> dict[str, float]:
    model.eval()
    solved_emb = get_solved_embedding(model, device, cache)

    bfs_abs_errors = []
    mid_labels = []
    mid_preds = []
    long_labels = []
    long_preds = []
    order_correct = 0
    order_total = 0
    greedy_correct = 0
    greedy_total = 0

    for batch in state_loader:
        predictions = predict_distances(model, to_device(batch["state"], device), solved_emb, device)
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

    for batch in order_loader:
        combined_grids = torch.cat(
            [to_device(batch["closer"], device), to_device(batch["farther"], device)],
            dim=0,
        )
        combined_emb = model.encode(combined_grids)
        batch_size = batch["closer"].shape[0]
        closer_emb = combined_emb[:batch_size]
        farther_emb = combined_emb[batch_size:]
        violations = model.order_violation(closer_emb, farther_emb)
        distances_closer = model.distance_to_solved(closer_emb, solved_emb)
        distances_farther = model.distance_to_solved(farther_emb, solved_emb)

        order_total += violations.shape[0]
        order_correct += int((violations <= order_violation_eps).sum().item())
        greedy_total += distances_closer.shape[0]
        greedy_correct += int((distances_farther > distances_closer).sum().item())

    extrapolation_labels = []
    extrapolation_preds = []
    if extrapolation_examples:
        for example in extrapolation_examples:
            state = to_device(cache.get(example["stateKey"]).unsqueeze(0), device)
            prediction = predict_distances(model, state, solved_emb, device).item()
            extrapolation_labels.append(example["distance"])
            extrapolation_preds.append(prediction)

    pred_at_solved = predict_distances(
        model,
        to_device(cache.get(SOLVED_STATE_KEY).unsqueeze(0), device),
        solved_emb,
        device,
    ).item()

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
        "order_acc": order_correct / max(order_total, 1),
        "greedy_step_acc": greedy_correct / max(greedy_total, 1),
        "D_at_solved": pred_at_solved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--run-name", default="oecnn-v1")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = resolve_device(config["training"].get("device", "auto"))
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
    print(f"Training device: {device}")
    checkpoint_dir = ROOT / config["paths"]["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_examples = load_jsonl(ROOT / config["paths"]["states_dataset"])
    order_examples = load_jsonl(ROOT / config["paths"]["order_pairs_dataset"])
    ranking_examples = load_jsonl(ROOT / config["paths"]["ranking_triplets_dataset"])
    extrapolation_examples = load_jsonl(ROOT / config["paths"]["extrapolation_dataset"])

    train_states, val_states = split_examples(
        state_examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"],
    )
    train_orders, val_orders = split_examples(
        order_examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"] + 1,
    )
    train_rankings, _val_rankings = split_examples(
        ranking_examples,
        train_split=config["training"]["train_split"],
        seed=config["training"]["seed"] + 2,
    )

    train_orders = subsample_solved_anchored(
        train_orders,
        target_count=config["training"]["solved_anchor_subsample"],
        seed=config["training"]["seed"] + 3,
    )

    face_grid_cache = FaceGridCache()
    print("Precomputing face-grid tensors for dataset cache...")
    face_grid_cache.prime(
        collect_state_keys(
            state_examples,
            order_examples,
            ranking_examples,
            extrapolation_examples,
        )
    )
    print(f"Cached {len(face_grid_cache)} unique cube states")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    print(
        f"Loaded {len(state_examples)} states "
        f"({len(train_states)} train, {len(val_states)} val), "
        f"{len(order_examples)} order pairs ({len(train_orders)} train), "
        f"{len(ranking_examples)} ranking triplets, "
        f"{len(extrapolation_examples)} extrapolation states"
    )

    train_state_loader = make_dataloader(
        StateDataset(train_states, face_grid_cache),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_state_loader = make_dataloader(
        StateDataset(val_states, face_grid_cache),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    train_order_loader = make_dataloader(
        OrderPairDataset(train_orders, face_grid_cache),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_order_loader = make_dataloader(
        OrderPairDataset(val_orders, face_grid_cache),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    train_ranking_loader = make_dataloader(
        RankingTripletDataset(train_rankings, face_grid_cache),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_rankings) >= batch_size,
    )

    model = OrderModel(
        latent_dim=config["model"]["latent_dim"],
        sticker_embed_dim=config["model"]["sticker_embed_dim"],
        conv_channels=config["model"]["conv_channels"],
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

    anchor_weight = config["training"]["anchor_weight"]
    calibration_weight = config["training"]["calibration_weight"]
    huber_delta = config["training"]["huber_delta"]
    grad_clip_norm = config["training"].get("grad_clip_norm")

    tensorboard_config = config.get("logging", {}).get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", True):
        log_dir = ROOT / tensorboard_config.get("log_dir", "runs/order-embedding-cnn") / args.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard log dir: {log_dir}")

    early_stop_metric = config["training"].get("early_stop_metric", "greedy_step_acc")
    early_stop_mode = config["training"].get("early_stop_mode", "max")
    best_mae_bfs = float("inf")
    best_order_acc = float("-inf")
    best_greedy_step = float("-inf")
    best_early_stop_value = float("inf") if early_stop_mode == "min" else float("-inf")
    patience = 0
    global_step = 0
    order_iter = iter(train_order_loader)
    ranking_iter = iter(train_ranking_loader) if len(train_rankings) > 0 else None

    solved_grid = to_device(face_grid_cache.get(SOLVED_STATE_KEY).unsqueeze(0), device)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_order_loss = 0.0
        train_cal_loss = 0.0
        train_anchor_loss = 0.0
        progress = tqdm(train_state_loader, desc=f"epoch {epoch + 1}")

        for batch in progress:
            try:
                order_batch = next(order_iter)
            except StopIteration:
                order_iter = iter(train_order_loader)
                order_batch = next(order_iter)

            ranking_batch = None
            if ranking_iter is not None and len(train_rankings) > 0:
                try:
                    ranking_batch = next(ranking_iter)
                except StopIteration:
                    ranking_iter = iter(train_ranking_loader)
                    ranking_batch = next(ranking_iter)

            encode_grids = [
                solved_grid,
                to_device(batch["state"], device),
                to_device(order_batch["closer"], device),
                to_device(order_batch["farther"], device),
            ]
            if ranking_batch is not None:
                encode_grids.extend(
                    [
                        to_device(ranking_batch["a"], device),
                        to_device(ranking_batch["b"], device),
                        to_device(ranking_batch["c"], device),
                    ]
                )

            combined_grids = torch.cat(encode_grids, dim=0)
            combined_emb = model.encode(combined_grids)

            offset = 0
            solved_emb = combined_emb[offset]
            offset += 1

            state_count = batch["state"].shape[0]
            state_emb = combined_emb[offset : offset + state_count]
            offset += state_count

            closer_count = order_batch["closer"].shape[0]
            closer_emb = combined_emb[offset : offset + closer_count]
            offset += closer_count

            farther_emb = combined_emb[offset : offset + closer_count]
            offset += closer_count

            distances = model.distance_to_solved(state_emb, solved_emb.expand_as(state_emb))

            cal_component = weighted_huber_loss(
                distances,
                to_device(batch["distance"], device),
                to_device(batch["loss_weight"], device),
                huber_delta,
            )
            anchor_component = solved_emb.pow(2).sum()

            order_violations = model.order_violation(closer_emb, farther_emb)
            order_component = weighted_order_loss(
                order_violations,
                to_device(order_batch["gap"], device),
                to_device(order_batch["loss_weight"], device),
            )

            ranking_component = torch.tensor(0.0, device=device)
            if ranking_batch is not None:
                triplet_count = ranking_batch["a"].shape[0]
                emb_a = combined_emb[offset : offset + triplet_count]
                emb_b = combined_emb[offset + triplet_count : offset + 2 * triplet_count]
                emb_c = combined_emb[offset + 2 * triplet_count : offset + 3 * triplet_count]
                violation_ab = model.order_violation(emb_a, emb_b)
                violation_bc = model.order_violation(emb_b, emb_c)
                ranking_weights = to_device(ranking_batch["loss_weight"], device)
                ranking_normalized = ranking_weights / ranking_weights.sum().clamp_min(1e-6)
                ranking_component = (
                    violation_ab / to_device(ranking_batch["gap_ab"], device).clamp_min(1.0)
                    + violation_bc / to_device(ranking_batch["gap_bc"], device).clamp_min(1.0)
                )
                ranking_component = (ranking_component * ranking_normalized).sum()

            loss = (
                order_component
                + ranking_component
                + anchor_weight * anchor_component
                + calibration_weight * cal_component
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_loss += loss.item()
            train_order_loss += order_component.item()
            train_cal_loss += cal_component.item()
            train_anchor_loss += anchor_component.item()
            global_step += 1

            if writer and global_step % tensorboard_config.get("log_every_steps", 10) == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/order", order_component.item(), global_step)
                writer.add_scalar("loss/calibration", cal_component.item(), global_step)
                writer.add_scalar("loss/anchor", anchor_component.item(), global_step)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                order=f"{order_component.item():.4f}",
            )

        if scheduler:
            scheduler.step()

        train_metrics = {
            "loss": train_loss / len(train_state_loader),
            "order_loss": train_order_loss / len(train_state_loader),
            "calibration_loss": train_cal_loss / len(train_state_loader),
            "anchor_loss": train_anchor_loss / len(train_state_loader),
        }
        val_metrics = evaluate(
            model,
            val_state_loader,
            val_order_loader,
            extrapolation_examples,
            device,
            config["training"]["order_violation_eps"],
            face_grid_cache,
        )

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_mae_bfs={val_metrics['mae_bfs']:.4f} "
            f"val_order_acc={val_metrics['order_acc']:.4f} "
            f"val_spearman_6_15={val_metrics['spearman_6_15']:.4f} "
            f"val_spearman_16_40={val_metrics['spearman_16_40']:.4f} "
            f"val_spearman_extrapolation={val_metrics['spearman_extrapolation']:.4f} "
            f"val_greedy_step_acc={val_metrics['greedy_step_acc']:.4f} "
            f"val_D_at_solved={val_metrics['D_at_solved']:.4f}"
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

        if val_metrics["order_acc"] > best_order_acc:
            best_order_acc = val_metrics["order_acc"]
            save_checkpoint(model, config, epoch + 1, val_metrics, checkpoint_dir, "best_order_acc.pt")

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
