# Raw ResNet-10

Distance-from-solved classifier over a one-hot 9×6 cube image. Same setup as `raw-resnet`, but with a **10-block** ResNet and a **~1.004M-sample** training set.

## Representation

- Six 3×3 faces are tiled in two rows of three: `[F, B, U]` on top and `[D, R, L]` on bottom.
- Final spatial layout: **9 columns × 6 rows**.
- Input tensor shape: `(6, 6, 9)` — six one-hot color channels.

## Model

- Custom ResNet with **10 residual blocks** (`models/raw_resnet.py`).
- Output: logits over distances `0..26` (27 classes).
- Training loss: cross-entropy on scramble depth labels.

## Dataset

Paths start at the solved cube and take **26 pruned random moves**. Before the first move and after each move, we record `(stateKey, distance)` where `distance` is the number of moves applied so far. Each path contributes **27 samples** with distances `0..26`.

Default split (target **1,004,000** train samples; each path contributes 27 rows):

| Split | Paths | Samples |
|-------|------:|--------:|
| Train | 37,185 | 1,003,995 |
| Val | 3,718 | 100,386 |
| Test | 3,718 | 100,386 |

Train count is the largest multiple of 27 not exceeding 1,004,000. Val/test use 10% of train path count (same ratio as `raw-resnet`).

Every distance class appears equally often within each split.

Scramble depth is a proxy for graph distance, not an exact label beyond shallow depths.

## Setup

From the repo root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

```sh
cd experiments/raw-resnet-10
node generate-dataset.js
```

Writes:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`
- `data/stats.json`

## Train

```sh
cd experiments/raw-resnet-10
python train.py --run-name raw-resnet-10-v1
```

Checkpoints:

| File | Criterion |
|------|-----------|
| `best_accuracy.pt` | Best validation accuracy |
| `best_mae.pt` | Best validation MAE (expected distance) |

Early stop tracks `val_mae` by default.

## TensorBoard

```sh
tensorboard --logdir experiments/raw-resnet-10/runs
```

Useful scalars:

| Tag | Meaning |
|-----|---------|
| `train/loss`, `epoch/val_loss` | Cross-entropy |
| `train/accuracy`, `epoch/val_accuracy` | Exact class match |
| `train/mae`, `epoch/val_mae` | Expected distance error |
| `train/top3_accuracy`, `epoch/val_top3_accuracy` | Distance within top-3 logits |
| `metrics/val_distance_{d}_accuracy` | Per-distance validation accuracy |
| `samples/cube_faces_rgb` | Input preview images (RGB render of one-hot faces) |

## Pruning

Dataset generation uses the same move pruning as tree search (`src/solvers/utils/movePruning.js`):

- No consecutive turns on the same face
- Parallel faces follow alphabetical order
- No redundant opposite-face triples (e.g. `F B F`)
