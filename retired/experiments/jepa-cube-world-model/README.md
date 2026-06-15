# JEPA Cube World Model

Action-conditioned JEPA prototype for Rubik's cube transitions: encode a start state and move sequence, predict the target-state embedding in latent space.

## Method

- **Encoder**: MLP over 54 face-index stickers → 128-d latent vector
- **Move context**: GRU over pruned move sequences (length 0–20)
- **Predictor**: MLP on `[z_start || move_context]` → predicted `z_end`
- **Target encoder**: EMA copy of context encoder (stop-gradient targets)
- **Loss**: L1 latent prediction + VICReg anti-collapse regularizer

## Dataset v2 (shallow near-solved states)

Training data is generated in Node from **reachable** states only:

1. Start at `SOLVED_STATE_KEY` (always included as a `startKey`)
2. Sample shallow scrambles of **0–15** pruned moves to reach additional `startKey` values
3. For each `startKey`, emit **5–10** independent pruned continuations (0–20 moves)
4. Record `(startKey, moves, endKey)` per row (~200k total)

This focuses on partial-structure variation near solved — better aligned with the similarity-to-solved HUD than deep scrambles.

## Library note

This experiment uses a **custom PyTorch JEPA-style module** (`models/cube_jepa.py`) tailored to tabular cube states. The [PyPI `jepa`](https://pypi.org/project/jepa/) package is a good reference for action-conditioned patterns, but vision-oriented trainers were not used here.

## Setup

```sh
cd experiments/jepa-cube-world-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset (~200,000 examples)

```sh
node generate-dataset.js
```

Writes `data/dataset.jsonl` and `data/stats.json`.

## Train

```sh
python train.py --run-name cube-200k-v2
```

Checkpoints saved to `checkpoints/`:

| File | Criterion |
|------|-----------|
| `best_cosine.pt` | Best `val_cosine` (used for HUD export) |
| `best_loss.pt` | Best `val_total` |
| `best.pt` | Alias of `best_loss.pt` |

Training defaults (see `configs/default.yaml`):

- Early stop on `val_cosine` (patience 15)
- State-disjoint val split (hold out 15% of `startKey` values)
- `vicreg_weight: 0.05`, cosine LR decay, grad clip 1.0

## Export encoder for browser HUD

```sh
python export_encoder.py
```

Defaults to `best_cosine.pt` (falls back to `best.pt`). Writes `assets/jepa/encoder_weights.json` at the repo root.

## Live visualization (TensorBoard)

```sh
tensorboard --logdir experiments/jepa-cube-world-model/runs
```

Open http://localhost:6006 for `loss/*`, `metrics/val_cosine`, `metrics/val_solved_cosine`, and `metrics/val_encoder_nn_accuracy`.

## Metrics

| Metric | Meaning |
|--------|---------|
| `loss/l1` | Latent prediction error |
| `loss/vicreg` | Collapse prevention penalty |
| `epoch/val_l1` | Validation L1 (without VICReg noise) |
| `metrics/val_cosine` | Predicted vs target embedding alignment |
| `metrics/val_solved_cosine` | Context encoder alignment to solved (HUD proxy) |
| `metrics/val_nn_accuracy` | Predictor NN retrieval of correct `endKey` |
| `metrics/val_encoder_nn_accuracy` | Context encoder NN retrieval of correct `endKey` |

## Results

### v1 (10k deep scramble)

- Best `val_total`: ~0.97 (epoch 11)
- Final `val_cosine`: ~0.79
- `val_nn_accuracy`: ~0%

### v2 (200k shallow, multi-continuation)

`cube-200k-v2` on Apple Silicon (early stop epoch 16, patience 15 on `val_cosine`):

- Dataset: 200k rows, 26,650 unique shallow `startKey` values, `SOLVED_STATE_KEY` included
- Best `val_cosine`: **0.946** (epoch 1) → saved to `best_cosine.pt`
- Best `val_total`: **0.835** (epoch 11) → saved to `best_loss.pt`
- Peak `val_nn_accuracy`: ~2.2% (epoch 11)
- `val_solved_cosine` remained negative early in training — HUD export uses `best_cosine.pt`; re-export after longer runs if solved alignment is the priority
