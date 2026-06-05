# Order-Embedding CNN

Vendrov-style order embeddings with a face-grid CNN encoder for asymmetric distance-to-solved. Powers the **order-embedding** solver and the **(OECNN) Distance to solved** HUD line.

## Method

- **Encoder**: shared Conv2d per face (6×3×3 grid) + cross-face neighbor fusion → 128-d embedding in ℝ₊ (softplus)
- **Distance**: `D(s) = ||relu(f(s) - f(solved))||₁` — no separate scalar head
- **Loss**: order-violation pairs + solved anchor + weighted Huber calibration on BFS labels
- **Labels**: same corpus as SDL (BFS 0–5 exact + scramble trajectories 6–40)

## Setup

```sh
cd experiments/order-embedding-cnn
source ../jepa-cube-world-model/.venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

```sh
node generate-dataset.js
```

Outputs:
- `data/states.jsonl`
- `data/order_pairs.jsonl`
- `data/ranking_triplets.jsonl`
- `data/extrapolation.jsonl`

## Train

On Apple Silicon Macs, training uses the **MPS GPU** automatically (`device: auto` in config). You can force it with `device: mps` in `configs/default.yaml`.

```sh
python train.py --run-name oecnn-v1
```

The training loop batches all face convolutions and fuses multiple encoder passes per step so MPS stays busy.

Checkpoints: `best_greedy_step.pt`, `best_order_acc.pt`, `best_mae_bfs.pt`.

## Export

```sh
python export_encoder.py
```

Writes `assets/order-embedding/encoder_weights.json`.

## Solver usage

Select **Order Embedding** in the emulator solver dropdown.
