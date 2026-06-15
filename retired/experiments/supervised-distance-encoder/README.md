# Supervised Distance Encoder

DeepCubeA-style cost-to-go encoder for Rubik's cube states. Powers the **supervised-distance** solver and the **(SDL) Distance to solved** HUD line.

## Method

- **Encoder**: MLP over 54 face-index stickers → 128-d latent
- **Head**: linear → `softplus` → non-negative predicted distance to solved
- **Loss**: weighted Huber regression + trajectory ordinal margin
- **Labels**:
  - BFS depth 0–5: exact (~621k states)
  - Scramble depth 6–15: approximate (`lossWeight = 1.0`)
  - Scramble depth 16–40: approximate upper bound (`lossWeight = 0.3`)

Scramble depth from solved is a valid upper bound on true graph distance. God's number ≤ 20 implies long scrambles are reducible; the model learns exact calibration on BFS data and monotone structure on trajectories.

## Setup

```sh
cd retired/experiments/supervised-distance-encoder
source ../../../.venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

```sh
node generate-dataset.js
```

Outputs:
- `data/states.jsonl` — BFS + trajectory state rows
- `data/ordinal_pairs.jsonl` — consecutive trajectory pairs
- `data/extrapolation.jsonl` — held-out 25/30/40-move eval states

## Train

```sh
python train.py --run-name sdl-v1
```

Checkpoints: `best_mae_bfs.pt`, `best_greedy_step.pt`. Early stop on `mae_bfs`.

## Export

```sh
python export_encoder.py
```

Writes `assets/supervised-distance/encoder_weights.json`.

## Metrics

| Metric | Meaning |
|--------|---------|
| `val_mae_bfs` | MAE on held-out BFS 0–5 states (primary) |
| `val_spearman_6_15` | Spearman(scramble depth, prediction) on 6–15 val |
| `val_spearman_16_40` | Extrapolation monotonicity on 16–40 val |
| `val_spearman_extrapolation` | Spearman on fixed 25/30/40 eval set |
| `val_greedy_step_acc` | Ordinal pairs where later state predicts lower distance |
| `val_pred_at_solved` | Should be ≈ 0 |

## Results (sdl-v1)

- Dataset: 769,498 training states (619,498 BFS exact + 150,000 trajectory)
- Best `val_mae_bfs`: **0.136** (epoch 4)
- `val_spearman_6_15`: **0.954**
- `val_spearman_16_40`: **0.805**
- `val_pred_at_solved`: **~0.0**
- Early stopped at epoch 19 (patience 15 on `mae_bfs`)
- Exported weights: solved ~0.0, 3-move scramble ~3.4, 20-move scramble ~8.1

## Solver usage

Select **Supervised Distance** in the emulator. Each step picks the pruned legal move that minimizes predicted distance to solved.

**Note:** Emulator scramble uses simpler face-only pruning; solver and training use full move pruning from `movePruning.js`.

**Tree Search(10)+SDL fallback:** When tree search fails, the browser runs depth-limited BFS with per-state deduplication, depth **3** (not 5), and scores at most **2000** unique states so the NN fallback stays interactive.
