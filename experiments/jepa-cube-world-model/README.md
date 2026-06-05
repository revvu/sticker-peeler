# JEPA Cube World Model

Action-conditioned JEPA prototype for Rubik's cube transitions: encode a start state and move sequence, predict the target-state embedding in latent space.

## Method

- **Encoder**: MLP over 54 face-index stickers → 128-d latent vector
- **Move context**: GRU over pruned move sequences (length 0–15)
- **Predictor**: MLP on `[z_start || move_context]` → predicted `z_end`
- **Target encoder**: EMA copy of context encoder (stop-gradient targets)
- **Loss**: L1 latent prediction + VICReg anti-collapse regularizer

Training data is generated in Node from **reachable** states only:

1. Apply a real 20-move scramble to solved
2. Apply a pruned continuation (0–15 moves, same rules as tree search)
3. Record `(startKey, moves, endKey)`

## Library note

This experiment uses a **custom PyTorch JEPA-style module** (`models/cube_jepa.py`) tailored to tabular cube states. The [PyPI `jepa`](https://pypi.org/project/jepa/) package is a good reference for action-conditioned patterns, but vision-oriented trainers were not used here.

## Research estimates

| Question | Answer |
|----------|--------|
| **Loss** | L1 on latent prediction + VICReg (variance + covariance) |
| **10k training time** | ~5–30 minutes on a laptop GPU / Apple Silicon |
| **10k data** | Enough to validate the pipeline; not enough for a general solver |
| **Practical scale** | 100k–1M for local dynamics; 200M+ for competitive solving (DeepCubeA literature) |

## Setup

```sh
cd experiments/jepa-cube-world-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset (10,000 examples)

```sh
node generate-dataset.js
```

Writes `data/dataset.jsonl` and `data/stats.json`.

## Train

```sh
python train.py --run-name cube-10k-v1
```

Checkpoints saved to `checkpoints/best.pt`.

## Export encoder for browser HUD

After training, export the context encoder weights to JSON for in-browser inference (no PyTorch):

```sh
python export_encoder.py
```

Writes `assets/jepa/encoder_weights.json` at the repo root. The Rubik's cube emulator loads this file and shows **(JEPA) Similarity to solved** in the top HUD — cosine similarity between the current state embedding and the solved-state embedding.

Re-run export after each training run to refresh the HUD weights.

## Live visualization (TensorBoard)

In a second terminal:

```sh
tensorboard --logdir experiments/jepa-cube-world-model/runs
```

Open http://localhost:6006 for live `loss/*` and `metrics/val_cosine` curves.

Optional Weights & Biases: set `logging.wandb.enabled: true` in `configs/default.yaml` and run `wandb login`.

## Metrics

| Metric | Meaning |
|--------|---------|
| `loss/l1` | Latent prediction error |
| `loss/vicreg` | Collapse prevention penalty |
| `metrics/val_cosine` | Alignment of predicted vs target embeddings |
| `metrics/val_nn_accuracy` | Nearest-neighbor retrieval of correct `endKey` |

## Results (initial 10k run)

Smoke training on Apple Silicon (100 epochs max, early stop patience 10):

- Best val total loss: ~0.97 (epoch 11)
- Val cosine: ~0.79 at best checkpoint
- Val NN accuracy: ~0% (expected at 10k — retrieval metric is strict)

## Training improvement roadmap

Ordered by expected impact for the next iteration:

| Priority | Change | Why |
|----------|--------|-----|
| 1 | **Scale data to 100k–500k** | 10k validates the pipeline; embeddings are not discriminative yet (NN accuracy ~0%) |
| 2 | **Increase continuation range** (0–20) | Match scramble depths; cover more dynamics |
| 3 | **Early stop on `val_cosine`** (maximize) not just `val_total` | Cosine kept improving after val loss plateaued |
| 4 | **More data diversity** — variable base scramble length (15–25), multiple continuations per start state | Reduces overfitting to unique start states |
| 5 | **Larger latent dim** (256) or deeper MLP | More capacity once data scales |
| 6 | **Multi-step rollout loss** (2-step predictor unroll) | V-JEPA 2-AC pattern; better long-horizon dynamics |
| 7 | **Export + use predictor** in browser | Enables latent rollout planning (future solver) |
| 8 | **Hard negatives in eval** — NN accuracy over full val bank | Better metric than cosine alone for "useful embedding" |

For the similarity HUD: even a weak model may show higher similarity when nearly solved and lower when scrambled — useful as a coarse distance indicator once retrained on more data.
