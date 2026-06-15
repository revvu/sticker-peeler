# Triplet Cube Encoder

Metric-learning encoder for Rubik's cube states using FaceNet-style triplet loss. Powers the **triplet-loss** solver and the **(Triplet Loss) Similarity to solved** HUD line in the emulator.

## Method (v2)

- **Encoder**: MLP over 54 face-index stickers → 128-d L2-normalized embedding
- **Loss**: `triplet_loss + λ * solved_cosine_loss`
  - Triplet: `max(0, d(a,p)² - d(a,n)² + margin)` using **dataset negatives only** (no cross-batch mining)
  - Solved cosine: `1 - cosine(e(state), e(solved))` on anchors, positives, and negatives
- **Labels**: Scramble depth from solved (pruned sequence length, 0–15 moves) as distance proxy; skip equal-length pairs. Full BFS distance-to-solved is too slow beyond depth ~6 on the cube graph.

## Research references

| Paper | Takeaway |
|-------|----------|
| [FaceNet (CVPR 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) | Triplet loss on normalized embeddings; batch mining only when negatives are semantically valid |
| [DeepCubeA (Nature MI 2019)](https://www.nature.com/articles/s42256-019-0070-z) | Train from shallow scrambles outward from goal using cost-to-go from solved |
| [CRTR (NeurIPS 2025)](https://princeton-rl.github.io/CRTR/) | Cube distances are hard to embed isometrically in Euclidean space |
| [Order Embeddings (ICLR 2016)](https://mlanthology.org/iclr/2016/vendrov2016iclr-order/) | Future upgrade for partial-order "closer to solved" |

## Dataset design (v2)

1. Sample two independent pruned sequences from `SOLVED_STATE_KEY` (length 0–15)
2. Label by scramble depth (`sequence.length`) — approximate distance-to-solved
3. Skip pairs where lengths are equal (no random tie-break)
4. Anchor = `SOLVED_STATE_KEY`; positive = shorter scramble; negative = longer scramble
5. ~200k triplets

This directly trains what the HUD and solver measure: `cosine(e(state), e(solved))`.

## Setup

```sh
cd experiments/triplet-cube-encoder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or reuse the JEPA venv:

```sh
source experiments/jepa-cube-world-model/.venv/bin/activate
```

## Generate triplets (~200k)

```sh
node generate-triplets.js
```

## Train

```sh
python train.py --run-name cube-triplet-v2
```

Checkpoints: `best_triplet_acc.pt`, `best_solved_cosine.pt`. Early stop on `solved_cosine` (maximize).

## Export for browser solver + HUD

```sh
python export_encoder.py
```

Writes `assets/triplet/encoder_weights.json` (defaults to `best_solved_cosine.pt`).

## Metrics

| Metric | Meaning |
|--------|---------|
| `metrics/val_triplet_accuracy` | `d(a,p) + margin < d(a,n)` on val set |
| `metrics/val_solved_cosine` | Mean cosine of pos/neg embeddings to solved |
| `metrics/val_distance_correlation` | Spearman(scramble depth, embedding distance to solved) |

## v1 failure analysis (`cube-triplet-v1`)

Parsed 19 epochs from `runs/cube-triplet/cube-triplet-v1/`:

| Metric | Epoch 1 | Best | Epoch 19 | Verdict |
|--------|---------|------|----------|---------|
| `val_triplet_accuracy` | 67.4% | **68.7%** (ep 4) | 67.9% | Plateaued immediately |
| `val_solved_cosine` | 0.198 | **0.198** (ep 1) | 0.129 | Never improved |
| `val_length_correlation` | 0.002 | 0.002 | -0.002 | ~0 — no distance structure |

### Root causes

1. **Objective mismatch** — v1 trained anchor-local order (`d(a,p) < d(a,n)` by continuation move length from anchor). The solver/HUD need `cosine(e(state), e(solved))` — a global distance-to-solved ranking. DeepCubeA trains explicitly on cost-to-go from solved; CRTR learns temporal structure along trajectories from goal, not anchor-local relative distances.

2. **Noisy labels** — move-sequence length is a poor proxy for graph distance. 6.2% of triplets used random tie-break when lengths were equal. `val_length_correlation ≈ 0` confirms embeddings did not encode even the proxy.

3. **Batch semi-hard mining** — `select_semi_hard_negatives()` replaced dataset negatives with states from other triplets in the batch, with no guaranteed "farther from this anchor" relationship. This contradicts length-based labels and likely capped accuracy around ~68%.

4. **Wrong early stop** — tracked `triplet_accuracy` (plateaued ep 4) while `solved_cosine` degraded after epoch 1. Exported `best_solved_cosine.pt` was epoch 1 with only 0.20 cosine.

### v2 fixes

| Setting | v1 | v2 |
|---------|----|----|
| Triplet semantics | Shorter continuation from anchor | Lower scramble depth from solved |
| Mining | Cross-batch semi-hard | Dataset negatives only |
| Auxiliary loss | none | Solved cosine loss (`λ=0.25`) |
| Tie-break | random | skip equal-distance |
| Early stop | `triplet_accuracy` | `solved_cosine` |
| Margin | 0.2 | 0.1 |

## Results

### cube-triplet-v1 (failed)

- Dataset: 200k triplets, 26,650 unique anchors, move-length proxy labels
- Best `val_triplet_accuracy`: **68.7%** (epoch 4)
- Best `val_solved_cosine`: **0.198** (epoch 1)
- Early stopped at epoch 19 (patience 15 on `triplet_accuracy`)

### cube-triplet-v2

- Dataset: 200k triplets, solved anchor, scramble-depth labels (skip equal length)
- Best `val_solved_cosine`: **0.928** (epoch 3)
- Best `val_distance_correlation`: **0.892** (epoch 4)
- Early stopped at epoch 18 (patience 15 on `solved_cosine`)
- Exported weights: `assets/triplet/encoder_weights.json` from `best_solved_cosine.pt` (epoch 3)

## Solver usage

Select **Triplet Loss** in the emulator solver dropdown. Each step picks the pruned legal move that maximizes cosine similarity to the solved embedding.
