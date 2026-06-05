# Triplet Cube Encoder

Metric-learning encoder for Rubik's cube states using FaceNet-style triplet loss. Powers the **triplet-loss** solver in the emulator.

## Method

- **Encoder**: MLP over 54 face-index stickers → 128-d L2-normalized embedding
- **Loss**: `max(0, d(a,p)² - d(a,n)² + margin)` with batch semi-hard negative mining
- **Labels**: From shallow anchors (0–15 moves from solved), compare pruned continuation lengths

## Research references

| Paper | Takeaway |
|-------|----------|
| [FaceNet (CVPR 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) | Triplet loss + semi-hard mining on normalized embeddings |
| [DeepCubeA (Nature MI 2019)](https://www.nature.com/articles/s42256-019-0070-z) | Train from shallow scrambles outward from goal |
| [CRTR (NeurIPS 2025)](https://princeton-rl.github.io/CRTR/) | Cube distances are hard to embed isometrically in Euclidean space |
| [Order Embeddings (ICLR 2016)](https://mlanthology.org/iclr/2016/vendrov2016iclr-order/) | Future upgrade for partial-order "closer to solved" |

## Dataset design

1. Seed `SOLVED_STATE_KEY` as an anchor
2. Sample shallow anchors: pruned scrambles 0–15 moves from solved
3. Per anchor, emit 7–8 triplets from pruned sequence pairs (length 0–15)
4. Shorter sequence → positive; longer → negative
5. Equal length → random positive/negative assignment (noisy labels)

**Cautions:** move-sequence length is a proxy for graph distance, not ground truth.

## Setup

```sh
cd experiments/triplet-cube-encoder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate triplets (~200k)

```sh
node generate-triplets.js
```

## Train

```sh
python train.py --run-name cube-triplet-v1
```

Checkpoints: `best_triplet_acc.pt`, `best_solved_cosine.pt`

## Export for browser solver

```sh
python export_encoder.py
```

Writes `assets/triplet/encoder_weights.json` (defaults to `best_solved_cosine.pt`).

## Metrics

| Metric | Meaning |
|--------|---------|
| `metrics/val_triplet_accuracy` | `d(a,p) + margin < d(a,n)` on val set |
| `metrics/val_solved_cosine` | Anchor alignment to solved embedding |
| `metrics/val_length_correlation` | Spearman(length diff, embedding distance) |

## Results (cube-triplet-v1)

- Dataset: 200k triplets, 26,650 unique anchors, solved anchor included
- Best `val_triplet_accuracy`: **67.4%** (epoch 1)
- Best `val_solved_cosine`: **0.198** (epoch 1)
- Early stopped at epoch 16 (patience 15 on `triplet_accuracy`)
- Exported weights: scrambled 3-move state cosine ~0.77 vs solved

## Solver usage

Select **Triplet Loss** in the emulator solver dropdown. Each step picks the pruned legal move that maximizes cosine similarity to the solved embedding.
