# Experiment Checkpoints

Summary of experiments and what we learned from each.

**Note:** Experiments 2–5 (JEPA, triplet, SDL, order-embedding) and their browser runtime live under `retired/`. The emulator now uses tree search only.

## Overview

There are **5 experiments**: one search-space measurement (still in `experiments/`) and four learned "distance to solved" approaches (retired). The four ML experiments previously exported weights into `assets/` and powered emulator solvers / HUD lines.

| Experiment | Goal | Status |
|---|---|---|
| `experiments/depth-5-search-node-count` | Measure BFS tree size | Active (informs tree search) |
| `retired/experiments/jepa-cube-world-model` | Predict next-state embeddings | Retired |
| `retired/experiments/triplet-cube-encoder` | Metric learning → cosine to solved | Retired |
| `retired/experiments/supervised-distance-encoder` | DeepCubeA-style cost-to-go | Retired |
| `retired/experiments/order-embedding-cnn` | Vendrov order embeddings + CNN | Retired |

---

## 1. Depth-5 search node count

**What it is:** A one-off measurement script — how big the pruned single-direction search tree is from solved out to 5 moves (same rules as the backward index).

**Takeaways:**

- **624,124** nodes explored, **621,649** unique states (~2.5k duplicate paths)
- That ~621k figure is why exact BFS labels for depth 0–5 are practical and show up again in SDL / order-embedding datasets
- Tree Search(10) uses bidirectional search with a pruned backward index built from the same ~621k-state depth-5 corpus measured in `experiments/depth-5-search-node-count`

---

## 2. JEPA cube world model

**What it is:** Action-conditioned JEPA — encode start state + move sequence, predict target embedding. Custom PyTorch module, not the PyPI `jepa` package.

**Takeaways:**

- **v1 (10k deep scrambles):** weak — ~0.79 val cosine, ~0% NN retrieval
- **v2 (200k shallow 0–15-move starts, multi-continuation):** much better transition prediction — **0.946 val cosine**, but only **~2.2%** NN accuracy
- **`val_solved_cosine` stayed negative** early on; export uses `best_cosine.pt`, which optimizes transition alignment, not "closeness to solved"
- **Lesson:** JEPA learns local dynamics well, but it's a poor direct proxy for a distance-to-solved heuristic / greedy solver

---

## 3. Triplet cube encoder

**What it is:** FaceNet-style triplet loss on 128-d L2-normalized embeddings; greedy solver maximizes `cosine(e(state), e(solved))`.

**Takeaways:**

- **v1 failed** (~68.7% triplet acc, ~0.20 solved cosine, ~0 Spearman). Root causes documented in the README:
  - Trained **anchor-local** order, but the app needs **global distance-from-solved**
  - Noisy labels + random tie-breaks on equal-length pairs
  - Cross-batch semi-hard mining contradicted the labels
  - Early-stopped on the wrong metric (`triplet_accuracy` while `solved_cosine` degraded)
- **v2 fixed it** (solved anchor, scramble-depth labels, skip ties, solved-cosine aux loss, no mining):
  - **0.928** val solved cosine, **0.892** distance correlation
- **Lesson:** Metric learning works if the objective matches what the solver actually scores, and labels are anchored at solved — not arbitrary continuation lengths

---

## 4. Supervised distance encoder (SDL)

**What it is:** MLP + scalar head predicting non-negative distance; weighted Huber on BFS exact labels + ordinal margins on trajectories.

**Takeaways (sdl-v1):**

- **0.136 MAE** on BFS 0–5 (exact labels from the ~621k-state corpus)
- **0.954 Spearman** on scramble depth 6–15; **0.805** on 16–40 extrapolation
- Predictions calibrate sensibly: solved ≈ 0, 3-move ≈ 3.4, 20-move ≈ 8.1
- **Lesson:** Mixing **exact BFS (0–5)** with **scramble-depth upper bounds (6–40)** gives the strongest calibrated cost-to-go signal — this is the most DeepCubeA-aligned approach and the best documented numbers overall

---

## 5. Order-embedding CNN

**What it is:** Vendrov-style asymmetric order distance `D(s) = ||relu(f(s) − f(solved))||₁` with a face-grid CNN encoder; same label corpus as SDL.

**Takeaways:**

- Implemented, trained (`oecnn-v1`), and exported to `assets/order-embedding/encoder_weights.json`
- **No run results are documented in the README yet** (unlike SDL / triplet / JEPA)
- Motivation from triplet README: cube distance isn't cleanly isometric in Euclidean space — order embeddings are the natural next step for partial-order "closer to solved"

---

## Cross-cutting themes

1. **Match training to deployment.** The triplet v1 postmortem is the clearest example: optimize the metric the HUD/solver actually uses.
2. **Anchor at solved, train outward.** Shallow scrambles (0–15) and BFS-near-solved data work better than deep random scrambles for proximity scoring.
3. **Exact labels where cheap, proxies where necessary.** BFS 0–5 is exact and huge (~621k states); beyond that, scramble depth is an upper bound, not true graph distance.
4. **Different architectures for different jobs:**
   - **SDL** → calibrated scalar distance (best documented metrics)
   - **Triplet v2** → ranking / cosine similarity (strong after the v2 redesign)
   - **JEPA** → transition modeling, not solved proximity
   - **Order embedding** → partial-order geometry (exported, metrics TBD)
5. **Search is the active solver.** Tree search handles ≤10-move solutions exactly. Retired NN models explored cost-to-go heuristics for harder cases.
