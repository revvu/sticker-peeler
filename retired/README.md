# Retired models

Neural distance-to-solved experiments and their browser runtime code. These are no longer wired into the emulator UI.

| Path | Former role |
|---|---|
| `experiments/jepa-cube-world-model` | JEPA world model |
| `experiments/triplet-cube-encoder` | Triplet-loss metric encoder |
| `experiments/supervised-distance-encoder` | Supervised distance (SDL) |
| `experiments/order-embedding-cnn` | Order-embedding CNN |
| `assets/*` | Exported encoder weights |
| `src/*` | Browser inference + solvers |

The active app uses **Tree Search(10)** only. See `experiment-checkpoints/EXPERIMENTS.md` for results and takeaways.

## Python environment

Use the shared repo-root venv and dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per-experiment `requirements.txt` files under `retired/experiments/` include the root file.
