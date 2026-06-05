# Depth-5 single-direction search counts

Measured how large the pruned search tree is when expanding from the solved cube up to 5 moves deep — the same single-direction pass used to build the backward index in tree search.

## Method

`measure-single-direction-search.js` walks all allowed move sequences from the solved state, using the solver's pruning rules:

- No consecutive turns on the same face
- No redundant opposite-face pairs (e.g. `F B F`, `F B B`)
- Parallel faces must follow alphabetical order (e.g. `B` then `F`, not `F` then `B`)

Two metrics are collected:

1. **Nodes explored** — total search-tree visits (a state counts again if reached via a different path)
2. **Unique cube states** — distinct configurations reachable within 5 moves

## Results

| Metric | Count |
|--------|------:|
| Nodes explored | 624,124 |
| Unique cube states | 621,649 |

The ~2,475 gap is duplicate visits: some states are reachable by more than one move sequence.

## Run

```sh
node experiments/depth-5-search-node-count/measure-single-direction-search.js
```
