import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  getMoveFace
} from "../../src/cube/CubeState.js";
import { isMoveAllowed } from "../../src/solvers/utils/movePruning.js";

const DEPTH = 5;

function measureSingleDirectionSearch(maxDepth) {
  const uniqueStates = new Set([SOLVED_STATE_KEY]);
  let nodesExplored = 1;

  function visit(stateKey, historyFaces, remainingDepth) {
    if (remainingDepth === 0) {
      return;
    }

    for (const move of ALL_MOVES) {
      if (!isMoveAllowed(historyFaces, move)) {
        continue;
      }

      const nextStateKey = applyMoveToKey(stateKey, move);
      nodesExplored += 1;
      uniqueStates.add(nextStateKey);

      visit(
        nextStateKey,
        [...historyFaces, getMoveFace(move)],
        remainingDepth - 1
      );
    }
  }

  visit(SOLVED_STATE_KEY, [], maxDepth);

  return {
    depth: maxDepth,
    nodesExplored,
    uniqueStates: uniqueStates.size
  };
}

const result = measureSingleDirectionSearch(DEPTH);

console.log(`Depth ${result.depth} single-direction search (from solved):`);
console.log(`1) nodes explored: ${result.nodesExplored}`);
console.log(`2) unique cube states: ${result.uniqueStates}`);
