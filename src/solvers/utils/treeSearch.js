import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  getCubeStateKey,
  getMoveFace,
  invertMoves
} from "../../cube/CubeState.js";
import { isMoveAllowed } from "./movePruning.js";
import { buildTransformationCache } from "./transformationCache.js";

export const TREE_SEARCH_MAX_DEPTH = 10;

export function treeSearch(cubeState, maxDepth = TREE_SEARCH_MAX_DEPTH) {
  const startStateKey = getCubeStateKey(cubeState);

  if (startStateKey === SOLVED_STATE_KEY) {
    return {
      moves: [],
      found: true,
      exhausted: true,
      maxDepth,
      searchedNodes: 1
    };
  }

  const forwardDepth = Math.floor(maxDepth / 2);
  const backwardDepth = maxDepth - forwardDepth;
  const transformationCache = buildTransformationCache(backwardDepth);
  let searchedNodes = 1;
  let solutionMoves = null;

  function tryMatch(stateKey, prefixMoves) {
    const cached = transformationCache.get(stateKey);

    if (!cached) {
      return false;
    }

    const moves = [...prefixMoves, ...invertMoves(cached.moves)];

    if (moves.length > maxDepth) {
      return false;
    }

    solutionMoves = moves;
    return true;
  }

  function visit(stateKey, moves, historyFaces, remainingDepth) {
    if (tryMatch(stateKey, moves)) {
      return true;
    }

    if (remainingDepth === 0) {
      return false;
    }

    for (const move of ALL_MOVES) {
      if (!isMoveAllowed(historyFaces, move)) {
        continue;
      }

      const nextStateKey = applyMoveToKey(stateKey, move);
      searchedNodes += 1;

      if (
        visit(
          nextStateKey,
          [...moves, move],
          [...historyFaces, getMoveFace(move)],
          remainingDepth - 1
        )
      ) {
        return true;
      }
    }

    return false;
  }

  visit(startStateKey, [], [], forwardDepth);

  return {
    moves: solutionMoves,
    found: Boolean(solutionMoves),
    exhausted: true,
    maxDepth,
    searchedNodes
  };
}
