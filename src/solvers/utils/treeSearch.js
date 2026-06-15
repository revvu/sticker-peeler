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

export function runForwardSearch(startStateKey, transformationCache, forwardDepth, maxDepth) {
  const searchStart = performance.now();
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
    searchedNodes,
    searchMs: performance.now() - searchStart
  };
}

export function treeSearch(cubeState, maxDepth = TREE_SEARCH_MAX_DEPTH) {
  const startStateKey = getCubeStateKey(cubeState);

  if (startStateKey === SOLVED_STATE_KEY) {
    return {
      moves: [],
      found: true,
      exhausted: true,
      maxDepth,
      searchedNodes: 1,
      cacheBuildMs: 0,
      searchMs: 0,
      cacheHit: true
    };
  }

  const forwardDepth = Math.floor(maxDepth / 2);
  const backwardDepth = maxDepth - forwardDepth;

  const cacheBuildStart = performance.now();
  const { cache: transformationCache, cacheHit } = buildTransformationCache(backwardDepth);
  const cacheBuildMs = performance.now() - cacheBuildStart;

  const forwardResult = runForwardSearch(
    startStateKey,
    transformationCache,
    forwardDepth,
    maxDepth
  );

  return {
    moves: forwardResult.moves,
    found: forwardResult.found,
    exhausted: true,
    maxDepth,
    searchedNodes: forwardResult.searchedNodes,
    cacheBuildMs,
    searchMs: forwardResult.searchMs,
    cacheHit
  };
}
