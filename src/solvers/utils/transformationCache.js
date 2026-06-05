import {
  ALL_MOVES,
  IDENTITY_PERMUTATION,
  SOLVED_STATE_KEY,
  composePermutations,
  getMoveFace,
  getMovePermutation,
  getTransformationKey
} from "../../cube/CubeState.js";
import { isMoveAllowed } from "./movePruning.js";

const cachesByDepth = new Map();

export function hasTransformationCache(maxDepth) {
  return cachesByDepth.has(maxDepth);
}

function storeTransformation(cache, permutation, moves) {
  const transformKey = getTransformationKey(permutation);
  const existing = cache.get(transformKey);

  if (!existing || moves.length < existing.moves.length) {
    cache.set(transformKey, { permutation, moves });
  }
}

export function buildTransformationCache(maxDepth) {
  const cached = cachesByDepth.get(maxDepth);

  if (cached) {
    return { cache: cached, cacheHit: true };
  }

  const cache = new Map();
  storeTransformation(cache, IDENTITY_PERMUTATION, []);

  function visit(permutation, moves, historyFaces, remainingDepth) {
    if (remainingDepth === 0) {
      return;
    }

    for (const move of ALL_MOVES) {
      if (!isMoveAllowed(historyFaces, move)) {
        continue;
      }

      const nextPermutation = composePermutations(permutation, getMovePermutation(move));
      const nextMoves = [...moves, move];

      storeTransformation(cache, nextPermutation, nextMoves);

      visit(
        nextPermutation,
        nextMoves,
        [...historyFaces, getMoveFace(move)],
        remainingDepth - 1
      );
    }
  }

  visit(IDENTITY_PERMUTATION, [], [], maxDepth);
  cachesByDepth.set(maxDepth, cache);
  return { cache, cacheHit: false };
}
