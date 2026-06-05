import {
  ALL_MOVES,
  FACE_TURNS,
  OPPOSITE_FACE,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  getCubeStateKey,
  getMoveFace,
  invertMoves
} from "../../cube/CubeState.js";

export const TREE_SEARCH_MAX_DEPTH = 10;

const solvedIndexesByDepth = new Map();

function areParallelFaces(firstFace, secondFace) {
  return (
    firstFace !== secondFace &&
    FACE_TURNS[firstFace].axis === FACE_TURNS[secondFace].axis
  );
}

function isMoveAllowed(historyFaces, move) {
  const face = getMoveFace(move);
  const previousFace = historyFaces[historyFaces.length - 1];

  if (face === previousFace) {
    return false;
  }

  if (
    previousFace &&
    areParallelFaces(face, previousFace) &&
    face < previousFace
  ) {
    return false;
  }

  const faceBeforePrevious = historyFaces[historyFaces.length - 2];

  if (
    faceBeforePrevious &&
    OPPOSITE_FACE[faceBeforePrevious] === previousFace &&
    (face === faceBeforePrevious || face === previousFace)
  ) {
    return false;
  }

  return true;
}

function buildSolvedIndex(maxDepth) {
  const cachedIndex = solvedIndexesByDepth.get(maxDepth);

  if (cachedIndex) {
    return cachedIndex;
  }

  const index = new Map([[SOLVED_STATE_KEY, []]]);

  function visit(stateKey, moves, historyFaces, remainingDepth) {
    if (remainingDepth === 0) {
      return;
    }

    for (const move of ALL_MOVES) {
      if (!isMoveAllowed(historyFaces, move)) {
        continue;
      }

      const nextStateKey = applyMoveToKey(stateKey, move);
      const nextMoves = [...moves, move];

      if (!index.has(nextStateKey)) {
        index.set(nextStateKey, nextMoves);
      }

      visit(nextStateKey, nextMoves, [...historyFaces, getMoveFace(move)], remainingDepth - 1);
    }
  }

  visit(SOLVED_STATE_KEY, [], [], maxDepth);
  solvedIndexesByDepth.set(maxDepth, index);
  return index;
}

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
  const solvedIndex = buildSolvedIndex(backwardDepth);
  let searchedNodes = 1;
  let solutionMoves = null;

  function tryMatch(stateKey, prefixMoves) {
    const solvedSideMoves = solvedIndex.get(stateKey);

    if (!solvedSideMoves) {
      return false;
    }

    const moves = [...prefixMoves, ...invertMoves(solvedSideMoves)];

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

      if (visit(nextStateKey, [...moves, move], [...historyFaces, getMoveFace(move)], remainingDepth - 1)) {
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
