import {
  ALL_MOVES,
  FACE_TURNS,
  OPPOSITE_FACE,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  getMoveFace
} from "../../src/cube/CubeState.js";

const DEPTH = 5;

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
