import { FACE_TURNS, OPPOSITE_FACE, getMoveFace } from "../../cube/CubeState.js";

function areParallelFaces(firstFace, secondFace) {
  return (
    firstFace !== secondFace &&
    FACE_TURNS[firstFace].axis === FACE_TURNS[secondFace].axis
  );
}

export function isMoveAllowed(historyFaces, move) {
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
