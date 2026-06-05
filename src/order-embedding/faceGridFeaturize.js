import { FACE_ORDER } from "../cube/CubeState.js";
import { stateKeyToIndices } from "../jepa/stateFeaturize.js";

const FACE_COUNT = FACE_ORDER.length;
const GRID_SIZE = 3;
const STICKER_COUNT = 54;

export function indicesToFaceGrid(indices) {
  const grid = Array.from({ length: FACE_COUNT }, () =>
    Array.from({ length: GRID_SIZE }, () => new Uint8Array(GRID_SIZE))
  );

  let offset = 0;
  for (let faceIndex = 0; faceIndex < FACE_COUNT; faceIndex += 1) {
    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let column = 0; column < GRID_SIZE; column += 1) {
        grid[faceIndex][row][column] = indices[offset];
        offset += 1;
      }
    }
  }

  return grid;
}

export function stateKeyToFaceGrid(stateKey) {
  return indicesToFaceGrid(stateKeyToIndices(stateKey));
}

export function faceGridToIndices(faceGrid) {
  const indices = new Uint8Array(STICKER_COUNT);
  let offset = 0;

  for (let faceIndex = 0; faceIndex < FACE_COUNT; faceIndex += 1) {
    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let column = 0; column < GRID_SIZE; column += 1) {
        indices[offset] = faceGrid[faceIndex][row][column];
        offset += 1;
      }
    }
  }

  return indices;
}
