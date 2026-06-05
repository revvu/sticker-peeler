import { FACE_ORDER } from "../cube/CubeState.js";

const FACE_TO_INDEX = Object.fromEntries(FACE_ORDER.map((face, index) => [face, index]));
const STICKER_COUNT = 54;

export function stateKeyToIndices(stateKey) {
  if (stateKey.length !== STICKER_COUNT) {
    throw new Error(`Expected state key length ${STICKER_COUNT}, got ${stateKey.length}`);
  }

  const indices = new Uint8Array(STICKER_COUNT);
  for (let index = 0; index < STICKER_COUNT; index += 1) {
    const faceIndex = FACE_TO_INDEX[stateKey[index]];
    if (faceIndex === undefined) {
      throw new Error(`Unknown face letter "${stateKey[index]}" at index ${index}`);
    }
    indices[index] = faceIndex;
  }

  return indices;
}
