const AXES = ["x", "y", "z"];
const AXIS_INDEX = { x: 0, y: 1, z: 2 };

export const FACE_TURNS = {
  F: { axis: "z", layer: 1, direction: -1, normal: [0, 0, 1] },
  B: { axis: "z", layer: -1, direction: 1, normal: [0, 0, -1] },
  U: { axis: "y", layer: 1, direction: -1, normal: [0, 1, 0] },
  D: { axis: "y", layer: -1, direction: 1, normal: [0, -1, 0] },
  R: { axis: "x", layer: 1, direction: -1, normal: [1, 0, 0] },
  L: { axis: "x", layer: -1, direction: 1, normal: [-1, 0, 0] }
};

export const FACE_ORDER = ["F", "B", "U", "D", "R", "L"];
export const MOVE_SUFFIXES = ["", "'", "2"];
export const ALL_MOVES = FACE_ORDER.flatMap((face) => MOVE_SUFFIXES.map((suffix) => `${face}${suffix}`));
export const OPPOSITE_FACE = { F: "B", B: "F", U: "D", D: "U", R: "L", L: "R" };

const STICKER_POSITIONS = [];
const STICKER_INDEX_BY_KEY = new Map();

function stickerKey(position, normal) {
  return `${position.join(",")}|${normal.join(",")}`;
}

function addStickerPosition(face, position) {
  const normal = FACE_TURNS[face].normal;
  const key = stickerKey(position, normal);
  STICKER_INDEX_BY_KEY.set(key, STICKER_POSITIONS.length);
  STICKER_POSITIONS.push({ face, position, normal });
}

for (const face of FACE_ORDER) {
  const { axis, layer } = FACE_TURNS[face];
  const freeAxes = AXES.filter((candidate) => candidate !== axis);

  for (let first = -1; first <= 1; first += 1) {
    for (let second = -1; second <= 1; second += 1) {
      const position = [0, 0, 0];
      position[AXIS_INDEX[axis]] = layer;
      position[AXIS_INDEX[freeAxes[0]]] = first;
      position[AXIS_INDEX[freeAxes[1]]] = second;
      addStickerPosition(face, position);
    }
  }
}

export const SOLVED_STATE_KEY = STICKER_POSITIONS.map((sticker) => sticker.face).join("");

function rotateVector(vector, axis, direction) {
  const [x, y, z] = vector;

  if (axis === "x") {
    return [x, -direction * z, direction * y];
  }

  if (axis === "y") {
    return [direction * z, y, -direction * x];
  }

  return [-direction * y, direction * x, z];
}

function buildQuarterPermutation(face, direction) {
  const { axis, layer } = FACE_TURNS[face];
  const axisIndex = AXIS_INDEX[axis];

  return STICKER_POSITIONS.map((sticker, index) => {
    if (sticker.position[axisIndex] !== layer) {
      return index;
    }

    const nextPosition = rotateVector(sticker.position, axis, direction);
    const nextNormal = rotateVector(sticker.normal, axis, direction);
    const nextIndex = STICKER_INDEX_BY_KEY.get(stickerKey(nextPosition, nextNormal));

    if (nextIndex === undefined) {
      throw new Error(`Unable to map ${face} turn sticker position.`);
    }

    return nextIndex;
  });
}

export const IDENTITY_PERMUTATION = STICKER_POSITIONS.map((_, index) => index);

export function composePermutations(firstPermutation, secondPermutation) {
  return firstPermutation.map((nextIndex) => secondPermutation[nextIndex]);
}

const MOVE_PERMUTATIONS = new Map();

for (const face of FACE_ORDER) {
  const clockwise = buildQuarterPermutation(face, FACE_TURNS[face].direction);
  const counterclockwise = buildQuarterPermutation(face, -FACE_TURNS[face].direction);
  const halfTurn = composePermutations(clockwise, clockwise);

  MOVE_PERMUTATIONS.set(face, clockwise);
  MOVE_PERMUTATIONS.set(`${face}'`, counterclockwise);
  MOVE_PERMUTATIONS.set(`${face}2`, halfTurn);
}

export class CubeState {
  constructor(stateKey = SOLVED_STATE_KEY) {
    this.key = getCubeStateKey(stateKey);
  }

  clone() {
    return new CubeState(this.key);
  }

  applyMove(move) {
    this.key = applyMoveToKey(this.key, move);
    return this;
  }

  applyMoves(moves) {
    for (const move of moves) {
      this.applyMove(move);
    }

    return this;
  }

  isSolved() {
    return this.key === SOLVED_STATE_KEY;
  }

  toString() {
    return this.key;
  }
}

export function createSolvedCubeState() {
  return new CubeState();
}

export function cloneCubeState(cubeState) {
  return new CubeState(getCubeStateKey(cubeState));
}

export function getCubeStateKey(cubeState) {
  if (cubeState instanceof CubeState) {
    return cubeState.key;
  }

  if (typeof cubeState === "string") {
    if (cubeState.length !== SOLVED_STATE_KEY.length) {
      throw new Error("Cube state key has an invalid length.");
    }

    return cubeState;
  }

  if (cubeState && typeof cubeState.key === "string") {
    return cubeState.key;
  }

  throw new Error("Expected a CubeState or state key.");
}

export function normalizeMove(move) {
  const notation = String(move).trim();
  const face = notation[0]?.toUpperCase();
  const suffix = notation.slice(1);
  const normalizedMove = `${face ?? ""}${suffix}`;

  if (!FACE_TURNS[face] || !MOVE_SUFFIXES.includes(suffix) || !MOVE_PERMUTATIONS.has(normalizedMove)) {
    throw new Error(`Invalid move notation: ${move}`);
  }

  return normalizedMove;
}

export function getMoveFace(move) {
  return normalizeMove(move)[0];
}

export function invertMove(move) {
  const normalizedMove = normalizeMove(move);
  const face = normalizedMove[0];
  const suffix = normalizedMove.slice(1);

  if (suffix === "'") {
    return face;
  }

  if (suffix === "2") {
    return normalizedMove;
  }

  return `${face}'`;
}

export function invertMoves(moves) {
  return [...moves].reverse().map(invertMove);
}

export function getMovePermutation(move) {
  return MOVE_PERMUTATIONS.get(normalizeMove(move));
}

export function applyPermutationToKey(stateKey, permutation) {
  const key = getCubeStateKey(stateKey);
  const nextState = new Array(key.length);

  for (let index = 0; index < key.length; index += 1) {
    nextState[permutation[index]] = key[index];
  }

  return nextState.join("");
}

export function getTransformationKey(permutation) {
  return applyPermutationToKey(SOLVED_STATE_KEY, permutation);
}

export function applyMoveToKey(stateKey, move) {
  return applyPermutationToKey(stateKey, getMovePermutation(move));
}

export function applyMovesToKey(stateKey, moves) {
  let nextKey = getCubeStateKey(stateKey);

  for (const move of moves) {
    nextKey = applyMoveToKey(nextKey, move);
  }

  return nextKey;
}

export function isSolved(cubeState) {
  return getCubeStateKey(cubeState) === SOLVED_STATE_KEY;
}

export function getTurnCommandsForNotation(move, duration) {
  const normalizedMove = normalizeMove(move);
  const face = normalizedMove[0];
  const suffix = normalizedMove.slice(1);
  const turn = FACE_TURNS[face];
  const turnCount = suffix === "2" ? 2 : 1;
  const direction = suffix === "'" ? -turn.direction : turn.direction;

  return Array.from({ length: turnCount }, () => ({
    axis: turn.axis,
    layer: turn.layer,
    direction,
    duration
  }));
}
