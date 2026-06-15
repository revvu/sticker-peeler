import { mkdirSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  getMoveFace
} from "../../src/cube/CubeState.js";
import { isMoveAllowed } from "../../src/solvers/utils/movePruning.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PATH_LENGTH = 26;
const SAMPLES_PER_PATH = PATH_LENGTH + 1;
const TRAIN_SAMPLES = 216_000;
const VAL_SAMPLES = 21_600;
const TEST_SAMPLES = 21_600;
const SEED = 42;

function createRng(seed) {
  let state = seed >>> 0;

  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function samplePrunedMove(historyFaces, random) {
  const allowedMoves = ALL_MOVES.filter((move) => isMoveAllowed(historyFaces, move));
  if (allowedMoves.length === 0) {
    throw new Error("No legal moves available while sampling a path.");
  }

  const index = Math.floor(random() * allowedMoves.length);
  return allowedMoves[index];
}

function buildPath(pathId, random) {
  const rows = [];
  let stateKey = SOLVED_STATE_KEY;
  const historyFaces = [];

  rows.push({
    stateKey,
    distance: 0,
    pathId,
    stepIndex: 0
  });

  for (let stepIndex = 1; stepIndex <= PATH_LENGTH; stepIndex += 1) {
    const move = samplePrunedMove(historyFaces, random);
    stateKey = applyMoveToKey(stateKey, move);
    historyFaces.push(getMoveFace(move));

    rows.push({
      stateKey,
      distance: stepIndex,
      pathId,
      stepIndex
    });
  }

  return rows;
}

function buildSplit(pathCount, pathIdOffset, random) {
  const rows = [];

  for (let pathIndex = 0; pathIndex < pathCount; pathIndex += 1) {
    rows.push(...buildPath(pathIdOffset + pathIndex, random));
  }

  return rows;
}

function writeJsonl(path, rows) {
  writeFileSync(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
}

function countByDistance(rows) {
  const counts = Array.from({ length: SAMPLES_PER_PATH }, () => 0);

  for (const row of rows) {
    counts[row.distance] += 1;
  }

  return counts;
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

const trainPathCount = TRAIN_SAMPLES / SAMPLES_PER_PATH;
const valPathCount = VAL_SAMPLES / SAMPLES_PER_PATH;
const testPathCount = TEST_SAMPLES / SAMPLES_PER_PATH;

if (!Number.isInteger(trainPathCount) || !Number.isInteger(valPathCount) || !Number.isInteger(testPathCount)) {
  throw new Error("Sample counts must be divisible by 27.");
}

const trainRandom = createRng(SEED);
const valRandom = createRng(SEED + 1);
const testRandom = createRng(SEED + 2);

console.log("Generating train paths...");
const trainRows = buildSplit(trainPathCount, 0, trainRandom);
console.log("Generating validation paths...");
const valRows = buildSplit(valPathCount, trainPathCount, valRandom);
console.log("Generating test paths...");
const testRows = buildSplit(testPathCount, trainPathCount + valPathCount, testRandom);

writeJsonl(join(dataDir, "train.jsonl"), trainRows);
writeJsonl(join(dataDir, "val.jsonl"), valRows);
writeJsonl(join(dataDir, "test.jsonl"), testRows);

const stats = {
  pathLength: PATH_LENGTH,
  samplesPerPath: SAMPLES_PER_PATH,
  train: {
    paths: trainPathCount,
    samples: trainRows.length,
    distanceCounts: countByDistance(trainRows)
  },
  val: {
    paths: valPathCount,
    samples: valRows.length,
    distanceCounts: countByDistance(valRows)
  },
  test: {
    paths: testPathCount,
    samples: testRows.length,
    distanceCounts: countByDistance(testRows)
  }
};

writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");

console.log(`Wrote ${trainRows.length} train, ${valRows.length} val, ${testRows.length} test samples.`);
console.log(`Each split has ${SAMPLES_PER_PATH} distances with equal counts per distance.`);
