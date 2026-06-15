import { mkdirSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  applyMovesToKey,
  getMoveFace
} from "../../src/cube/CubeState.js";
import { isMoveAllowed } from "../../src/solvers/utils/movePruning.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const BFS_MAX_DEPTH = 5;
const TRAJECTORY_TARGET_COUNT = 150_000;
const EXTRAPOLATION_TARGET_COUNT = 5_000;
const EXTRAPOLATION_LENGTHS = [25, 30, 40];
const MID_MIN = 6;
const MID_MAX = 15;
const LONG_MIN = 16;
const LONG_MAX = 40;
const LONG_LOSS_WEIGHT = 0.3;

function randomInt(min, max) {
  return min + Math.floor(Math.random() * (max - min + 1));
}

function samplePrunedSequence(targetLength) {
  const moves = [];
  const historyFaces = [];

  while (moves.length < targetLength) {
    const move = ALL_MOVES[Math.floor(Math.random() * ALL_MOVES.length)];
    if (!isMoveAllowed(historyFaces, move)) {
      continue;
    }

    moves.push(move);
    historyFaces.push(getMoveFace(move));
  }

  return moves;
}

function buildBfsStates(maxDepth = BFS_MAX_DEPTH) {
  const states = [];
  const seen = new Set();

  function visit(stateKey, historyFaces, depth) {
    if (seen.has(stateKey)) {
      return;
    }

    seen.add(stateKey);
    states.push({
      stateKey,
      distance: depth,
      labelType: "bfs",
      lossWeight: 1.0
    });

    if (depth >= maxDepth) {
      return;
    }

    for (const move of ALL_MOVES) {
      if (!isMoveAllowed(historyFaces, move)) {
        continue;
      }

      visit(
        applyMoveToKey(stateKey, move),
        [...historyFaces, getMoveFace(move)],
        depth + 1
      );
    }
  }

  visit(SOLVED_STATE_KEY, [], 0);
  return states;
}

function buildTrajectoryRows(targetLength) {
  const scramble = samplePrunedSequence(targetLength);
  const lossWeight = targetLength >= LONG_MIN ? LONG_LOSS_WEIGHT : 1.0;
  const labelType = targetLength >= LONG_MIN ? "scramble_long" : "scramble_mid";
  const stateRows = [];
  const ordinalRows = [];

  let stateKey = SOLVED_STATE_KEY;
  const historyFaces = [];

  for (let depth = 0; depth < scramble.length; depth += 1) {
    const move = scramble[depth];
    const nextKey = applyMoveToKey(stateKey, move);
    const nextDepth = depth + 1;

    stateRows.push({
      stateKey: nextKey,
      distance: nextDepth,
      labelType,
      lossWeight
    });

    ordinalRows.push({
      earlierKey: stateKey,
      laterKey: nextKey,
      gap: 1,
      labelType,
      lossWeight
    });

    stateKey = nextKey;
    historyFaces.push(getMoveFace(move));
  }

  return { stateRows, ordinalRows };
}

function buildTrajectoryCorpus(targetCount) {
  const stateRows = [];
  const ordinalRows = [];

  while (stateRows.length < targetCount) {
    const targetLength = randomInt(MID_MIN, LONG_MAX);
    const trajectory = buildTrajectoryRows(targetLength);
    stateRows.push(...trajectory.stateRows);
    ordinalRows.push(...trajectory.ordinalRows);
  }

  return {
    stateRows: stateRows.slice(0, targetCount),
    ordinalRows
  };
}

function buildExtrapolationCorpus(targetCount, seedRows = []) {
  const rows = [...seedRows];

  while (rows.length < targetCount) {
    const targetLength = EXTRAPOLATION_LENGTHS[randomInt(0, EXTRAPOLATION_LENGTHS.length - 1)];
    const scramble = samplePrunedSequence(targetLength);
    const stateKey = applyMovesToKey(SOLVED_STATE_KEY, scramble);

    rows.push({
      stateKey,
      distance: targetLength,
      labelType: "extrapolation",
      lossWeight: 0.0
    });
  }

  return rows.slice(0, targetCount);
}

function buildStats(bfsStates, trajectoryStates, ordinalPairs, extrapolationStates) {
  const labelTypeCounts = {};
  const distanceCounts = {};

  for (const row of [...bfsStates, ...trajectoryStates, ...extrapolationStates]) {
    labelTypeCounts[row.labelType] = (labelTypeCounts[row.labelType] ?? 0) + 1;
    distanceCounts[row.distance] = (distanceCounts[row.distance] ?? 0) + 1;
  }

  return {
    bfsStateCount: bfsStates.length,
    trajectoryStateCount: trajectoryStates.length,
    ordinalPairCount: ordinalPairs.length,
    extrapolationStateCount: extrapolationStates.length,
    totalStateCount: bfsStates.length + trajectoryStates.length,
    labelTypeCounts,
    distanceCounts,
    bfsMaxDepth: BFS_MAX_DEPTH,
    trajectoryDepthRange: [MID_MIN, LONG_MAX],
    longLossWeight: LONG_LOSS_WEIGHT
  };
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

console.log("Building exact BFS corpus (depth 0-5)...");
const bfsStates = buildBfsStates();
console.log(`BFS states: ${bfsStates.length}`);

console.log("Building trajectory scramble corpus...");
const { stateRows: trajectoryStates, ordinalRows } = buildTrajectoryCorpus(TRAJECTORY_TARGET_COUNT);
console.log(`Trajectory states: ${trajectoryStates.length}`);
console.log(`Ordinal pairs: ${ordinalRows.length}`);

console.log("Building extrapolation eval corpus...");
const extrapolationStates = buildExtrapolationCorpus(EXTRAPOLATION_TARGET_COUNT);
console.log(`Extrapolation states: ${extrapolationStates.length}`);

const stats = buildStats(bfsStates, trajectoryStates, ordinalRows, extrapolationStates);
const trainingStates = [...bfsStates, ...trajectoryStates];

writeFileSync(
  join(dataDir, "states.jsonl"),
  `${trainingStates.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8"
);
writeFileSync(
  join(dataDir, "ordinal_pairs.jsonl"),
  `${ordinalRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8"
);
writeFileSync(
  join(dataDir, "extrapolation.jsonl"),
  `${extrapolationStates.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8"
);
writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");

console.log(`Wrote ${trainingStates.length} training states`);
console.log(JSON.stringify(stats, null, 2));
