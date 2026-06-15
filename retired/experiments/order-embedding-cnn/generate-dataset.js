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
const BFS_PAIR_TARGET_COUNT = 200_000;
const RANKING_TRIPLET_TARGET_COUNT = 50_000;
const MID_MIN = 6;
const MID_MAX = 15;
const LONG_MIN = 16;
const LONG_MAX = 40;
const LONG_LOSS_WEIGHT = 0.3;
const BFS_PAIR_GAPS = [1, 2, 3];

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
  const orderRows = [];
  const rankingRows = [];

  let stateKey = SOLVED_STATE_KEY;
  const historyFaces = [];
  const pathKeys = [SOLVED_STATE_KEY];
  const pathDepths = [0];

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

    orderRows.push({
      closerKey: stateKey,
      fartherKey: nextKey,
      gap: 1,
      labelType,
      lossWeight
    });

    pathKeys.push(nextKey);
    pathDepths.push(nextDepth);
    stateKey = nextKey;
    historyFaces.push(getMoveFace(move));
  }

  if (pathKeys.length >= 3) {
    const startIndex = randomInt(0, pathKeys.length - 3);
    rankingRows.push({
      aKey: pathKeys[startIndex],
      bKey: pathKeys[startIndex + 1],
      cKey: pathKeys[startIndex + 2],
      gapAb: pathDepths[startIndex + 1] - pathDepths[startIndex],
      gapBc: pathDepths[startIndex + 2] - pathDepths[startIndex + 1],
      labelType,
      lossWeight
    });
  }

  return { stateRows, orderRows, rankingRows };
}

function buildTrajectoryCorpus(targetCount) {
  const stateRows = [];
  const orderRows = [];
  const rankingRows = [];

  while (stateRows.length < targetCount) {
    const targetLength = randomInt(MID_MIN, LONG_MAX);
    const trajectory = buildTrajectoryRows(targetLength);
    stateRows.push(...trajectory.stateRows);
    orderRows.push(...trajectory.orderRows);
    rankingRows.push(...trajectory.rankingRows);
  }

  return {
    stateRows: stateRows.slice(0, targetCount),
    orderRows,
    rankingRows
  };
}

function buildBfsOrderPairs(bfsStates, targetCount) {
  const byDepth = new Map();

  for (const row of bfsStates) {
    const bucket = byDepth.get(row.distance) ?? [];
    bucket.push(row.stateKey);
    byDepth.set(row.distance, bucket);
  }

  const pairs = [];
  const seen = new Set();

  while (pairs.length < targetCount) {
    const gap = BFS_PAIR_GAPS[randomInt(0, BFS_PAIR_GAPS.length - 1)];
    const closerDepth = randomInt(0, BFS_MAX_DEPTH - gap);
    const fartherDepth = closerDepth + gap;
    const closerBucket = byDepth.get(closerDepth) ?? [];
    const fartherBucket = byDepth.get(fartherDepth) ?? [];

    if (closerBucket.length === 0 || fartherBucket.length === 0) {
      continue;
    }

    const closerKey = closerBucket[randomInt(0, closerBucket.length - 1)];
    const fartherKey = fartherBucket[randomInt(0, fartherBucket.length - 1)];
    const dedupeKey = `${closerKey}|${fartherKey}`;

    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    pairs.push({
      closerKey,
      fartherKey,
      gap,
      labelType: "bfs",
      lossWeight: 1.0
    });
  }

  return pairs;
}

function buildSolvedAnchoredPairs(trainingStates) {
  const seen = new Set();
  const pairs = [];

  for (const row of trainingStates) {
    if (row.stateKey === SOLVED_STATE_KEY || seen.has(row.stateKey)) {
      continue;
    }

    seen.add(row.stateKey);
    pairs.push({
      closerKey: SOLVED_STATE_KEY,
      fartherKey: row.stateKey,
      gap: Math.max(row.distance, 1),
      labelType: row.labelType,
      lossWeight: row.lossWeight
    });
  }

  return pairs;
}

function buildBfsRankingTriplets(bfsStates, targetCount) {
  const byDepth = new Map();

  for (const row of bfsStates) {
    const bucket = byDepth.get(row.distance) ?? [];
    bucket.push(row.stateKey);
    byDepth.set(row.distance, bucket);
  }

  const triplets = [];
  const seen = new Set();

  while (triplets.length < targetCount) {
    const depthA = randomInt(0, BFS_MAX_DEPTH - 2);
    const depthB = depthA + 1;
    const depthC = depthB + 1;
    const bucketA = byDepth.get(depthA) ?? [];
    const bucketB = byDepth.get(depthB) ?? [];
    const bucketC = byDepth.get(depthC) ?? [];

    if (bucketA.length === 0 || bucketB.length === 0 || bucketC.length === 0) {
      continue;
    }

    const aKey = bucketA[randomInt(0, bucketA.length - 1)];
    const bKey = bucketB[randomInt(0, bucketB.length - 1)];
    const cKey = bucketC[randomInt(0, bucketC.length - 1)];
    const dedupeKey = `${aKey}|${bKey}|${cKey}`;

    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    triplets.push({
      aKey,
      bKey,
      cKey,
      gapAb: 1,
      gapBc: 1,
      labelType: "bfs",
      lossWeight: 1.0
    });
  }

  return triplets;
}

function buildExtrapolationCorpus(targetCount) {
  const rows = [];

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

function buildStats(
  bfsStates,
  trajectoryStates,
  orderPairs,
  rankingTriplets,
  extrapolationStates
) {
  const labelTypeCounts = {};
  const orderSourceCounts = {};

  for (const row of [...bfsStates, ...trajectoryStates, ...extrapolationStates]) {
    labelTypeCounts[row.labelType] = (labelTypeCounts[row.labelType] ?? 0) + 1;
  }

  for (const row of orderPairs) {
    orderSourceCounts[row.labelType] = (orderSourceCounts[row.labelType] ?? 0) + 1;
  }

  return {
    bfsStateCount: bfsStates.length,
    trajectoryStateCount: trajectoryStates.length,
    orderPairCount: orderPairs.length,
    rankingTripletCount: rankingTriplets.length,
    extrapolationStateCount: extrapolationStates.length,
    totalStateCount: bfsStates.length + trajectoryStates.length,
    labelTypeCounts,
    orderSourceCounts,
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
const {
  stateRows: trajectoryStates,
  orderRows: trajectoryOrderPairs,
  rankingRows: trajectoryRankingTriplets
} = buildTrajectoryCorpus(TRAJECTORY_TARGET_COUNT);
console.log(`Trajectory states: ${trajectoryStates.length}`);
console.log(`Trajectory order pairs: ${trajectoryOrderPairs.length}`);

console.log("Building BFS order pairs...");
const bfsOrderPairs = buildBfsOrderPairs(bfsStates, BFS_PAIR_TARGET_COUNT);
console.log(`BFS order pairs: ${bfsOrderPairs.length}`);

const trainingStates = [...bfsStates, ...trajectoryStates];

console.log("Building solved-anchored order pairs...");
const solvedAnchoredPairs = buildSolvedAnchoredPairs(trainingStates);
console.log(`Solved-anchored order pairs: ${solvedAnchoredPairs.length}`);

const orderPairs = [...trajectoryOrderPairs, ...bfsOrderPairs, ...solvedAnchoredPairs];
console.log(`Total order pairs: ${orderPairs.length}`);

console.log("Building ranking triplets...");
const bfsRankingTriplets = buildBfsRankingTriplets(
  bfsStates,
  Math.max(RANKING_TRIPLET_TARGET_COUNT - trajectoryRankingTriplets.length, 0)
);
const rankingTriplets = [...trajectoryRankingTriplets, ...bfsRankingTriplets].slice(
  0,
  RANKING_TRIPLET_TARGET_COUNT
);
console.log(`Ranking triplets: ${rankingTriplets.length}`);

console.log("Building extrapolation eval corpus...");
const extrapolationStates = buildExtrapolationCorpus(EXTRAPOLATION_TARGET_COUNT);
console.log(`Extrapolation states: ${extrapolationStates.length}`);

const stats = buildStats(
  bfsStates,
  trajectoryStates,
  orderPairs,
  rankingTriplets,
  extrapolationStates
);

writeFileSync(
  join(dataDir, "states.jsonl"),
  `${trainingStates.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8"
);
writeFileSync(
  join(dataDir, "order_pairs.jsonl"),
  `${orderPairs.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8"
);
writeFileSync(
  join(dataDir, "ranking_triplets.jsonl"),
  `${rankingTriplets.map((row) => JSON.stringify(row)).join("\n")}\n`,
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
