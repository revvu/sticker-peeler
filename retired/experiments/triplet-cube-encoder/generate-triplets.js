import { mkdirSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMovesToKey,
  getMoveFace
} from "../../src/cube/CubeState.js";
import { isMoveAllowed } from "../../src/solvers/utils/movePruning.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const TARGET_TRIPLET_COUNT = 200_000;
const SEQUENCE_MIN = 0;
const SEQUENCE_MAX = 15;
const MAX_PAIR_ATTEMPTS = 50;

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

function buildTriplet() {
  for (let attempt = 0; attempt < MAX_PAIR_ATTEMPTS; attempt += 1) {
    const seqA = samplePrunedSequence(randomInt(SEQUENCE_MIN, SEQUENCE_MAX));
    const seqB = samplePrunedSequence(randomInt(SEQUENCE_MIN, SEQUENCE_MAX));
    const distA = seqA.length;
    const distB = seqB.length;

    if (distA === distB) {
      continue;
    }

    const stateA = applyMovesToKey(SOLVED_STATE_KEY, seqA);
    const stateB = applyMovesToKey(SOLVED_STATE_KEY, seqB);
    const positiveKey = distA < distB ? stateA : stateB;
    const negativeKey = distA < distB ? stateB : stateA;
    const positiveDist = Math.min(distA, distB);
    const negativeDist = Math.max(distA, distB);

    return {
      anchorKey: SOLVED_STATE_KEY,
      positiveKey,
      negativeKey,
      positiveDist,
      negativeDist,
      labelSource: "scramble_depth"
    };
  }

  return null;
}

function buildTriplets() {
  const triplets = [];
  let failedAttempts = 0;

  while (triplets.length < TARGET_TRIPLET_COUNT) {
    const triplet = buildTriplet();

    if (!triplet) {
      failedAttempts += 1;
      if (failedAttempts > TARGET_TRIPLET_COUNT) {
        throw new Error("Failed to generate enough solved-relative triplets");
      }
      continue;
    }

    triplets.push(triplet);
    failedAttempts = 0;
  }

  return triplets;
}

function buildStats(triplets) {
  const distanceDiffCounts = {};

  for (const triplet of triplets) {
    const diff = triplet.negativeDist - triplet.positiveDist;
    distanceDiffCounts[diff] = (distanceDiffCounts[diff] ?? 0) + 1;
  }

  return {
    tripletCount: triplets.length,
    uniqueAnchors: new Set(triplets.map((triplet) => triplet.anchorKey)).size,
    includesSolvedAnchor: triplets.every((triplet) => triplet.anchorKey === SOLVED_STATE_KEY),
    sequenceRange: [SEQUENCE_MIN, SEQUENCE_MAX],
    labelSource: "scramble_depth_from_solved",
    distanceDiffCounts
  };
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

console.log("Generating solved-relative triplets (scramble-depth labels)...");
const triplets = buildTriplets();
const stats = buildStats(triplets);

writeFileSync(
  join(dataDir, "triplets.jsonl"),
  `${triplets.map((triplet) => JSON.stringify(triplet)).join("\n")}\n`,
  "utf8"
);
writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");

console.log(`Wrote ${triplets.length} solved-relative triplets`);
console.log(JSON.stringify(stats, null, 2));
