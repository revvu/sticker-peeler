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
const SCRAMBLE_MIN = 0;
const SCRAMBLE_MAX = 15;
const SEQUENCE_MIN = 0;
const SEQUENCE_MAX = 15;
const TRIPLETS_MIN = 7;
const TRIPLETS_MAX = 8;
const INCLUDE_SOLVED_ANCHOR = true;
const MAX_ANCHOR_ATTEMPTS = 5_000_000;

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

function sampleShallowScramble() {
  return samplePrunedSequence(randomInt(SCRAMBLE_MIN, SCRAMBLE_MAX));
}

function collectAnchors() {
  const anchors = new Set();

  if (INCLUDE_SOLVED_ANCHOR) {
    anchors.add(SOLVED_STATE_KEY);
  }

  let attempts = 0;
  while (attempts < MAX_ANCHOR_ATTEMPTS) {
    attempts += 1;
    const scramble = sampleShallowScramble();
    anchors.add(applyMovesToKey(SOLVED_STATE_KEY, scramble));

    const averageTriplets = (TRIPLETS_MIN + TRIPLETS_MAX) / 2;
    if (anchors.size * averageTriplets >= TARGET_TRIPLET_COUNT) {
      break;
    }
  }

  if (attempts >= MAX_ANCHOR_ATTEMPTS) {
    throw new Error(`Failed to collect enough anchors after ${MAX_ANCHOR_ATTEMPTS} attempts`);
  }

  return [...anchors];
}

function buildTriplet(anchorKey) {
  const seqA = samplePrunedSequence(randomInt(SEQUENCE_MIN, SEQUENCE_MAX));
  const seqB = samplePrunedSequence(randomInt(SEQUENCE_MIN, SEQUENCE_MAX));
  const lenA = seqA.length;
  const lenB = seqB.length;

  let positiveSeq;
  let negativeSeq;
  let tieBreak = "length";

  if (lenA < lenB) {
    positiveSeq = seqA;
    negativeSeq = seqB;
  } else if (lenB < lenA) {
    positiveSeq = seqB;
    negativeSeq = seqA;
  } else if (Math.random() < 0.5) {
    positiveSeq = seqA;
    negativeSeq = seqB;
    tieBreak = "random";
  } else {
    positiveSeq = seqB;
    negativeSeq = seqA;
    tieBreak = "random";
  }

  return {
    anchorKey,
    positiveKey: applyMovesToKey(anchorKey, positiveSeq),
    negativeKey: applyMovesToKey(anchorKey, negativeSeq),
    positiveLen: positiveSeq.length,
    negativeLen: negativeSeq.length,
    tieBreak
  };
}

function buildTriplets(anchors) {
  const triplets = [];

  for (const anchorKey of anchors) {
    const tripletCount = randomInt(TRIPLETS_MIN, TRIPLETS_MAX);

    for (let index = 0; index < tripletCount; index += 1) {
      triplets.push(buildTriplet(anchorKey));

      if (triplets.length >= TARGET_TRIPLET_COUNT) {
        return triplets;
      }
    }
  }

  return triplets;
}

function buildStats(triplets) {
  const uniqueAnchors = new Set();
  const lengthDiffCounts = {};
  const tieBreakCounts = { length: 0, random: 0 };
  let includesSolvedAnchor = false;

  for (const triplet of triplets) {
    uniqueAnchors.add(triplet.anchorKey);
    const diff = Math.abs(triplet.positiveLen - triplet.negativeLen);
    lengthDiffCounts[diff] = (lengthDiffCounts[diff] ?? 0) + 1;
    tieBreakCounts[triplet.tieBreak] += 1;

    if (triplet.anchorKey === SOLVED_STATE_KEY) {
      includesSolvedAnchor = true;
    }
  }

  return {
    tripletCount: triplets.length,
    uniqueAnchors: uniqueAnchors.size,
    includesSolvedAnchor,
    scrambleRange: [SCRAMBLE_MIN, SCRAMBLE_MAX],
    sequenceRange: [SEQUENCE_MIN, SEQUENCE_MAX],
    tripletsPerAnchor: [TRIPLETS_MIN, TRIPLETS_MAX],
    lengthDiffCounts,
    tieBreakCounts
  };
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

const anchors = collectAnchors();
const triplets = buildTriplets(anchors);
const stats = buildStats(triplets);

writeFileSync(
  join(dataDir, "triplets.jsonl"),
  `${triplets.map((triplet) => JSON.stringify(triplet)).join("\n")}\n`,
  "utf8"
);
writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");

console.log(`Wrote ${triplets.length} triplets from ${anchors.length} anchors`);
console.log(JSON.stringify(stats, null, 2));
