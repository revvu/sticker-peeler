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

const TARGET_EXAMPLE_COUNT = 200_000;
const SCRAMBLE_MIN = 0;
const SCRAMBLE_MAX = 15;
const CONTINUATIONS_MIN = 5;
const CONTINUATIONS_MAX = 10;
const MAX_CONTINUATION_LENGTH = 20;
const INCLUDE_SOLVED_START = true;
const MAX_SCRAMBLE_ATTEMPTS = 5_000_000;

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
  const length = randomInt(SCRAMBLE_MIN, SCRAMBLE_MAX);
  return samplePrunedSequence(length);
}

function collectStartKeys() {
  const startKeys = new Set();

  if (INCLUDE_SOLVED_START) {
    startKeys.add(SOLVED_STATE_KEY);
  }

  let attempts = 0;
  while (attempts < MAX_SCRAMBLE_ATTEMPTS) {
    attempts += 1;
    const scramble = sampleShallowScramble();
    const startKey = applyMovesToKey(SOLVED_STATE_KEY, scramble);
    startKeys.add(startKey);

    const averageContinuations = (CONTINUATIONS_MIN + CONTINUATIONS_MAX) / 2;
    const projectedRows = startKeys.size * averageContinuations;
    if (projectedRows >= TARGET_EXAMPLE_COUNT) {
      break;
    }
  }

  if (attempts >= MAX_SCRAMBLE_ATTEMPTS) {
    throw new Error(`Failed to collect enough shallow start keys after ${MAX_SCRAMBLE_ATTEMPTS} attempts`);
  }

  return [...startKeys];
}

function buildExamples(startKeys) {
  const examples = [];

  for (const startKey of startKeys) {
    const continuationCount = randomInt(CONTINUATIONS_MIN, CONTINUATIONS_MAX);

    for (let index = 0; index < continuationCount; index += 1) {
      const continuationLength = randomInt(0, MAX_CONTINUATION_LENGTH);
      const moves = samplePrunedSequence(continuationLength);
      examples.push({
        startKey,
        moves,
        endKey: applyMovesToKey(startKey, moves)
      });

      if (examples.length >= TARGET_EXAMPLE_COUNT) {
        return examples;
      }
    }
  }

  return examples;
}

function buildStats(examples) {
  const lengthCounts = {};
  const uniqueStarts = new Set();
  const uniqueEnds = new Set();
  let includesSolvedStart = false;

  for (const example of examples) {
    const length = example.moves.length;
    lengthCounts[length] = (lengthCounts[length] ?? 0) + 1;
    uniqueStarts.add(example.startKey);
    uniqueEnds.add(example.endKey);
    if (example.startKey === SOLVED_STATE_KEY) {
      includesSolvedStart = true;
    }
  }

  return {
    exampleCount: examples.length,
    uniqueStartStates: uniqueStarts.size,
    uniqueEndStates: uniqueEnds.size,
    includesSolvedStart,
    scrambleRange: [SCRAMBLE_MIN, SCRAMBLE_MAX],
    continuationsPerStart: [CONTINUATIONS_MIN, CONTINUATIONS_MAX],
    maxContinuationLength: MAX_CONTINUATION_LENGTH,
    lengthCounts
  };
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

const startKeys = collectStartKeys();
const examples = buildExamples(startKeys);
const stats = buildStats(examples);

writeFileSync(join(dataDir, "dataset.jsonl"), `${examples.map((example) => JSON.stringify(example)).join("\n")}\n`, "utf8");
writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");

console.log(`Wrote ${examples.length} examples from ${startKeys.length} shallow start states`);
console.log(JSON.stringify(stats, null, 2));
