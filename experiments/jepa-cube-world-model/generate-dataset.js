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
const EXAMPLE_COUNT = 10_000;
const SCRAMBLE_LENGTH = 20;
const MAX_CONTINUATION_LENGTH = 15;

const SCRAMBLE_FACES = ["F", "B", "U", "D", "R", "L"];
const SCRAMBLE_SUFFIXES = ["", "'", "2"];

function generateScramble(length = SCRAMBLE_LENGTH) {
  const moves = [];
  let previousFace = "";

  while (moves.length < length) {
    const face = SCRAMBLE_FACES[Math.floor(Math.random() * SCRAMBLE_FACES.length)];
    if (face === previousFace) {
      continue;
    }

    const suffix = SCRAMBLE_SUFFIXES[Math.floor(Math.random() * SCRAMBLE_SUFFIXES.length)];
    moves.push(`${face}${suffix}`);
    previousFace = face;
  }

  return moves;
}

function samplePrunedSequence(maxLength) {
  const targetLength = Math.floor(Math.random() * (maxLength + 1));
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

function generateExample() {
  const scramble = generateScramble();
  const startKey = applyMovesToKey(SOLVED_STATE_KEY, scramble);
  const continuation = samplePrunedSequence(MAX_CONTINUATION_LENGTH);
  const endKey = applyMovesToKey(startKey, continuation);

  return {
    startKey,
    moves: continuation,
    endKey
  };
}

function buildStats(examples) {
  const lengthCounts = {};
  const uniqueStarts = new Set();
  const uniqueEnds = new Set();

  for (const example of examples) {
    const length = example.moves.length;
    lengthCounts[length] = (lengthCounts[length] ?? 0) + 1;
    uniqueStarts.add(example.startKey);
    uniqueEnds.add(example.endKey);
  }

  return {
    exampleCount: examples.length,
    lengthCounts,
    uniqueStartStates: uniqueStarts.size,
    uniqueEndStates: uniqueEnds.size
  };
}

const dataDir = join(__dirname, "data");
mkdirSync(dataDir, { recursive: true });

const examples = Array.from({ length: EXAMPLE_COUNT }, generateExample);
const jsonl = examples.map((example) => JSON.stringify(example)).join("\n");
writeFileSync(join(dataDir, "dataset.jsonl"), `${jsonl}\n`, "utf8");
writeFileSync(join(dataDir, "stats.json"), `${JSON.stringify(buildStats(examples), null, 2)}\n`, "utf8");

console.log(`Wrote ${EXAMPLE_COUNT} examples to ${join(dataDir, "dataset.jsonl")}`);
console.log(JSON.stringify(buildStats(examples), null, 2));
