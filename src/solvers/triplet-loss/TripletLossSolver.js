import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  cloneCubeState,
  getCubeStateKey
} from "../../cube/CubeState.js";
import { getSimilarityToSolved, initTripletEncoder, isTripletEncoderReady } from "../../triplet/similarity.js";
import { Solver } from "../Solver.js";
import { isMoveAllowed } from "../utils/movePruning.js";

function getPrunedMoves(historyFaces = []) {
  return ALL_MOVES.filter((move) => isMoveAllowed(historyFaces, move));
}

export class TripletLossSolver extends Solver {
  constructor() {
    super();
    this.recentStateKeys = new Set();
    initTripletEncoder();
  }

  get id() {
    return "triplet-loss";
  }

  get label() {
    return "Triplet Loss";
  }

  get noResultLabel() {
    return "triplet encoder unavailable";
  }

  async step(cubeState) {
    await initTripletEncoder();

    if (!isTripletEncoderReady()) {
      await this.print("Triplet encoder weights not loaded.");
      return null;
    }

    const state = cloneCubeState(cubeState);
    const startStateKey = getCubeStateKey(state);

    if (startStateKey === SOLVED_STATE_KEY) {
      await this.print("Cube already solved.");
      return [];
    }

    const currentSimilarity = getSimilarityToSolved(startStateKey);
    const candidates = getPrunedMoves().map((move) => {
      const nextKey = applyMoveToKey(startStateKey, move);
      return {
        move,
        nextKey,
        similarity: getSimilarityToSolved(nextKey)
      };
    });

    candidates.sort((left, right) => right.similarity - left.similarity);

    let chosen = candidates.find((candidate) => !this.recentStateKeys.has(candidate.nextKey));
    if (!chosen) {
      chosen = candidates[0];
    }

    this.recentStateKeys.add(startStateKey);
    if (this.recentStateKeys.size > 12) {
      this.recentStateKeys.delete([...this.recentStateKeys][0]);
    }

    await this.print(
      `Triplet step: ${chosen.move} ` +
      `(similarity ${currentSimilarity.toFixed(3)} → ${chosen.similarity.toFixed(3)})`
    );

    return [chosen.move];
  }
}
