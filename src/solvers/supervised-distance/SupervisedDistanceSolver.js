import {
  ALL_MOVES,
  SOLVED_STATE_KEY,
  applyMoveToKey,
  cloneCubeState,
  getCubeStateKey
} from "../../cube/CubeState.js";
import {
  initDistanceModel,
  isDistanceModelReady,
  predictDistanceToSolved
} from "../../supervised-distance/predict.js";
import { Solver } from "../Solver.js";
import { isMoveAllowed } from "../utils/movePruning.js";

function getPrunedMoves(historyFaces = []) {
  return ALL_MOVES.filter((move) => isMoveAllowed(historyFaces, move));
}

export class SupervisedDistanceSolver extends Solver {
  constructor() {
    super();
    this.recentStateKeys = new Set();
    initDistanceModel();
  }

  get id() {
    return "supervised-distance";
  }

  get label() {
    return "Supervised Distance";
  }

  get noResultLabel() {
    return "distance model unavailable";
  }

  async step(cubeState) {
    await initDistanceModel();

    if (!isDistanceModelReady()) {
      await this.print("Supervised distance model weights not loaded.");
      return null;
    }

    const state = cloneCubeState(cubeState);
    const startStateKey = getCubeStateKey(state);

    if (startStateKey === SOLVED_STATE_KEY) {
      await this.print("Cube already solved.");
      return [];
    }

    const currentDistance = predictDistanceToSolved(startStateKey);
    const candidates = getPrunedMoves().map((move) => {
      const nextKey = applyMoveToKey(startStateKey, move);
      return {
        move,
        nextKey,
        distance: predictDistanceToSolved(nextKey)
      };
    });

    candidates.sort((left, right) => left.distance - right.distance);

    let chosen = candidates.find((candidate) => !this.recentStateKeys.has(candidate.nextKey));
    if (!chosen) {
      chosen = candidates[0];
    }

    this.recentStateKeys.add(startStateKey);
    if (this.recentStateKeys.size > 12) {
      this.recentStateKeys.delete([...this.recentStateKeys][0]);
    }

    await this.print(
      `Distance step: ${chosen.move} ` +
      `(${currentDistance.toFixed(1)} → ${chosen.distance.toFixed(1)})`
    );

    return [chosen.move];
  }
}
