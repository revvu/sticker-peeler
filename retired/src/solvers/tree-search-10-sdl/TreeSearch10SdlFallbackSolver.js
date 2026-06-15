import { Solver } from "../Solver.js";
import { cloneCubeState, getCubeStateKey, SOLVED_STATE_KEY } from "../../cube/CubeState.js";
import {
  SDL_FALLBACK_DEPTH,
  SDL_FALLBACK_MAX_SCORES,
  TREE_SEARCH_MAX_DEPTH,
  runDepthLimitedDistanceSearch,
  runForwardSearch
} from "../utils/treeSearch.js";
import {
  buildTransformationCache,
  hasTransformationCache
} from "../utils/transformationCache.js";
import {
  initDistanceModel,
  isDistanceModelReady,
  predictDistanceToSolved
} from "../../supervised-distance/predict.js";

export class TreeSearch10SdlFallbackSolver extends Solver {
  constructor() {
    super();
    initDistanceModel();
  }

  get id() {
    return "tree-search-10-sdl";
  }

  get label() {
    return `Tree Search(${TREE_SEARCH_MAX_DEPTH})+SDL`;
  }

  get noResultLabel() {
    return "no tree solution; SDL fallback unavailable";
  }

  async step(cubeState) {
    await this.print("Tree search started...");

    const state = cloneCubeState(cubeState);
    const startStateKey = getCubeStateKey(state);

    if (startStateKey === SOLVED_STATE_KEY) {
      await this.print("Search: 0.00 ms");
      await this.print("Solution found (0.00 ms total)");
      return [];
    }

    const forwardDepth = Math.floor(TREE_SEARCH_MAX_DEPTH / 2);
    const backwardDepth = TREE_SEARCH_MAX_DEPTH - forwardDepth;

    if (!hasTransformationCache(backwardDepth)) {
      await this.print("Building cache...");
    }

    const cacheBuildStart = performance.now();
    const { cache: transformationCache, cacheHit } = buildTransformationCache(backwardDepth);
    const cacheBuildMs = performance.now() - cacheBuildStart;

    if (cacheHit) {
      await this.print(`Cache: reused (${this.formatDuration(cacheBuildMs)})`);
    } else {
      await this.print(`Cache build: ${this.formatDuration(cacheBuildMs)}`);
    }

    await this.print("Searching...");

    const forwardResult = runForwardSearch(
      startStateKey,
      transformationCache,
      forwardDepth,
      TREE_SEARCH_MAX_DEPTH
    );

    const totalMs = cacheBuildMs + forwardResult.searchMs;

    await this.print(`Search: ${this.formatDuration(forwardResult.searchMs)}`);

    if (forwardResult.moves) {
      await this.print(`Tree search solution (${this.formatDuration(totalMs)} total)`);
      return forwardResult.moves;
    }

    await this.print(`No tree solution (${this.formatDuration(totalMs)} total)`);
    await this.print(`SDL fallback: searching depth ${SDL_FALLBACK_DEPTH}...`);

    await initDistanceModel();

    if (!isDistanceModelReady()) {
      await this.print("Supervised distance model weights not loaded.");
      return null;
    }

    const fallbackResult = runDepthLimitedDistanceSearch(
      startStateKey,
      SDL_FALLBACK_DEPTH,
      predictDistanceToSolved
    );

    await this.print(
      `SDL fallback: ${this.formatDuration(fallbackResult.searchMs)}, ` +
      `${fallbackResult.searchedNodes} states scored`
    );

    if (fallbackResult.truncated) {
      await this.print(
        `SDL fallback: truncated at ${SDL_FALLBACK_MAX_SCORES} scores ` +
        `(depth ${SDL_FALLBACK_DEPTH})`
      );
    }

    if (fallbackResult.moves.length === 0) {
      await this.print(
        `SDL fallback: no improving path ` +
        `(distance ${fallbackResult.startDistance.toFixed(1)})`
      );
      return null;
    }

    await this.print(
      `SDL fallback: ${fallbackResult.moves.length} move(s) ` +
      `(${fallbackResult.startDistance.toFixed(1)} → ${fallbackResult.bestDistance.toFixed(1)})`
    );

    return fallbackResult.moves;
  }
}
