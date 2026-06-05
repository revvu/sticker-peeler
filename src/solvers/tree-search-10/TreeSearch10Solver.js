import { Solver } from "../Solver.js";
import { cloneCubeState, getCubeStateKey, SOLVED_STATE_KEY } from "../../cube/CubeState.js";
import { TREE_SEARCH_MAX_DEPTH, runForwardSearch } from "../utils/treeSearch.js";
import {
  buildTransformationCache,
  hasTransformationCache
} from "../utils/transformationCache.js";

export class TreeSearch10Solver extends Solver {
  get id() {
    return "tree-search-10";
  }

  get label() {
    return `Tree Search(${TREE_SEARCH_MAX_DEPTH})`;
  }

  get noResultLabel() {
    return `no solution within ${TREE_SEARCH_MAX_DEPTH} moves`;
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
      await this.print(`Solution found (${this.formatDuration(totalMs)} total)`);
    } else {
      await this.print(`No solution found (${this.formatDuration(totalMs)} total)`);
    }

    return forwardResult.moves;
  }
}
