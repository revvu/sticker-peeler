import { Solver } from "../Solver.js";
import { cloneCubeState } from "../../cube/CubeState.js";
import { TREE_SEARCH_MAX_DEPTH, treeSearch } from "../utils/treeSearch.js";

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

  step(cubeState) {
    const timerLabel = this.id;
    this.print("Tree search started...");
    this.startTimer(timerLabel);
    const searchResult = treeSearch(cloneCubeState(cubeState), TREE_SEARCH_MAX_DEPTH);
    const duration = this.formatDuration(this.endTimer(timerLabel));

    if (searchResult.moves) {
      this.print(`Solution found in ${duration}`);
    } else {
      this.print(`No solution found in ${duration}`);
    }

    return searchResult.moves;
  }
}
