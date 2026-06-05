import { Solver } from "../Solver.js";
import { cloneCubeState } from "../../cube/CubeState.js";
import { treeSearch } from "../utils/treeSearch.js";

export class TreeSearch7Solver extends Solver {
  get id() {
    return "tree-search-7";
  }

  get label() {
    return "Tree Search(7)";
  }

  get noResultLabel() {
    return "no solution within 7 moves";
  }

  step(cubeState) {
    const searchResult = treeSearch(cloneCubeState(cubeState), 7);
    return searchResult.moves;
  }
}
