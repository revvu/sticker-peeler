import { SupervisedDistanceSolver } from "./supervised-distance/SupervisedDistanceSolver.js";
import { TreeSearch10Solver } from "./tree-search-10/TreeSearch10Solver.js";
import { TreeSearch10SdlFallbackSolver } from "./tree-search-10-sdl/TreeSearch10SdlFallbackSolver.js";
import { TripletLossSolver } from "./triplet-loss/TripletLossSolver.js";

export const availableSolvers = [
  new TreeSearch10Solver(),
  new TreeSearch10SdlFallbackSolver(),
  new TripletLossSolver(),
  new SupervisedDistanceSolver()
];
