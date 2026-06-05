import { TreeSearch10Solver } from "./tree-search-10/TreeSearch10Solver.js";
import { TripletLossSolver } from "./triplet-loss/TripletLossSolver.js";

export const availableSolvers = [
  new TreeSearch10Solver(),
  new TripletLossSolver()
];
