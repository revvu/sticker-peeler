export class Solver {
  get id() {
    throw new Error("Solver subclasses must define an id.");
  }

  get label() {
    return this.id;
  }

  get noResultLabel() {
    return "no solution found";
  }

  step() {
    throw new Error("Solver subclasses must implement step(cubeState).");
  }
}
