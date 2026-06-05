import { yieldToBrowser } from "./utils/yieldToBrowser.js";

export class Solver {
  constructor({ consoleSink = null } = {}) {
    this.consoleSink = consoleSink;
    this.timers = new Map();
  }

  get id() {
    throw new Error("Solver subclasses must define an id.");
  }

  get label() {
    return this.id;
  }

  get noResultLabel() {
    return "no solution found";
  }

  setConsoleSink(consoleSink) {
    this.consoleSink = consoleSink;
  }

  async print(message) {
    const text = String(message);

    if (this.consoleSink) {
      this.consoleSink(text, this);
      await yieldToBrowser();
      return;
    }

    console.log(text);
  }

  startTimer(label) {
    this.timers.set(label, performance.now());
  }

  endTimer(label) {
    const startedAt = this.timers.get(label);
    const elapsed = startedAt === undefined ? 0 : performance.now() - startedAt;
    this.timers.delete(label);
    return elapsed;
  }

  formatDuration(milliseconds) {
    return `${milliseconds.toFixed(2)} ms`;
  }

  step() {
    throw new Error("Solver subclasses must implement step(cubeState).");
  }
}
