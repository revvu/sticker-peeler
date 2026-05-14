# Challenges
1. Given *any* initial scramble of the Rubik's Cube, find a solution within a reasonable amount of time (< 5 min).
2. Given *any* initial scramble of the Rubik's Cube, find a solution within the WR time (< 3 sec).
3. Given *any* initial scramble of the Rubik's Cube, find the **shortest** solution within a reasonable amount of time (< 5 min).
4. Given *any* initial scramble of the Rubik's Cube, find the **shortest** solution within the WR time (< 3 sec).
5. Implement a physical solver that takes in an input scramble, finds the solution using one of the previous methods, and converts that solution into physical movements.
6. Implement a physical solver that takes in an input scramble *or* a pre-scrambled cube, correctly processes the cube state, and finds the solution using one of the previous methods.

# Sub-Tasks
- Create an emulator
- Find a way to represent cube states and transformations
- Create a value network that scores different cube states according to proximity (move-wise) to the solved state
- Implement a traversal algorithm based on the value network

# Interesting Questions
### Representation
- What is the most information-dense way to represent a cube state? By how much does this "optimal compression" reduce the overall search space?
- What does the mapping look like to go from a compressed state to its corresponding fastest solution?

### HCI
- What patterns (if any) do cube states that are "close to solved" share? Are any of these patterns human-interpretable? Are there unique abstractions that help us understand the cube in a computer-friendly way?
- What might "Beginner's Method" look like for a computer? In other words, can computer solutions be broken down into checkpoints and heuristics at the expense of a few moves?

