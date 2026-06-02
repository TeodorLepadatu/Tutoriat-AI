# 2025 AI exam solutions

## Part 1

### Problem 1: Sims

In the Sims game, the players control some characters. Let's simulate plate washing! (excuse me, what?) The characters are located in an empty house, surrounded by black walls (this is getting weird). The game elements are plates, a sink and walls (which are impassable obstacles and represented as black squares). The goal is to bring every plate to the sink.

**Limitations:**

- The player can carry a maximum of F plates at a time.
- The player can move to any of the 8 adjacent tiles, but only if they are not obstacles.
- Every step made on a line or a column costs 1 energy point, while a diagonal step costs 1.5 energy points.
- The plates and the sink are not obstacles. To pick up a plate, the player must enter the tile where the plate is located. To drop a plate, the player must enter the tile where the sink is located. These actions do not cost energy points.

**Notations:**

- NPF = number of plates that the player is currently carrying
- NFR = number of remaining plates on the map
- RC = row of the sink
- (RP, CP) = (row, column) of the player
- RF = row of the closest plate to the player calculated using the Manhattan distance

**Map description:**

The map has 10 rows (0 to 9) and 20 columns (0 to 19).

The plates are located at the following positions:
- (3,5)
- (2,15)
- (2,16)
- (4,17)
- (7,1)
- (6,10)
- (8,2)

The sink is located at position (8,4)

The map looks like this:

```
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
  +-----------------------------------------+
0 | # # # # # # # # # # # # # # # # # # # # |
1 | # . . . . . . # . . . . . . . . . . . # |
2 | # . . . . . . # . . . # . . . P P . . # |
3 | # . . . . P . # . . . # . . . . . . . # |
4 | # . . . . . . . . . . # . . . . . P . # |
5 | # # # # # # # # # . # # # # . # # # # # |
6 | # . . . . . . . . . P . # . . . # . . # |
7 | # P . . . . . . . . . . # . . . . . . # |
8 | # . P . S . . . . . . . # . . . # . . # |
9 | # # # # # # # # # # # # # # # # # # # # |
  +-----------------------------------------+
```
A state is represented as a tuple of the form (RP, CP, NPF, list of coordinates of the uncolleceted plates on the map).

A state is represented as a tuple of the form (RP, CP, NPF, list of coordinates of the uncolleceted plates on the map).

#### Questions:

1. For this question, the character starts at position (1,1). We are trying to find the shortest path from the initial state to a final state, where all plates are collected and dropped at the sink. What is the greatest number of children that a state node can have in the search tree?

a) 7; 

b) 0;

c) 2; 

d) 1; 

e) 8; 

f) None of the above.

Answer: e) 8;

Explaination: A state node expands based on the available movement actions. The player can move in 8 different directions (horizontally, vertically, and diagonally). The actions of picking up or dropping plates occur automatically upon entering the corresponding tiles ("must enter the tile") and do not represent separate branching actions in the state space tree. Therefore, in the absence of obstacles or map boundaries (e.g., if the player is in the middle of the room), a node can generate a maximum of 8 children.

2. For NFR = 0, we consider that the following heuristics return a value of 0. For NFR > 0, which of the following heuristics are admissible?

a) sum of the Manhattan distances from each uncollected plate to the exit (the closest to the sink) of the room that it is located in;

b) sum of the Manhattan distances from each uncollected plate to the sink;

c) sum of the Euclidian distances from each uncollected plate to the sink;

d) number of remaining plates on the map (NFR);

e) the smallest Euclidian distance between 2 uncollected plates (or 0 if NFR = 1);

f) None of the above.

Answer: e) the smallest Euclidian distance between 2 uncollected plates (or 0 if NFR = 1);

Explaination: An admissible heuristic must never overestimate the true cost to reach the goal. Options a, b, and c are inadmissible because summing distances for *each* plate ignores the player's ability to carry multiple plates simultaneously, leading to massive overestimations. Option d (NFR) is inadmissible because picking up adjacent plates can cost less energy than the count of the plates (e.g., cost to travel between two adjacent plates is 1, but they account for 2 in the NFR count; a single step could potentially collect a plate and drop at a sink simultaneously for a cost lower than the plate count). Option e is strictly admissible: if NFR >= 2, the player must traverse at least the minimum distance between the two closest plates to link them. Since Euclidean distance is always less than or equal to the true movement cost grid, it provides a guaranteed lower bound.

3. Consider the following estimation:

If $NPF < F$, then: $\hat{h_1}(state) = |RP-RF|$.

If $NPF = F$ or $NFR = 0$, then: $\hat{h_1}(state) = |RP-RC|$.

For this question, the character starts at position (8,4) and F = 3. What will be the value of $\hat{f}$ on the position of the plate at (7,1), when the corresponding node is introduced for the first time in the OPEN list?

a) 3;

b) 0;

c) 4.5;

d) 3.5;

e) 5;

f) None of the above.

Answer: c) 4.5;

Explaination: A* expands nodes by their estimated total cost $f = g + h_1$. From the starting position S at (8,4), the node at (7,2) can be reached with a true cost $g = 2.5$. Expanding (7,2) generates a step into the plate at (7,1) with $g = 3.5$. At this specific point of generation, the plate at (8,2) has not yet been picked up on this specific path branch. The remaining closest plate to (7,1) by Manhattan distance is (8,2) (thus, RF = 8). The heuristic calculates as $h_1 = |RP - RF| = |7 - 8| = 1$. The total estimated cost is $f = g + h_1 = 3.5 + 1 = 4.5$. 

4. For this question, the character starts at position (1,1). We are considering the heuristic from the previous question and F = 3. What would be the order in which the plates are collected returned by A* so that we have a minimal energy cost?

a) (8,2), (7,1), (6,10), (3,5), (2,15), (2,16), (4,17);

b) (3,5), (6,10), (2,15), (2,16), (4,17), (8,2), (7,1);

c) (2,15), (4,17), (2,16), (7,1), (6,10), (3,5), (8,2);

d) (6,10), (3,5), (2,15), (2,16), (4,17), (8,2), (7,1) ;

e) (6,10), (3,5), (2,15), (7,1), (2,16), (4,17), (8,2) ;

f) None of the above.

Answer: b) (3,5), (6,10), (2,15), (2,16), (4,17), (8,2), (7,1);

Explaination: The optimal path minimizes energy by grouping plates intelligently, avoiding backtracking, and strictly adhering to the F=3 capacity limit. The Eastern cluster of plates ((2,15), (2,16), (4,17)) is very far from the sink and should be collected in a single, dedicated trip to prevent crossing the map multiple times. A 2-3-2 trip grouping handles this best: Trip 1 collects the mid-map plates (3,5) and (6,10) on the initial way from (1,1) to the sink. Trip 2 is dedicated exclusively to the far Eastern cluster, maximizing the 3-plate capacity limit. Trip 3 cleans up the remaining nearby Western plates (8,2) and (7,1). This sequence matches option b.

5. Consider the following heursistics: $\hat{h_1}$ (defined in question 3), $\hat{h_2}(state) = NPF$, $\hat{h_3}(state) = 1$, $\hat{h_4}(state) = 0$. Then, for any map configuration, which of the following statements are true?

a) $\hat{h_1}$ inadmissible;

b) $\hat{h_2}$ consistent;

c) $\hat{h_2}$ inadmissible;

d) $\hat{h_3}$ inadmissible;

e) $\hat{h_4}$ admissible;

f) None of the above.

Answer: c) $\hat{h_2}$ inadmissible; d) $\hat{h_3}$ inadmissible; e) $\hat{h_4}$ admissible;

Explaination: A heuristic is admissible if it never overestimates the true cost to reach the goal. 
- Statement **e** is true: $\hat{h_4} = 0$ is unconditionally admissible as true cost is always $\ge 0$. 
- Statement **d** is true: $\hat{h_3} = 1$ is inadmissible because at the final goal state the true cost is 0, but the heuristic returns 1, thereby overestimating. 
- Statement **c** is true: $\hat{h_2} = NPF$ is inadmissible because dropping plates costs 0 energy. If the player is on or adjacent to the sink carrying F plates, the true cost is 0 or 1, but $\hat{h_2}$ returns F (e.g., 3), overestimating the cost.
Because $\hat{h_2}$ is inadmissible, it logically cannot be consistent, making **b** false. Finally, $\hat{h_1}$ represents a true row-distance lower bound (since vertical traversal is required to reach plates/sinks), meaning it is admissible, making **a** false.

6. Consider the heuristics from the previous question. For any map configuration, which of the following heuristics are admissiblle?

a) $\hat{h_1} + \hat{h_2}$;

b) $\hat{h_1} + \hat{h_3}$;

c) $\frac{\hat{h_3}}{2}$;

d) $\hat{h_2}*0.3$;

e) $3*\hat{h_2}$;

f) None of the above.

Answer: f) None of the above.

Explaination: None of these heuristic combinations guarantee admissibility across any map configuration. For options a, d, and e, the inclusion of $\hat{h_2}$ (which scales with NPF) causes overestimations. For instance, if the character is at the sink holding F plates, the true cost to finish is 0 energy points. Any multiplier or addition of $\hat{h_2}$ will result in a value $> 0$, overestimating the cost. Regarding options b and c, the inclusion of $\hat{h_3}$ means that at the exact goal state (true cost = 0), the function will still evaluate to a number greater than 0 (e.g., $\frac{1}{2} = 0.5$), which violates the primary admissibility rule where $h(goal) = 0$.
