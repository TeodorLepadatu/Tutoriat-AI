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

### Problem 2: 4-piece game
#### Game Setup

Grid: $7 \times 7$ grid, initially empty.

#### Players: 

MAX (the computer) controls the White pieces and starts from corner $(0,0)$.

MIN (the user) controls the Black pieces and starts from corner $(6,6)$.

Starting Player: White (MAX) takes the first turn.

#### Player's Turn

A player's turn consists of up to two actions:
If the current player's starting position is empty, a new piece of their color automatically appears on it.

Regardless of whether a new piece was placed, the player can move any one of their pieces already on the board by $X$ positions ($1 \le X \le 6$) along a single row or column.

#### Valid Move Conditions 
A move is only valid if the piece stays within the board boundaries and does not overlap with or pass through any other piece (of either color) at any point during its movement.

#### End of the Game
#### Winning Condition
A player wins the moment they form a sequence of at least 4 consecutive adjacent pieces of their own color along any single row or column.

#### Exception
If all pieces of the sequence are located entirely on the perimeter (the very first or very last row/column of the grid), the sequence is not considered a winning one.

#### Draw Condition

If a player has no valid moves available at the beginning of their turn, the game ends immediately and is declared a draw.

Game 1:
The piece at $(3,2)$ can move to $(1,2)$, $(2,2)$, $(3,1)$, $(3,3)$, and $(4,2)$. It cannot move to $(0,2)$, $(3,0)$, $(3,4)$, $(3,5)$, $(3,6)$, $(5,2)$, or $(6,2)$ because it cannot pass through or land on other pieces.

```
0 1 2 3 4 5 6
  +---------------+
0 | O . O . O . . |
1 | . X . . . . X |
2 | . . . . . . . |
3 | . . X . X . . |
4 | . . . . . . . |
5 | . . O . . O . |
6 | . X . . . . X |
  +---------------+
```

  Game 3:
  This is not a winning state for 0. Even though there are 4 consecutive 0 pieces at $(0,1)$, $(0,2)$, $(0,3)$, and $(0,4)$, they are located entirely on row 0 (the edge/border of the matrix), which does not count as a win according to the exception rule.

  ```
  0 1 2 3 4 5 6
  +---------------+
0 | . O O O O . . |
1 | . X O X X . X |
2 | O X . . . X . |
3 | . O . X . . . |
4 | O . . . . O . |
5 | O X . . O O . |
6 | . . X X . . X |
  +---------------+
```
Game 6:

It is the 0 player's turn

```
0 1 2 3 4 5 6
  +---------------+
0 | O . . . O O . |
1 | . O . X O . X |
2 | . X . X . O . |
3 | O X . . . X X |
4 | O X . . O . . |
5 | O O O X X . X |
6 | . X X . . X . |
  +---------------+
```
#### Notations

* **NMAX, NMIN** – The number of pieces belonging to MAX and MIN, respectively.
* **V(node)** – The Minimax value of the node.
* **Level numbering** starts from 0.
* For multiple move options that share the exact same Minimax value, choose the **first one from the left** (within the search tree).

#### Utility Function (Heuristic)

$$f(\text{state}) = \begin{cases} 
+\infty & \text{MAX wins} \\ 
-\infty & \text{MIN wins} \\ 
0 & \text{Draw} \\ 
f_u(\text{state}) & \text{for non-final states (defined below)} 
\end{cases}$$

```
for r in 0..6:
    for c in 0..6:
        for r_new in 0..6:
            for c_new in 0..6:
                if valid_move(r, c, r_new, c_new):
                    add move to the end of the move list
```

7. For Game 3, assuming it is MAX's turn (the automatic piece placement has not happened yet), select the true variants regarding the generation of the Minimax tree with a maximum depth of 100 and $f_u(\text{state}) = \text{NMAX} - \text{NMIN}$:

a) In the resulting tree, there exist two identical game boards (with pieces arranged exactly the same way).

b) For any node in the tree, $|\text{NMAX} - \text{NMIN}| \le 2$. (Note: Written as 'a' again in the original text).

c) There is no leaf in the tree located at exactly level 100.

d) There is no node that does not have empty squares on the board.

e) There is a leaf at level 1.

f) There is a leaf at level 2.

#### Solution:
a) - True. In this game framework, pieces can move back and forth along rows and columns. Because players can move a piece and then move it back on a subsequent turn (or cycles can form across multiple turns), the game state space contains cycles. A Minimax tree expanded to a depth of 100 without a transposition table strictly tracking history will definitely encounter repeating board configurations along different paths or deeper down the same branch.

b) - True. Let's track how pieces are added:

Initially, at the root before the turn: $\text{NMAX} = 12$, $\text{NMIN} = 11 \implies |\text{NMAX} - \text{NMIN}| = 1$.

MAX's turn starts $\rightarrow$ 'O' spawns at $(0,0)$ $\rightarrow$ $\text{NMAX} = 13, \text{NMIN} = 11 \implies |\text{NMAX} - \text{NMIN}| = 2$.

MAX moves a piece. Board state evaluated here or passed to MIN.

MIN's turn starts $\rightarrow$ MIN's starting corner is $(6,6)$. Looking at the board, $(6,6)$ is currently occupied by an 'X'. Thus, no piece spawns for MIN. The counts stay $\text{NMAX} = 13, \text{NMIN} = 11 \implies |\text{NMAX} - \text{NMIN}| = 2$.

If $(6,6)$ becomes empty later because MIN moves that piece away, then on a subsequent MIN turn, an 'X' will spawn, bringing counts to $\text{NMAX} = 13, \text{NMIN} = 12 \implies |\text{NMAX} - \text{NMIN}| = 1$.

Since only 1 piece can ever spawn per turn sequence, and no pieces are ever captured/removed from the board, NMAX can only ever be at most $\text{NMIN} + 2$ (when MAX spawns and MIN doesn't) or equal to $\text{NMIN}$ (if MIN caught up). The absolute difference will structurally never exceed 2.

c) - False. Minimax trees built with a maximum depth constraint convert any node at depth == max_depth (Level 100) automatically into a leaf node evaluated by the heuristic function $f_u(\text{state})$, unless the game ends sooner on all branches (which it doesn't, given the cyclic moving space and lack of immediate forced wins). Thus, there will be plenty of leaves at level 100.

d) - True. The board is $7 \times 7 = 49$ total squares. At the root, there are $12 + 11 = 23$ pieces, leaving 26 empty spaces. Since pieces are only added when a starting corner is empty, and never more than one per turn, the total number of pieces can never grow fast enough to fill all 49 squares within the tree's depth, let alone instantly. Every reachable state will have empty squares.

e) - False. Level 1 represents the tree states immediately after MAX's first move choice. For a leaf to exist at Level 1, MAX must be able to win the game in exactly one move:

Row 0 has 4 consecutive Os, but they are entirely on the perimeter, so it doesn't count as a win.

There is no legal single-move displacement that allows MAX to form a non-perimeter line of 4 consecutive O pieces anywhere else on the board.

A Draw leaf at Level 1 is also impossible because a draw only occurs if a player has no valid moves, and MAX has many open paths. Therefore, no branch terminates at Level 1.

f) - False: Because nobody can win in just one round, and nobody can run out of moves, every single game state at level 2 is still actively playable. Since no path triggers an immediate game-over, there are no terminal leaves at level 2.

8. Let Game 6 be evaluated using Minimax with a maximum depth of 1.
$$f_u(\text{state}) = \text{LMAX}(3) - \text{LMIN}(3)$$
Where $\text{LMAX}(n) =$ the number of groups of 4 adjacent cells aligned in a chain along the same row or column, in which there are exactly $n$ pieces belonging to MAX, with the remaining cells being empty. $\text{LMIN}(n)$ represents the exact same calculation from MIN's perspective.
MAX will choose the move:


 a) $(4,4) \rightarrow (4,5)$
 
 b) $(3,5) \rightarrow (5,5)$
 
 c) $(0,0) \rightarrow (0,1)$
 
 d) $(0,0) \rightarrow (0,3)$
 
 e) The number of nodes at level 1 is 29.
 
 f) None of the options are correct.

 #### Solution
a) - False. There is an O at $(4,4)$ and $(4,5)$ is empty, so this move is legal.
 
 Moving here changes Row 4 to O X . . . O .. This does not form any new non-perimeter consecutive lines of 3 O pieces.

b) - False. Invalid Move. The square $(3,5)$ contains an X (MIN's piece). MAX cannot move opponent pieces.

c) - False. There is an O at $(0,0)$ and $(0,1)$ is empty, so this move is legal.

Row 0 becomes: . O . . O O .

Testing 4-cell windows on Row 0:

Columns 0–3: . O . . (1 O)

Columns 1–4: O . . O (2 O)

Columns 2–5: . . O O (2 O)

Columns 3–6: . O O . (2 O)

No 4-cell window contains exactly 3 O pieces.

d) - True. There is an O at $(0,0)$ and $(0,3)$ is empty. The intermediate squares $(0,1)$ and $(0,2)$ are also completely empty, making this a legal straight-line move.

Moving the piece changes Row 0 to: . . . O O O .

Let's check the 4-cell windows across Row 0 now:

Columns 0–3: . . . O (1 O)Columns 1–4: . . O O (2 O)

Columns 2–5: . O O O $\rightarrow$ Contains exactly 3 O pieces and 1 empty cell! ($+1$ to $\text{LMAX}(3)$)

Columns 3–6: O O O . $\rightarrow$ Contains exactly 3 O pieces and 1 empty cell! ($+1$ to $\text{LMAX}(3)$)

By sliding to $(0,3)$, MAX aligns a consecutive string of three O pieces ((0,3), (0,4), (0,5)), satisfying two scoring windows simultaneously. This increases the heuristic evaluation value by $+2$.

e) False: Piece at $(0,0)$: Can move right to $(0,1)$, $(0,2)$, $(0,3)$. (It is blocked from moving further by the O at $(0,4)$). It can move down to $(1,0)$ and $(2,0)$ (blocked by the O at $(3,0)$).

Moves = 5 ((0,1), (0,2), (0,3), (1,0), (2,0))

Piece at $(0,4)$: Blocked horizontally by O at $(0,0)$ and $(0,5)$. Can move down to $(1,4)$? No, $(1,4)$ is occupied by another O.

Moves = 0

Piece at $(0,5)$: Blocked left by O at $(0,4)$. Can move right to $(0,6)$. Can move down to $(1,5)$? Yes, $(1,5)$ is empty, but $(2,5)$ is occupied by O.

Moves = 2 ((0,6), (1,5))

Piece at $(1,1)$: Can move left to $(1,0)$. Blocked right by X at $(1,3)$. Can move up to $(0,1)$. Can move down to $(2,1)$? No, $(2,1)$ is X.

Moves = 2 ((1,0), (0,1))

Piece at $(1,4)$: Can move left to $(1,2)$. Blocked right by X at $(1,6)$. Blocked up by O at $(0,4)$. Can move down to $(2,4)$ and $(3,4)$ (blocked by O at $(4,4)$).

Moves = 3 ((1,2), (2,4), (3,4))

Piece at $(2,5)$: Can move left to $(2,4)$, $(2,2)$, $(2,0)$? Let's check Row 2: . X . X . O .. It can move left to $(2,4)$ and $(2,2)$ (blocked by X at $(2,3)$). Can move right to $(2,6)$. Can move up to $(1,5)$. Can move down to $(3,5)$? No, $(3,5)$ is X.

Moves = 4 ((2,4), (2,2), (2,6), (1,5))

Piece at $(3,0)$: Blocked up by $(2,0)$. Can move down to $(3,1)$? No, X. Can move right to $(3,2)$, $(3,3)$, $(3,4)$ (blocked by X at $(3,5)$).

Moves = 3 ((3,2), (3,3), (3,4))

Piece at $(4,0)$: Can move up/down to $(3,0)$ or $(5,0)$? No, both occupied by O. Can move right to $(4,2)$, $(4,3)$ (blocked by O at $(4,4)$). Note that $(4,1)$ is X, so it's blocked immediately at $(4,1)$.

Moves = 0

Piece at $(4,4)$: Can move left to $(4,3)$, $(4,2)$ (blocked by X at $(4,1)$). Can move right to $(4,5)$, $(4,6)$. Can move up to $(3,4)$, $(2,4)$ (blocked by O at $(1,4)$). Can move down to $(5,4)$? No, $(5,4)$ is X.

Moves = 4 ((4,3), (4,2), (4,5), (4,6))

Piece at $(5,0)$: Blocked up by O at $(4,0)$. Can move down to $(6,0)$. Blocked right by O at $(5,1)$.

Moves = 1 ((6,0))

Piece at $(5,1)$: Blocked left by $(5,0)$, right by $(5,2)$. Can move up to $(4,1)$? No, X. Can move down to $(6,1)$? No, X.

Moves = 0

Piece at $(5,2)$: Blocked left by $(5,1)$. Can move right to $(5,3)$? No, X. Can move up to $(4,2)$, $(3,2)$, $(2,2)$, $(1,2)$, $(0,2)$. Let's check column 2: all these are empty! Can move down to $(6,2)$? No, X.

Moves = 5 ((4,2), (3,2), (2,2), (1,2), (0,2))

Summing these up: $5 + 0 + 2 + 2 + 3 + 4 + 3 + 0 + 4 + 1 + 0 + 5 = \mathbf{29}$ nodes.

Because the total number of legal moves at Level 1 is exactly 29, statement e is technically a true fact about the tree structure. However, the question asks "MAX will choose the move:", meaning the target answer must explicitly identify MAX's chosen move choice.

f) False.

9. Let the initial state be Game 6. Alpha-Beta pruning is applied with a maximum depth of 2, and the evaluation function is:
$$f_u(\text{state}) = \text{NMAX} - \text{NMIN}$$

a) The principal variation (the optimal path) has a length of 2.

b)  There exists a child of the root node with $V(\text{child}) = +\infty$.

c) There exists a child of the root node with $V(\text{child}) = -\infty$.

d) On level 2, there exists a node with $V(\text{node}) = -1$.

e)On level 2, there exists a node with $V(\text{node}) = -\infty$.

f) None of the options are correct.

10. Same requirement

a) The game value is $0$.

b) The game value is $+\infty$.

c) The game value is $-\infty$.

d) There is an unpruned leaf node with $V(\text{leaf}) = +\infty$.

e) There is an unpruned leaf node with $V(\text{leaf}) = 1$.

f) None of the options are correct.

#### Solution for 9 and 10
We apply Alpha-Beta pruning with a maximum depth of 2 (Level 0 = MAX, Level 1 = MIN, Level 2 = Leaves evaluated via $f_u(\text{state}) = \text{NMAX} - \text{NMIN}$).

Step 1: Analyze if MAX can win immediately at Depth 1

Can MAX achieve 4 consecutive Os in a non-perimeter position in 1 move?

Look at Column 0: $(3,0), (4,0), (5,0)$ are Os. If MAX could move another O to $(2,0)$, it would create a 4-in-a-row. However, Column 0 is a perimeter edge, which is an exception and does not count as a winning state.

Look at Row 5: $(5,0), (5,1), (5,2)$ are Os. The adjacent cells are $(5,3)$, which is occupied by X. MAX cannot win here.

There are no other 3-in-a-row setups for MAX that can be completed in a non-perimeter line. Thus, no child of the root node evaluates to $+\infty$. (Option b is false).

Step 2: Analyze MIN's Counter-play (Level 1 to Level 2)

After MAX makes a move, it becomes MIN's turn.

MIN's Spawn Phase: MIN controls Black (X) and starts at corner $(6,6)$. Looking at the board, $(6,6)$ is currently completely empty (.).

Therefore, at the start of every legal variation belonging to MIN's turn at Level 1, an X will automatically spawn at $(6,6)$.

Piece count shift: Since $\text{NMAX}$ stays at $12$ (no spawn happened for MAX at Level 0), and $\text{NMIN}$ automatically goes from $14$ to $15$ due to the spawn, the baseline utility calculation for any standard non-terminal leaf at Level 2 becomes:

$$f_u(\text{state}) = \text{NMAX} - \text{NMIN} = 12 - 15 = -3$$

If MIN moves a piece without winning, the evaluation score stays exactly $-3$.

Step 3: Checking for a MIN Win ($V = -\infty$)

Can MIN form a non-perimeter 4-in-a-row sequence during its turn to force a victory ($-\infty$)?Look at Column 1: It features $(2,1)=\text{X}, (3,1)=\text{X}, (4,1)=\text{X}$. The cell $(1,1)$ is occupied by $\text{O}$, and $(5,1)$ is occupied by $\text{O}$. MIN is blocked from extending this chain.

Look at Row 5: contains $(5,3)=\text{X}, (5,4)=\text{X}$. The cell $(5,2)$ is occupied by $\text{O}$.

Look at Column 3: contains $(1,3)=\text{X}, (2,3)=\text{X}$.

Since MIN cannot achieve an immediate winning configuration on its turn, no leaf node drops to $-\infty$. (Options 9c, 9e, and 10c are false).

Step 4: Assessing Other Score Variations

Since no pieces are added or captured outside of MIN's deterministic spawning, the evaluation function is statically pinned at $\text{12} - \text{15} = -3$ for all leaf nodes.

It is impossible to achieve a node value of $-1$ or $1$ because piece counts cannot arbitrarily fluctuate to those numbers within depth 2.

Since all valid paths end up yielding an identical terminal leaf value of $-3$, the minimax calculation flows perfectly down the tree, resulting in an optimal game value of $-3$.

Conclusions for 9 and 10:

For Question 9: All detailed specific valuation scenarios (b, c, d, e) fail. The principal variation path will evaluate non-terminally through depth 2 (length 2).

For Question 10: The final determined minimax game value is $-3$. None of the explicit properties given match this value.

Question 9 Correct Answer: a) The principal variation (the optimal path) has a length of 2. 

Question 10 Correct Answer: f) None of the options are correct.

11. Let Game 6 be evaluated using Alpha-Beta pruning with a maximum depth of 1 (at the root node, $\alpha = -\infty$ and $\beta = +\infty$).

The evaluation function is $f_u : \{\text{set of non-final states}\} \rightarrow \mathbb{R}^*$, which is an arbitrary but fixed function.

Compared to the default loops provided in the pseudocode, if we change the iteration order:

a) Iterating `c_new` from 6 down to 0, we will obtain a smaller total number of nodes in the tree.

b) Iterating `c_new` from 6 down to 0, we will obtain a larger total number of nodes in the tree.

c) Iterating `r` from 6 down to 0, we will obtain a smaller total number of nodes in the tree.

d) Iterating `r` from 6 down to 0, we will obtain a larger total number of nodes in the tree.

e) ?

f) None of the options are correct ($\emptyset$).

#### Solution
For options a) and c) (Reversing always makes the tree smaller): If $f_u$ happens to assign the highest scores to positions near the upper-left of the board (low coordinates like row 0, column 0), the default loop (0..6) will find them first and prune the tree efficiently. Reversing the loop to count down from 6 would force the algorithm to evaluate the worst moves first, resulting in a larger tree, not a smaller one.

For options b) and d) (Reversing always makes the tree larger): Conversely, if $f_u$ happens to favor positions near the lower-right of the board (high coordinates like row 6, column 6), the default loop will evaluate the worst moves first. Reversing the loop to count down from 6 would suddenly cause it to find the best moves immediately, making the total tree size smaller, not larger.

Thus, the solution is f).
## Part 2

### Problem 1: Multiple choice questions
1. Which of the following variables allows us to search for the maximum margin hyperplane in the case of a non-linearly separable configuration of points, by ignoring certain points?

a) $w$

b) Not applicable

c) $\gamma$

d) $\xi$"

Solution: d) $\xi$"
* **a) $w$**: the weight vector, which defines the orientation and slope of the separating hyperplane.
* **c) $\gamma$**: denote the geometric margin itself
* **d) $\xi$**: used to handle non-linearly separable data.

2. What is the effect of applying the batch normalization operation on a batch consisting of a single example?

a) The norm of the example becomes 1

b) The norm of the example becomes 0

c) The example remains unmodified

d) It cannot be applied because it implies division by 0

Solution: b) The norm of the example becomes 0
The standard formula for Batch Normalization is:
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
If a batch consists of only a single example ($N = 1$), the batch mean $\mu$ is equal to the value of the example itself ($x$), meaning the numerator becomes $x - \mu = 0$. Concurrently, the sample variance $\sigma^2$ is $0$. A tiny smoothing term $\epsilon$ is added to avoid a hard mathematical division by zero ($\frac{0}{0}$). This results in:
  $$\hat{x} = \frac{0}{\sqrt{0 + \epsilon}} = 0$$

3. How could the formula $\max\left(0, \frac{e^x - e^{-x}}{e^x + e^{-x}}\right)$ be rewritten?

a) $ReLU(\tanh(x))$

b) $\tanh(ReLU(x))$

c) $ReLU(\sigma(x))$

d) $\sigma(\tanh(x))$

Solution: a) $ReLU(\tanh(x))$

4. The result of applying the string kernel function based on presence bits using 2-grams on the strings "ana are mere" and "ioana are banane" is:

a) 11

b) 7

c) 8

d) 10

Soltuion: b) 7
A presence-bit 2-gram string kernel evaluates the dot product of two binary vectors representing whether unique character pairs exist in the texts.
  * Unique 2-grams in `"ana are mere"`: `an`, `na`, `a `, ` a`, `ar`, `re`, `e `, ` m`, `me`, `er`
  * Unique 2-grams in `"ioana are banane"`: `io`, `oa`, `an`, `na`, `a `, ` a`, `ar`, `re`, `e `, ` b`, `ba`, `ne`
  
  The intersecting shared 2-grams between both strings are: `an`, `na`, `a `, ` a`, `ar`, `re`, and `e `. Counting these common unique tokens yields exactly **7** overlapping elements.

5. Which of the following operations produces a valid kernel function?

a) $k_1(k_2(x, z), k_3(x, z))$

b) $k_1(x) + k_2(y)$

c) $k_1(x, y) + k_2(x^2, y) \cdot k_3(x, y^2)$

d) None of the choices

Solution: d) None of the choices
Valid kernels must adhere strictly to Mercer's theorem and structural algebraic closure properties (such as closure under addition and multiplication over matching variable domains). 
  * Option B combines unrelated single coordinates ($x$ and $y$) linearly without evaluating pair relationships properly.
  * Option C features asymmetrical mappings within the coordinate spaces ($x^2$), which violates the positive semi-definiteness framework required for general validation.
    
6. What function can be applied to transform the $L_2$ distance into a kernel function?

a) Linear kernel

b) RBF kernel

c) Intersection kernel

d) Polynomial kernel

Solution:b) RBF kernel
The Radial Basis Function (Gaussian) kernel is defined as:
  $$k(x, y) = \exp\left(-\gamma \|x - y\|^2\right)$$
  This formula applies a non-linear exponential transformation directly onto the squared $L_2$ Euclidean distance ($\|x - y\|$) separating two vectors.

8. What happens if we set momentum equal to 1 in the SGD with momentum algorithm?

a) The algorithm gets stuck in a local minimum

b) Optimization goes to infinity

c) The algorithm becomes equivalent to standard SGD

d) The algorithm diverges

Solution: d) The algorithm diverges
In Stochastic Gradient Descent with momentum, the velocity update follows $v_t = \beta v_{t-1} + \eta \nabla L$. Setting $\beta = 1$ removes all friction or velocity decay from previous iterations. The historical kinetic energy builds up boundlessly, forcing the optimizer to completely diverge.

8. What condition must a neural network satisfy to learn non-linearly separable data?

a) It must contain at least one hidden layer with non-linear activation

b) It must contain at least one hidden layer

c) It must contain at least two hidden layers

d) It must use at least one softmax layer

Solution: a) It must contain at least one hidden layer with non-linear activation
Per the Universal Approximation Theorem, a multi-layer network requires at least one hidden layer paired with a non-linear activation function (such as ReLU, Sigmoid, or Tanh) to fit non-linear boundaries. Linear hidden layers collapse mathematically into a single, straightforward linear transformation.

9. When $n \to \infty$, the empirical error becomes equivalent to:

a) Optimization error

b) Modeling error

c) None of the choices

d) Generalization error

Solution: d) Generalization error
As the size of the training dataset approaches infinity ($n \to \infty$), the empirical risk computed on that sample group converges directly to the true expected risk across the entire data distribution, which is defined as the generalization error.

10. The Kendall $\tau$ correlation for the labels $[0.1; 0.2; 0.3; 0.4; 0.5]$ and predictions $[0.4; 0.2; 0.6; 1.0; 0.8]$ is:

a) 0.4

b) 0.5

c) 0.6

d) 1.5

Solution: c) 0.6
Kendall's $\tau$ evaluates the relative ranking ordering between pairs. 
  * Labels (Sorted ascending): $[0.1, 0.2, 0.3, 0.4, 0.5] \to$ Ranks: $1, 2, 3, 4, 5$
  * Predictions: $[0.4, 0.2, 0.6, 1.0, 0.8] \to$ Relative ordering sequence: $2, 1, 3, 5, 4$

  Now we count the concordant and discordant pairs out of the $\binom{5}{2} = 10$ combinations:
  * (1,2): Labels $(1 < 2)$, Preds $(2 > 1) \to$ **Discordant**
  * (1,3): Labels $(1 < 3)$, Preds $(2 < 3) \to$ **Concordant**
  * (1,4): Labels $(1 < 4)$, Preds $(2 < 5) \to$ **Concordant**
  * (1,5): Labels $(1 < 5)$, Preds $(2 < 4) \to$ **Concordant**
  * (2,3): Labels $(2 < 3)$, Preds $(1 < 3) \to$ **Concordant**
  * (2,4): Labels $(2 < 4)$, Preds $(1 < 5) \to$ **Concordant**
  * (2,5): Labels $(2 < 5)$, Preds $(1 < 4) \to$ **Concordant**
  * (3,4): Labels $(3 < 4)$, Preds $(3 < 5) \to$ **Concordant**
  * (3,5): Labels $(3 < 5)$, Preds $(3 < 4) \to$ **Concordant**
  * (4,5): Labels $(4 < 5)$, Preds $(5 > 4) \to$ **Discordant**

  $$\text{Concordant (C)} = 8, \quad \text{Discordant (D)} = 2$$
  $$\tau = \frac{C - D}{\text{Total Pairs}} = \frac{8 - 2}{10} = 0.6$$

### Problem 2: Neuron

Consider the following set of examples: S={([1,1,0],1), ([0,1,0],0), ([0,1,1],1), ([1,1,1],0)}.

a) Apply a neuron with the weights w = [0,-1,0] and bias b = 0.5 using the ReLU activation function to the examples in S.

Solution:

The pre-activation value is $z = w \cdot x + b$ and the ReLU activation is $a = \max(0, z)$.

For $x^{(1)} = [1, 1, 0]$: $z_1 = 0(1) - 1(1) + 0(0) + 0.5 = -0.5 \implies a_1 = \max(0, -0.5) = 0$

For $x^{(2)} = [0, 1, 0]$: $z_2 = 0(0) - 1(1) + 0(0) + 0.5 = -0.5 \implies a_2 = \max(0, -0.5) = 0$

For $x^{(3)} = [0, 1, 1]$: $z_3 = 0(0) - 1(1) + 0(1) + 0.5 = -0.5 \implies a_3 = \max(0, -0.5) = 0$

For $x^{(4)} = [1, 1, 1]$: $z_4 = 0(1) - 1(1) + 0(1) + 0.5 = -0.5 \implies a_4 = \max(0, -0.5) = 0$


b) What is the problem that occurs when optimizing this neuron? Propose a new value for b so that you solve the identified problem. Prove that the new value actually solves the problem.

Solution:

The neuron suffers from the "dying ReLU" problem on this dataset. Because the pre-activation $z$ is negative for all inputs in $S$, the output is consistently $0$, and the derivative of the ReLU function is exactly $0$. Consequently, no gradients will flow backwards through this neuron during backpropagation, and the weights cannot be updated.

We must ensure $z > 0$ for at least some examples to yield a non-zero gradient. Since $w \cdot x = -1$ for all examples in $S$, we need $b > 1$. Let's choose $b = 2$.

Using the new bias $b = 2$:
$z_i = w \cdot x^{(i)} + b = -1 + 2 = 1 \text{ for all } i \in \{1, 2, 3, 4\}$
The activation becomes $a_i = \max(0, 1) = 1$. 
Because $z_i > 0$, the local derivative of the ReLU activation is $1$ (not $0$). This allows the gradient to pass through the neuron during backpropagation, successfully resolving the dying ReLU problem.

c) Is there any neuron with the maxout activation function that can correctly predict (with an error of 0) the labels of the examples in S?

Solution:

Yes. A maxout unit computes the maximum of $k$ affine functions: $f(x) = \max_{j=1}^k (w^{(j)} \cdot x + b_j)$ (so basically a generalization of ReLU). 
Notice that $x_2 = 1$ for all inputs in $S$, rendering it a constant. The mapping of the remaining features $(x_1, x_3)$ to the labels forms an XOR problem. A maxout neuron with $k=2$ linear functions can perfectly represent this mapping.

Let the parameters for the two linear functions be:
$w^{(1)} = [1, 0, -1]$ and $b_1 = 0$,
$w^{(2)} = [-1, 0, 1]$ and $b_2 = 0$.

Applying $f(x) = \max(w^{(1)} \cdot x + b_1, w^{(2)} \cdot x + b_2)$ to $S$:

* For $x^{(1)} = [1, 1, 0]$: $f(x^{(1)}) = \max(1(1) + 0(1) - 1(0), -1(1) + 0(1) + 1(0)) = \max(1, -1) = 1$ (Matches label $1$)
* For $x^{(2)} = [0, 1, 0]$: $f(x^{(2)}) = \max(0, 0) = 0$ (Matches label $0$)
* For $x^{(3)} = [0, 1, 1]$: $f(x^{(3)}) = \max(1(0) + 0(1) - 1(1), -1(0) + 0(1) + 1(1)) = \max(-1, 1) = 1$ (Matches label $1$)
* For $x^{(4)} = [1, 1, 1]$: $f(x^{(4)}) = \max(1(1) + 0(1) - 1(1), -1(1) + 0(1) + 1(1)) = \max(0, 0) = 0$ (Matches label $0$)

This maxout neuron yields exactly the correct labels, thus achieving an error of $0$.

### Problem 3: Normalization

Consider a non-empty set of examples S from $\mathbb{R}^3$.

a) What is the geometric object on which the points from S lie after normalizing them using the L2 norm?

Solution:

After L2 normalization, each point $x \in S$ is transformed to a new vector $x' = \frac{x}{\|x\|_2}$. Consequently, the L2 norm of every resulting point becomes exactly $1$ (i.e., $\|x'\|_2 = 1$). In $\mathbb{R}^3$, the set of all points with a Euclidean distance of $1$ from the origin forms the surface of a unit sphere centered at the origin.

b) Is there a neuron that can separate the normalized points $x \in \mathbb{R}^3$ with $x>0$ from all the other?

Solution:

The question is ambiguous. We have 2 main ways of interpreting this (so make sure to ask your teacher what the real interpretation should be):

If we consider that all the points are normalized, then we need to separate the points that have x>0 from those that have x<=0. This can be achieved with a simple linear neuron defined by the weights $w = [1, 0, 0]$ and bias $b = 0$. The decision boundary is the plane defined by $w \cdot x + b = 0$, which simplifies to $x_1 = 0$. This neuron will output a positive value for points where $x_1 > 0$ and a non-positive value for points where $x_1 \leq 0$, effectively separating the two groups of points.

If we consider that not all the points are normalized, then we can't create such a neuron. The normalized points lie on the positive surface of the unit sphere. Because the sphere is curved, no single linear decision boundary (2d plane) can separate the points. The intuitive explaination is that the normalized points are kind of surrounded by the non-normalized points, so we can't find a plane that separates them. In this case, we would need a more complex model (e.g., a multi-layer neural network) to achieve separation.
