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
TODO

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
