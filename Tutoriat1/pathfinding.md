# Domain Overview


The field of artificial intelligence is divided into several fundamental subdomains, including:

- **Knowledge Representation (KR)**
- **Machine Learning (ML)**

In turn, ML is divided into three main paradigms:

- **Supervised learning:** where the model learns from labeled training examples.
- **Unsupervised learning:** focused on identifying hidden structures and patterns in unlabeled data.
- **Reinforcement Learning:** focused on optimizing the decisions of an agent interacting with an environment based on a reward system.

The most successful paradigm is **supervised learning**, which is further divided into two directions:

- *Classification:* where the model learns to associate input data with a finite number of object types/classes.
- *Regression:* where the model learns to predict a continuous numerical value (a function).

Due to the success of these algorithms, ML has generated highly complex applied subdomains dedicated to processing specific types of unstructured data:

- **Natural Language Processing (NLP):** the use of ML models to understand, interpret, and generate human text or speech.
- **Computer Vision (CV):** the use of ML models to extract information, recognize objects, or identify visual patterns from digital images and video content.

# Intelligent Pathfinding Algorithms

For an agent to find the shortest path to a destination in a given problem, it can use problem-specific knowledge (in addition to its formal description).

Thus, it uses functions called *heuristics* (estimates of the remaining path cost to the goal) to reduce the number of explored states.

Denoting the heuristic function as $h$, it has the following fundamental properties:

- $h(end) = 0$ (the estimated cost from the destination to the destination is zero)
- $h(x) \ge 0$ (the estimated distance cannot be negative)
- **Admissibility:** $h(x) \le h^*(x)$, where $h^*(x)$ is the actual minimum cost from node $x$ to the destination (the heuristic never overestimates the remaining cost)
- **Consistency (Monotonicity):** $h(x) \le c(x, y) + h(y)$, where $c(x, y)$ is the actual cost of transitioning from node $x$ to its successor $y$ (satisfies the triangle inequality)



**Examples of heuristics:** Euclidean distance (for movement at any angle), Manhattan distance (for strict horizontal and vertical movement on a grid), or Chebyshev distance (for movement in 8 directions, including diagonally).

## Best-First Search


This algorithm represents an informed search approach. It differs from Breadth-First Search (BFS) in that it does not expand nodes blindly (level by level), but uses a heuristic function $h(n)$ to prioritize exploration.

At each step, the algorithm selects the node $n$ with the lowest estimated value to the destination for expansion. Its evaluation function is strictly the heuristic:
$f(n) = h(n)$

### Implementation in Python

The algorithm uses a priority queue (Min-Heap) to efficiently extract the node with the lowest heuristic value.

```python
import heapq

def best_first_search(graph, start, goal, heuristics):

    priority_queue = [(heuristics[start], start, [start])]
    visited = set()

    while priority_queue:

        _, current_node, path = heapq.heappop(priority_queue)

        if current_node == goal:
            return path

        if current_node not in visited:
            visited.add(current_node)

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristics[neighbor], neighbor, path + [neighbor]))

    return None
```

The time complexity is $O(b^m)$, where $b$ is the maximum branching factor (successors of a node) and $m$ is the maximum depth of the search space. If the heuristic is perfect, the complexity is $O(b \cdot d)$, where $d$ is the depth of the solution. This drastic reduction in complexity is due to the fact that no backtracking step is needed, as the agent always finds the optimal node.The space complexity is $O(b^m)$ in the worst case, because the algorithm must keep all generated nodes in memory.

## A*

This algorithm represents an informed search approach and is considered one of the most efficient pathfinding algorithms. It differs from Greedy Best-First Search by taking into account both the actual accumulated cost from the start node and the estimated cost to the destination.At each step, the algorithm selects the node $n$ that minimizes the total evaluation function for expansion: f(n) = g(n) + h(n), where $g(n)$ is the exact cost of the path from start to node $n$, and $h(n)$ is the heuristic (estimated cost from $n$ to the destination). If the heuristic is admissible, A* guarantees finding the minimum cost (optimal) path.

### Python Implementation

```python
import heapq

def a_star(graph, start, goal, heuristics):

    # priority_queue stores: (f_score, g_score, current_node, path)
    priority_queue = [(heuristics[start], 0, start, [start])]
    visited = set()

    while priority_queue:

        f_score, g_score, current_node, path = heapq.heappop(priority_queue)

        if current_node == goal:
            return path

        if current_node not in visited:
            visited.add(current_node)

            # graph is a dictionary of dictionaries: {current_node: {neighbor: transition_cost}}
            for neighbor, cost in graph.get(current_node, {}).items():
                if neighbor not in visited:
                    new_g_score = g_score + cost
                    new_f_score = new_g_score + heuristics[neighbor]
                    heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor, path + [neighbor]))

    return None
```

The time complexity is $O(b^d)$, and if the heuristic is perfect, the complexity drops to $O(b \cdot d)$.The space complexity is also $O(b^d)$.

## IDA*

This algorithm (Iterative Deepening A*) is a memory-optimized variant of the A* algorithm. It combines the heuristic idea from A* with the iterative deepening depth-first search strategy.Like A*, it uses the evaluation function: $f(n) = g(n) + h(n)$ . The major difference is that IDA* does not keep all nodes in memory. Instead, it performs DFS searches limited by a threshold applied to the value $f(n)$. It starts with an initial threshold: $threshold = h(start)$ A DFS is performed, but only nodes for which $f(n) \le threshold$ are expanded.If the solution is not found, the threshold is updated to the minimum $f(n)$ that exceeded the limit, and the algorithm restarts.

### Python Implementation

```python
def ida_star(graph, start, goal, heuristics):

    def search(path, g, threshold):
        current_node = path[-1]
        f = g + heuristics[current_node]

        if f > threshold:
            return f

        if current_node == goal:
            return path

        min_threshold = float('inf')

        for neighbor, cost in graph.get(current_node, {}).items():
            if neighbor not in path:  # avoid cycles
                result = search(path + [neighbor], g + cost, threshold)

                if isinstance(result, list):
                    return result

                min_threshold = min(min_threshold, result)

        return min_threshold

    threshold = heuristics[start]

    while True:
        result = search([start], 0, threshold)

        if isinstance(result, list):
            return result

        if result == float('inf'):
            return None

        threshold = result
```

The time complexity is $O(b^d)$, but in practice it is higher than A* because nodes can be expanded multiple times. The space complexity is $O(d)$ because the algorithm only uses the recursive stack specific to DFS.
