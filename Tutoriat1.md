# Prezentarea domeniului

Domeniul inteligenței artificiale este împărțit în mai multe subdomenii fundamentale, printre care:

- **Reprezentarea cunoștințelor (KR - Knowledge Representation)** (master)
- **Rezolvarea Problemelor și căutare (PS - Problem Solving & Search)**
- **Învățare automată (ML - Machine Learning)**

La rândul său, ML-ul este împărțit în trei paradigme principale:

- **Învățare supervizată:** unde modelul învață după niște exemple de antrenare etichetate.
- **Învățare nesupervizată:** axată pe identificarea structurilor și tiparelor ascunse în date neetichetate.
- **Învățare prin recompensă (Reinforcement Learning):** axată pe optimizarea deciziilor unui agent care interacționează cu un mediu pe baza unui sistem de recompense.

Cea mai de succes paradigmă este **învățarea supervizată**, care este și ea împărțită în două direcții:

- *Clasificare:* unde modelul învață să asocieze datele de intrare unui număr finit de tipuri/clase de obiecte.
- *Regresie:* unde modelul învață să prezică o valoare numerică continuă (o funcție).

Datorită succesului acestor algoritmi, ML-ul a generat subdomenii aplicate extrem de complexe, dedicate procesării unor tipuri specifice de date nestructurate:

- **Procesarea limbajului natural (NLP - Natural Language Processing):** utilizarea modelelor de ML pentru a înțelege, interpreta și genera text sau vorbire umană.
- **Vederea Artificială (CV - Computer Vision):** utilizarea modelelor de ML pentru a extrage informații, a recunoaște obiecte sau tipare vizuale din imagini digitale și conținut video.

# Algoritmi inteligenți de pathfinding

Pentru ca un agent să găsească drumul cel mai scurt până la o destinație într-o anumită problemă, el poate folosi cunoștințe specifice problemei (pe lângă descrierea formală a acesteia).

Astfel, el utilizează niște funcții numite *euristici* (estimări ale costului drumului rămas de parcurs până la final), cu scopul de a reduce numărul de stări explorate.

Notând funcția euristică cu h (de la *heuristic*), aceasta are următoarele proprietăți fundamentale:

- h(end) = 0 (costul estimat de la destinație la destinație este zero)
- h(x) ≥ 0 (distanța estimată nu poate fi negativă)
- **Admisibilitate:** h(x) ≤ h*(x), unde h*(x) este costul real minim de la nodul x la destinație (euristica nu supraestimează niciodată costul rămas)
- **Consistență (Monotonie):** h(x) ≤ c(x, y) + h(y), unde c(x, y) este costul real al tranziției de la nodul x la succesorul său y (respectă inegalitatea triunghiului)

**Exemple de euristici:** distanța euclidiană (pentru mișcare în orice unghi), distanța Manhattan (pentru mișcare strict pe orizontală și verticală într-un grid) sau distanța Cebîșev (pentru mișcare în 8 direcții, inclusiv pe diagonală).

## Best-First Search

Acest algoritm reprezintă o abordare de căutare informată. Diferă de Breadth-First Search (BFS) prin faptul că nu extinde nodurile orbește (nivel cu nivel), ci utilizează o funcție euristică $h(n)$ pentru a prioritiza explorarea. 

La fiecare pas, algoritmul selectează pentru expandare nodul $n$ care are cea mai mică valoare estimată până la destinație. Funcția sa de evaluare este strict euristica:
$f(n) = h(n)$

### Implementare în Python

Algoritmul utilizează o coadă de priorități (Min-Heap) pentru a extrage eficient nodul cu cea mai mică valoare euristică.

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

Complexitatea timp este de $O(b^m)$, unde b este numărul maxim de succesori ai unui nod, iar m este adâncimea maximă a spațiului de căutare. În cazul în care euristica este perfectă, complexitatea este $O(b*d)$, unde d este adâncimea soluției. Această reducere drastiă a complexității se datorează faptului că nu mai este nevoie de niciun pas de backtracking, agentul găsind întotdeauna nodul optim.

Complexitatea spațiu este de $O(b^m)$ în cel mai rău caz, deoarece algoritmul trebuie să mențină în memorie toate nodurile generate.

## A*

Acest algoritm reprezintă o abordare de căutare informată și este considerat unul dintre cei mai eficienți algoritmi de pathfinding. Diferă de Greedy Best-First Search prin faptul că ia în calcul atât costul real acumulat de la nodul de start, cât și estimarea până la destinație.

La fiecare pas, algoritmul selectează pentru expandare nodul $n$ care minimizează funcția de evaluare totală:
$f(n) = g(n) + h(n)$

unde $g(n)$ este costul exact al drumului de la start la nodul $n$, iar $h(n)$ este euristica (costul estimat de la $n$ la destinație). Dacă euristica este admisibilă, A* garantează găsirea drumului de cost minim (optim).

### Implementare în Python

```python
import heapq

def a_star(graph, start, goal, heuristics):

    # priority_queue stocheaza: (f_score, g_score, current_node, path)
    priority_queue = [(heuristics[start], 0, start, [start])]
    visited = set()

    while priority_queue:

        f_score, g_score, current_node, path = heapq.heappop(priority_queue)

        if current_node == goal:
            return path

        if current_node not in visited:
            visited.add(current_node)

            # graful este un dictionar de dictionare: {nod_curent: {vecin: cost_tranzitie}}
            for neighbor, cost in graph.get(current_node, {}).items():
                if neighbor not in visited:
                    new_g_score = g_score + cost
                    new_f_score = new_g_score + heuristics[neighbor]
                    heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor, path + [neighbor]))

    return None
```

Complexitatea timp este $O(b^d)$, iar în cazul în care euristica este perfectă, complexitatea scade la $O(b*d)$. 

Complexitatea spațiu este tot $O(b^d)$.

## IDA*

Acest algoritm (Iterative Deepening A*) reprezintă o variantă optimizată din punct de vedere al memoriei a algoritmului A*. Combină ideea de euristică din A* cu strategia de explorare în adâncime iterativă (Iterative Deepening Depth-First Search).

La fel ca A*, folosește funcția de evaluare:

$f(n) = g(n) + h(n)$

Diferența majoră constă în faptul că IDA* nu păstrează toate nodurile în memorie. În schimb, efectuează căutări DFS limitate de un threshold aplicat asupra valorii $f(n)$.

Se pornește cu un prag inițial:
$threshold = h(start)$

Se face un DFS, dar sunt expandate doar nodurile pentru care $f(n) \leq threshold$.

Dacă soluția nu este găsită pragul este actualizat la cel mai mic $f(n)$ care a depășit limita si algoritmul se reia.

### Implementare în Python:

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
            if neighbor not in path:  # evitam cicluri
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

Complexitatea timp este $O(b^d)$, însă în practică este mai mare decât în cazul A*, deoarece nodurile pot fi expandate de mai multe ori.

Complexitatea spațiu este $O(d)$, deoarece algoritmul folosește doar stiva recursivă specifică DFS.


[English]


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

This algorithm represents an informed search approach and is considered one of the most efficient pathfinding algorithms. It differs from Greedy Best-First Search by taking into account both the actual accumulated cost from the start node and the estimated cost to the destination.At each step, the algorithm selects the node $n$ that minimizes the total evaluation function for expansion:$f(n) = g(n) + h(n)$where $g(n)$ is the exact cost of the path from start to node $n$, and $h(n)$ is the heuristic (estimated cost from $n$ to the destination). If the heuristic is admissible, A* guarantees finding the minimum cost (optimal) path.

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
