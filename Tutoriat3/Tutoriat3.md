# Genetic Algorithms

A Genetic Algorithm is a population-based evolutionary optimization technique inspired by the principles of natural selection and genetics. It is the boundary between searching algorithms and machine learning.

Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems. More generically, they are used to find optimal solutions for complex, non-linear mathematical problems.

## The components of a genetic algorithm

- **Gene**: the smallest unit of information (usually a bit, but can be a real number as well)

- **Chromosome**: a candidate solution to the problem. It can also be defined as an oredered set of elements (or genes) whose values determine the characteristics of an individual.
  For example, it can be a sequence of bits: 1011100100

- **Population**: a collection of chromosomes that exist at a particular stage

- **Fitness function**: represents how good a certain chromosome (or solution) is

- **Generation**: a stage in the evolution of the solutions

- **Crossover**: the chromosomes of the new generation inherit their parents' genes

- **Mutation**: some genes of a new individual can be changed

- **Elitism**: a top of very good individuals (in terms of fitness) go to the next generation as well

## Algorithm structure

The code implementation for a genetic algorithm depends on the problem. You can use OOP and create a general class for any genetic algorithm, but there is no need for it. 

```plaintext
Set generation counter t = 0
Initialize Population P(t) by randomly generating individuals from the search space

WHILE the termination condition is NOT met (e.g., target fitness reached or max generations):

    Create an empty population for the next generation: P(t+1)

    // Selection
    // Evaluate fitness of P(t) and choose individuals to be parents
    Intermediate_Population P_select = Select_Parents( P(t) )

    // Crossover 
    // Combine pairs of parents from P_select to create offspring. Depends on the problem.
    Intermediate_Population P_cross = Apply_Crossover( P_select )

    // Mutation
    // Introduce random alterations to the offspring to maintain genetic diversity, with probability prob.
    P(t+1) = Apply_Mutation( P_cross, prob )

    // Elitism (optional)
    // Directly copy the top k individuals from P(t) into P(t+1)
    Append_Elites( P(t), P(t+1), k )

    // Move to the next generation
    t = t + 1

END WHILE

RETURN the individual with the highest fitness score from the final population P(t)
```

# Example

Given $n$ objects, each characterized by a value and a probability (subunitary value) of being transported intact. The goal is to select some objects from the $n$ available to maximize the total value of the shipment, while ensuring a probability of at least $P$ that the entire contents arrive intact at the destination. PS: For any 2 objects, the events of them arriving intact are independent.

**Requirements:** In designing a genetic algorithm for this problem:

---

### a) Chromosome Encoding

Describe what encoding you would use to obtain a chromosome.

- What is the length of the chromosome?
- What would the value of each gene represent?

---

### b) Fitness Function

Describe how you would model a fitness function for this problem.

---

### c) Crossover
For $n = 8$, $P = 1/8$ and the objects with their value and probability of arriving intact are:

(4, 4/5), (6, 3/5), (3, 4/5), (7, 1/3), (2, 7/10), (10, 1/3), (5, 3/4), (3, 2/3)

Generate 2 chromosomes at random (mentioning the value of each) and illustrate the **crossover** (recombination) operation on this pair.

# Solution

### a) Chromosome Encoding

We can use chromosomes of length n, where gene i has value 1 if we take object i and 0 otherwise.

### b) Fitness Function

Let P(C) be the probability that all objects arrive intact in the solution encoded by chromosome C.
Let S(C) be the sum of the values of the objects selected in the solution encoded by chromosome C.
We can use the fitness function f(C) = S(C) if P(C) >= P, otherwise 0.

### c) Crossover

C1 = 01001100

P(C1) = 3/5 * 7/10 * 1/3 = 63/450 = 0.14 >= 1/8

S(C1) = 6 + 2 + 10 = 18

f(C1) = 18

C2 = 11011001

P(C2) = 4/5 * 3/5 * 1/3 * 7/10 * 2/3 = 168/2250 = 0.074(6) < 1/8

f(C2) = 0

We cross the two chromosomes at position 4 and obtain C3 = 01001001
