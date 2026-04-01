# Genetic Algorithms

A Genetic Algorithm is a population-based evolutionary optimization technique inspired by the principles of natural selection and genetics. It is the boundary between searching algorithms and machine learning.

Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems. More generically, they are used to find optimal solutions for complex, non-linear mathematical problems.

## The components of a genetic algorithm

- **Gene**: the smallest unit of information (usually a bit)

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
