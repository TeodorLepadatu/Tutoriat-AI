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
