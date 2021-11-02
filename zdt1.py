import math
import random
import numpy as np
from matplotlib import pyplot as plt

from nsga2 import NSGA2Solution, NSGA2
from nsga2.solutions import MonoRepresentationSolution

from operators import linear_crossover, gaussian_mutation, dominance_operator


class ZDT1Solution(NSGA2Solution, MonoRepresentationSolution):
    n_variables: int = 10  # default value

    @classmethod
    def set_num_variables(cls, n):
        cls.n_variables = n

    @classmethod
    def calculate_fitness(cls, phenotype: np.ndarray) -> np.ndarray:
        f_1 = phenotype[0]
        g = 1.0 + 9 * sum(phenotype[1:]) / (cls.n_variables - 1)
        h = 1.0 - math.sqrt(f_1 / g)
        f_2 = g * h
        return np.array([f_1, f_2])

    @classmethod
    def crossover(cls, parent1, parent2):
        gene1 = parent1.genotype
        gene2 = parent2.genotype
        new1, new2 = linear_crossover(gene1, gene2, 0.0, 1.0)
        return cls.from_genotype(new1), cls.from_genotype(new2)

    @classmethod
    def mutate(cls, parent):
        new = gaussian_mutation(parent.genotype, 0.5, 0.0, 1.0)
        return cls.from_genotype(new)

    @classmethod
    def random_init(cls):
        random_gene = np.random.uniform(0, 1, size=(cls.n_variables,))
        return cls.from_genotype(random_gene)

    def dominate(self, another: NSGA2Solution):
        return dominance_operator(self.fitness, another.fitness, larger_better=False)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    ZDT1Solution.set_num_variables(5)
    nsga_ii = NSGA2(
        solution_type=ZDT1Solution,
        population_size=30,
        n_generations=100,
        crossover_rate=0.30,
        mutation_rate=0.05
    )

    pareto_set = nsga_ii.search()

    randomized = [ZDT1Solution.random_init() for _ in range(200)]
    f1_rand = [s.fitness[0] for s in randomized]
    f2_rand = [s.fitness[1] for s in randomized]

    f1 = [solution.fitness[0] for solution in pareto_set]
    f2 = [solution.fitness[1] for solution in pareto_set]

    x = np.linspace(0, 1, 100)
    y = 1 - np.sqrt(x)
    plt.plot(x, y, c='red', label='true front')

    plt.scatter(f1, f2, c='blue', label='nsga_ii front')
    plt.scatter(f1_rand, f2_rand, c='green', label='random')
    plt.legend()
    plt.xlabel('f_1')
    plt.ylabel('f_2')
    plt.title('ZDT1 Objective Functions (no. variables = %d)' % ZDT1Solution.n_variables)
    plt.show()
