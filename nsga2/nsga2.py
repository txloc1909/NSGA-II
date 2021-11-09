import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .solutions import GASolution


class NSGA2Solution(GASolution, ABC):
    """ Abstract base class for multi-objective GA solution, or NSGA2's solution """

    @abstractmethod
    def dominate(self, another: 'NSGA2Solution'):
        pass


@dataclass
class NonDominatedSortItem(object):
    """ Wrapper class around NSGA2Solution
        Allow an immutable NSGA2Solution to have states.
        These states are used in order to perform fast non-dominated sort.
    """
    __slots__ = ('solution', 'n_dominated', 'dominating_set')
    solution: NSGA2Solution
    n_dominated: int
    dominating_set: List['NonDominatedSortItem']


def fast_non_dominated_sort(population: List[NSGA2Solution]) -> List[List[NSGA2Solution]]:
    assert len(population) > 0
    sorting_items = [NonDominatedSortItem(solution, 0, []) for solution in population]
    fronts: List[List[NonDominatedSortItem]] = [[]]

    for item1 in sorting_items:
        for item2 in sorting_items:
            if item1 is item2:
                continue

            if item1.solution.dominate(item2.solution):
                item1.dominating_set.append(item2)
            elif item2.solution.dominate(item1.solution):
                item1.n_dominated += 1

        if item1.n_dominated == 0:
            fronts[0].append(item1)

    i = 0
    while True:
        new_front = []
        for dominating_item in fronts[i]:
            for dominated_item in dominating_item.dominating_set:
                dominated_item.n_dominated -= 1

                if dominated_item.n_dominated == 0:
                    new_front.append(dominated_item)

        if not new_front:
            break

        i += 1
        fronts.append(new_front)

    # unwrap fronts to return the actual solutions
    return [[item.solution for item in front] for front in fronts]


def crowding_distance_sorting(population: List[NSGA2Solution]) -> List[NSGA2Solution]:
    assert len(population) > 0

    pop_size = len(population)
    n_objectives = len(population[0].fitness)
    crowding_distance = np.zeros((n_objectives, pop_size), dtype=np.float)

    for obj in range(n_objectives):
        distance = crowding_distance[obj]
        sorted_pop = sorted(np.arange(pop_size), key=lambda j: population[j].fitness[obj])
        id_min = sorted_pop[0]
        id_max = sorted_pop[-1]
        obj_min = population[id_min].fitness[obj]
        obj_max = population[id_max].fitness[obj]

        distance[id_min] = np.inf
        distance[id_max] = np.inf
        for i in sorted_pop[1:-1]:
            next_sorted = sorted_pop[sorted_pop.index(i) + 1]
            prev_sorted = sorted_pop[sorted_pop.index(i) - 1]
            distance[i] = (population[next_sorted].fitness[obj] - population[prev_sorted].fitness[obj])
            distance[i] = distance[i] / (obj_max - obj_min + 1e-6)  # plus 1e-6 to avoid division by zero

    crowding_distance = np.sum(crowding_distance, axis=0)
    sorted_by_total_crowding = np.argsort(crowding_distance)

    return [population[i] for i in sorted_by_total_crowding[::-1]]


class NSGA2:

    def __init__(self,
                 solution_type: type(NSGA2Solution),
                 population_size: int,
                 n_generations: int,
                 crossover_rate: float,
                 mutation_rate: float):
        self.solution_type = solution_type
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self) -> List[NSGA2Solution]:
        return [self.solution_type.random_init() for _ in range(self.population_size)]

    def do_crossover(self, current_pop) -> (Optional[NSGA2Solution], Optional[NSGA2Solution]):
        assert current_pop

        prob = np.random.uniform()
        if prob > self.crossover_rate:
            return None, None

        n = len(current_pop)
        i, j = 0, 0
        while i == j:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)

        parent1 = current_pop[i]
        parent2 = current_pop[j]
        return self.solution_type.crossover(parent1, parent2)

    def do_mutation(self, current_pop) -> Optional[NSGA2Solution]:
        assert current_pop

        prob = np.random.uniform()
        if prob > self.mutation_rate:
            return None

        parent = random.choice(current_pop)
        return self.solution_type.mutate(parent)

    def reproduce(self, current_pop) -> List[NSGA2Solution]:
        offsprings_pop: List[NSGA2Solution] = []

        for _ in range(self.population_size):
            o1, o2 = self.do_crossover(current_pop)
            if o1 is not None and o2 is not None:
                offsprings_pop.append(o1)
                offsprings_pop.append(o2)

            o3 = self.do_mutation(current_pop)
            if o3 is not None:
                offsprings_pop.append(o3)

        return offsprings_pop

    def search(self):
        population = self.initialize_population()

        for _ in range(self.n_generations):
            assert len(population) == self.population_size

            offspring_pop = self.reproduce(population)
            fronts = fast_non_dominated_sort(population + offspring_pop)

            population = []
            i = 0
            while len(population) + len(fronts[i]) < self.population_size:
                population += fronts[i]
                i += 1

            last_front = crowding_distance_sorting(fronts[i])
            population += last_front[0:(self.population_size - len(population))]

        final_fronts = fast_non_dominated_sort(population)

        return final_fronts[0]
