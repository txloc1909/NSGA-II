import cProfile
import pstats

from zdt1 import ZDT1Solution
from nsga2.nsga2 import NSGA2


def main():
    ZDT1Solution.set_num_variables(20)
    algo = NSGA2(
        solution_type=ZDT1Solution,
        population_size=50,
        n_generations=200,
        crossover_rate=0.25,
        mutation_rate=0.01
    )

    pr = cProfile.Profile()
    pr.enable()
    algo.search()
    pr.disable()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename="profiling.prof")


if __name__ == '__main__':
    main()
