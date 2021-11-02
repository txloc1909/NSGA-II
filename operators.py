import random
import numpy as np


def dominance_operator(x1: np.ndarray, x2: np.ndarray, larger_better=True) -> bool:
    """ Return whether x1 dominate x2 """
    diff = (x1 - x2) if larger_better else (x2 - x1)

    # (x1 is no worse in all objective) && (x1 is strictly better in at least one objective)
    return np.all(diff >= 0) & np.any(diff > 0)


def linear_crossover(x1: np.ndarray,
                     x2: np.ndarray,
                     lb: float = None,
                     ub: float = None) -> (np.ndarray, np.ndarray):
    """ Linear crossover operator for floating number genotype"""
    o1 = np.clip(0.5 * x1 + 0.5 * x2, lb, ub)
    o2 = np.clip(1.5 * x1 - 0.5 * x2, lb, ub)
    o3 = np.clip(1.5 * x2 - 0.5 * x1, lb, ub)

    return random.choices([o1, o2, o3], k=2)


def gaussian_mutation(x: np.ndarray,
                      stddev: float,
                      lb: float,
                      ub: float) -> np.ndarray:
    """ Mutation of real-value genotype using Gaussian distribution """

    return np.clip(x + np.random.normal(0, stddev, size=x.shape), lb, ub)
