from abc import ABC, abstractmethod
from copy import deepcopy


class GASolution(ABC):
    """ Abstract base class for Genetic Algorithm solution

        A solution has three essential elements:
        * Genotype: gene representation of solution. Used in genetic operators (crossover, mutate,...)
        * Phenotype: problem-specific representation of solution. Determine solution's objective value
        * Fitness: the objective value

        A solution instance should be immutable. This means when a solution is created,
        it already has three elements above figured out, and doesn't change throughout its lifetime.

        This abstract base class allows implementing a custom genotype-phenotype mapping.
    """
    __slots__ = ('_genotype', '_phenotype', '_fitness')

    def __init__(self, genotype, phenotype):
        self._genotype = genotype
        self._phenotype = phenotype
        self._fitness = self.calculate_fitness(self._phenotype)

    @classmethod
    @abstractmethod
    def calculate_fitness(cls, phenotype):
        pass

    @classmethod
    @abstractmethod
    def genotype_to_phenotype(cls, genotype):
        pass

    @classmethod
    @abstractmethod
    def phenotype_to_genotype(cls, phenotype):
        pass

    @classmethod
    @abstractmethod
    def crossover(cls, parent1, parent2):
        pass

    @classmethod
    @abstractmethod
    def mutate(cls, parent):
        pass

    @classmethod
    def from_phenotype(cls, phenotype):
        genotype = cls.phenotype_to_genotype(phenotype)

        return cls(genotype, phenotype)

    @classmethod
    def from_genotype(cls, genotype):
        phenotype = cls.genotype_to_phenotype(genotype)

        return cls(genotype, phenotype)

    @classmethod
    @abstractmethod
    def random_init(cls):
        pass

    @property
    def genotype(self):
        return deepcopy(self._genotype)

    @property
    def phenotype(self):
        return deepcopy(self._phenotype)

    @property
    def fitness(self):
        return deepcopy(self._fitness)


class MonoRepresentationSolution(GASolution, ABC):
    """ Abstract base class for solution
        which its genotype and phenotype are identical

        Created in order to reduce boilerplate code.
    """

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        return genotype

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        return phenotype
