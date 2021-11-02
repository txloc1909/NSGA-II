from abc import ABC, abstractmethod
from copy import deepcopy


class GASolution(ABC):
    __slots__ = ('__genotype', '__phenotype', '__fitness')

    def __init__(self, genotype, phenotype):
        self.__genotype = genotype
        self.__phenotype = phenotype
        self.__fitness = self.calculate_fitness(self.__phenotype)

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
        return deepcopy(self.__genotype)

    @property
    def phenotype(self):
        return deepcopy(self.__phenotype)

    @property
    def fitness(self):
        return deepcopy(self.__fitness)


class MonoRepresentationSolution(GASolution, ABC):

    @classmethod
    def genotype_to_phenotype(cls, genotype):
        return genotype

    @classmethod
    def phenotype_to_genotype(cls, phenotype):
        return phenotype
