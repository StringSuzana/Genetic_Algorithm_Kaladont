from collections import namedtuple
from functools import partial
from typing import List, Callable, Tuple
from random import choices, randint, randrange, random
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas import DataFrame

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

Thing = namedtuple('Thing', ['name', 'value', 'weight'])
things = [
    Thing('Laptop', 500, 2200),
    Thing('Headphones', 150, 160),
    Thing('Mug', 60, 350),
    Thing('Notepad', 40, 333),
    Thing('Bottle', 30, 192)
]
more_things = [
                  Thing('Mints', 5, 25),
                  Thing('Socks', 10, 38),
                  Thing('Tissue', 15, 80),
                  Thing('Phone', 500, 200),
                  Thing('Cap', 100, 70)
              ] + things


def generate_genome(length: int) -> Genome:
    # instead of choices get string list of words connected
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same length")

    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0
    return value


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(population=population, weights=[fitness_func(genome) for genome in population], k=2)


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("a and b must be of the same length")
    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population_loc = populate_func()

    for i in range(generation_limit):
        population_loc = sorted(
            population_loc,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        if fitness_func(population_loc[0]) >= fitness_limit:
            break
        next_generation = population_loc[0:2]
        for j in range(int(len(population_loc) / 2) - 1):
            parents = selection_func(population_loc, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        population_loc = next_generation

    population_loc = sorted(
        population_loc,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population_loc, i


def genome_to_things(genome: Genome, things_list: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things_list):
        if genome[i] == 1:
            result += [thing.name]
    return result


def put_in_main_func_genetic():
    population, generations = run_evolution(
        populate_func=partial(generate_population, size=10, genome_length=len(things)),
        fitness_func=partial(fitness, things=things, weight_limit=3000),
        fitness_limit=740,
        generation_limit=100
    )
    print(f"number of generations: {generations}")
    print(f"Best solution: {genome_to_things(population[0], things)}")


if __name__ == '__main__':
    put_in_main_func_genetic()
