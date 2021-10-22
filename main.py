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

GAME_OVER = False
Word = pd.Series
WordsTable = pd.DataFrame
WordsGenome = WordsTable
WordsPopulation = [WordsTable]
last_word: Word


# generate a chain of size elements
def generate_genome_words(size: int, words_table: WordsTable, last_used_word: Word) -> WordsGenome:
    df_accepted: WordsTable
    data = []
    last_selected_word = [str(last_used_word.last_two_letters)]

    for i in range(size):
        current_word_group = words_table.loc[(words_table['first_two_letters'] == last_selected_word[i])]

        preferred_group: WordsTable = current_word_group.loc[(
                (current_word_group.can_connect_with - current_word_group.words_with_same_ending) >= 0)]

        not_preferred_group: WordsTable = current_word_group.loc[
            ((current_word_group.can_connect_with - current_word_group.words_with_same_ending) < 0) &
            (current_word_group.can_connect_with > 0)]

        game_ending_group: WordsTable = current_word_group.loc[(words_table.can_connect_with == 0)]

        if not preferred_group.empty:
            selected = preferred_group.sample(1)
            data.append(selected)
            last_selected_word.append(selected.last_two_letters.values[0])

        elif not not_preferred_group.empty:
            selected = not_preferred_group.sample(1)
            data.append(selected)
            last_selected_word.append(selected.last_two_letters.values[0])

        elif not game_ending_group.empty:
            selected = game_ending_group.sample(1)
            data.append(selected)
            last_selected_word.append(selected.last_two_letters.values[0])

            global GAME_OVER
            GAME_OVER = True
            break  # exit the loop because there are no more words after this

    df_accepted = pd.concat(data, axis=0)

    # If last word cannot connect to anything or if the chain contains duplicates
    # <<< or the length is smaller than expected, re-do the method | (len(df_accepted) != size) >>>
    if (data[len(data) - 1].can_connect_with.values[0] <= 0) | (len(df_accepted[df_accepted.duplicated()]) > 0):
        generate_genome_words(size, words_table, last_used_word)

    return df_accepted


def print_genome_info(words_genome: WordsGenome):
    print(f"length of words_genome {len(words_genome)}")
    print(f"last word is \n {words_genome.tail(1).loc[:, ['word', 'last_two_letters', 'can_connect_with']]}")


def generate_population_words(population_limit: int, genome_size: int, words_table: WordsTable,
                              last_used_word: Word) -> WordsPopulation:
    return [generate_genome_words(genome_size, words_table, last_used_word) for _ in range(population_limit)]


def get_fitness_score_for_genome(words_genome: WordsGenome, wanted_genome_size: int):
    size_fitness_score = 1
    continue_fitness = 0
    # calculate
    words_genome['connectability'] = (words_genome.can_connect_with - words_genome.words_with_same_ending)
    # 1.std per word (can_connect_with - words_ending_with)
    rel_std = words_genome.std(axis=0, numeric_only=True).connectability / words_genome.median(axis=0,
                                                                                               numeric_only=True).connectability
    std_score = 1 / rel_std
    # 2.if size is equal to wanted_size give 2, else give 1
    if len(words_genome) > wanted_genome_size:
        size_fitness_score = 3
    elif len(words_genome) == wanted_genome_size:
        size_fitness_score = 2

    # 3.If last can_connect_to is zero, give 0
    if words_genome.iloc[len(words_genome) - 1].can_connect_with > 0:
        continue_fitness = 1

    final_fitness_score = std_score * size_fitness_score * continue_fitness
    # print(f"Fitness = {final_fitness_score}")
    # plot_fitness(words_genome)
    return final_fitness_score


def plot_fitness(words_genome):
    x_value = range(len(words_genome))
    y_value_ending = words_genome['words_with_same_ending']
    y_value_connecting = words_genome['can_connect_with']

    plt.plot(x_value, y_value_ending,
             label='words_with_same_ending')
    plt.plot(x_value, y_value_connecting, label='can_connect_with')

    plt.title("fitness")
    plt.xlabel('All words')
    plt.ylabel('Number')
    plt.legend()
    plt.show()


def single_point_crossover(a: WordsGenome, b: WordsGenome) -> Tuple[WordsGenome, WordsGenome]:
    # check if crossover is possible
    cutting_places = (
        a.loc[:, 'first_two_letters'].isin(b.first_two_letters))
    word_chosen_from_a = a[cutting_places].sample(1)
    cutting_a_at_index = word_chosen_from_a.index[0]

    first_part_a: DataFrame = a.loc[:cutting_a_at_index, :]
    first_part_a = first_part_a.iloc[:-1, :]
    # remove the last word because it will be included in the next line

    second_part_a = a.loc[cutting_a_at_index:, :]
    word_chosen_from_b = b.loc[(b["first_two_letters"] == a.loc[cutting_a_at_index].first_two_letters)]
    cutting_b_at_index = word_chosen_from_b.head(1).index[0]

    first_part_b: DataFrame = b.loc[:cutting_b_at_index, :]
    first_part_b = first_part_b.iloc[:-1, :]
    second_part_b = b.loc[cutting_b_at_index:, :]

    a_b = first_part_a.append(second_part_b)
    b_a = first_part_b.append(second_part_a)

    if (len(a_b[a_b.duplicated()]) > 0) | (len(b_a[b_a.duplicated()]) > 0):
        return a, b

    return a_b, b_a


def run_evolution(population_limit: int,  # How many chains
                  genome_size: int,  # How many words are in a chain of words
                  words_table: WordsTable,
                  last_used_word: Word,
                  generation_limit: int = 20,  # generation limit is how many times the foor loop is mutating and
                  # searching for the chain with best fitnesss
                  fitness_limit: int = 2.8):
    population = generate_population_words(population_limit, genome_size, words_table, last_used_word)
    i: int = 0
    for i in range(generation_limit):
        i = i
        population = sorted(
            population,
            key=lambda genome: get_fitness_score_for_genome(genome, wanted_genome_size=genome_size),
            reverse=True
        )
        if get_fitness_score_for_genome(population[0], wanted_genome_size=genome_size) >= fitness_limit:
            break
        next_generation = population[0:2]  # I am taking two top fit genoms into next generation
        for j in range(int(len(population) / 2) - 1):
            parents = selection_pair(population, genome_size)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            # Drop in mutation func?
            next_generation += [offspring_a, offspring_b]
            print(f"fitness parent a == {get_fitness_score_for_genome(parents[0], wanted_genome_size=genome_size)}")
            print(f"fitness parent b ==  {get_fitness_score_for_genome(parents[0], wanted_genome_size=genome_size)}")
            print(f"fitness child a == {get_fitness_score_for_genome(offspring_a, wanted_genome_size=genome_size)}")
            print(f"fitness child b == {get_fitness_score_for_genome(offspring_b, wanted_genome_size=genome_size)}")
            print("")
        population = next_generation

    population = sorted(
        population,
        key=lambda genome: get_fitness_score_for_genome(genome, wanted_genome_size=genome_size),
        reverse=True
    )
    return population[0], (i + 1)  # return best fitted genome and number of number_of_generations


def selection_pair(population: WordsPopulation, wanted_genome_size: int) -> WordsPopulation:
    return choices(population=population,
                   weights=[get_fitness_score_for_genome(genome, wanted_genome_size=wanted_genome_size) for genome in
                            population], k=2)


if __name__ == '__main__':
    df_all_words = pd.read_csv('words_filled.csv')

    df_words = df_all_words
    last_word: Word = df_words.loc[16, :]

    while (GAME_OVER != True):
        best_genome, number_of_generations = run_evolution(population_limit=10,
                                                           genome_size=40,
                                                           words_table=df_words,
                                                           last_used_word=last_word)
        last_word = best_genome.iloc[(len(best_genome) - 1)]

        # Umanjit df_words za listu populacije
        rows_to_keep = [x for x in df_words.index if x not in best_genome.index]
        df_words = df_words.loc[rows_to_keep, :]

        gr_end = best_genome.groupby(['last_two_letters'])[['first_two_letters', 'last_two_letters']]
        all_ending_letters = gr_end.first()['last_two_letters']  # This is Series

        for w in all_ending_letters:
            all_first_two_same = df_words[df_words["first_two_letters"] == w]
            df_words.loc[df_words["last_two_letters"] == w, 'can_connect_with'] = len(all_first_two_same)
            df_words.loc[df_words["last_two_letters"] == w, 'words_with_same_ending'] = len(
                df_words[df_words["last_two_letters"] == w])

        print(f"number of generations: {number_of_generations}")
        print(f"Best solution: {best_genome}")
