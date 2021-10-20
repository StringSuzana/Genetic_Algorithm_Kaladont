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

POPULATION_LIMIT = 50
GAME_OVER = False
Word = pd.Series
WordsTable = pd.DataFrame
WordsGenome = WordsTable
WordsPopulation = [WordsTable]
last_word: Word


def generate_genome_words(size: int, words_table: WordsTable, last_used_word: Word) -> WordsGenome:
    # generate a chain of size elements
    df_accepted: WordsTable
    data = []
    last_selected_word = [str(last_used_word.last_two_letters)]
    print(last_selected_word)

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
    # <<< or the chain could not be finished  | (len(df_accepted[df_accepted.duplicated()]) > 0) >>>
    # <<< or the length is smaller than expected, re-do the method | (len(df_accepted) != size) >>>
    if data[len(data) - 1].can_connect_with.values[0] <= 0:
        generate_genome_words(size, words_table, last_used_word)

    # print_genome_info(df_accepted)
    # get_fitness_score_for_words(df_accepted, size)

    return df_accepted


def print_genome_info(words_genome: WordsGenome):
    print(f"length of words_genome {len(words_genome)}")
    print(f"last word is \n {words_genome.tail(1).loc[:, ['word', 'last_two_letters', 'can_connect_with']]}")


def generate_population_words(population_limit: int, genome_size: int, words_table: WordsTable,
                              last_used_word: Word) -> WordsPopulation:
    return [generate_genome_words(genome_size, words_table, last_used_word) for _ in range(population_limit)]


def get_fitness_score_for_words(words_genome: WordsGenome, wanted_size: int):
    size_fitness_score = 1
    continue_fitness = 0
    # calculate
    words_genome['connectability'] = words_genome.can_connect_with - words_genome.words_with_same_ending
    # 1.std per word (can_connect_with - words_ending_with)
    std_score = words_genome.std(axis=0, numeric_only=True).connectability / words_genome.median(axis=0,
                                                                                                 numeric_only=True) \
        .connectability
    # 2.if size is equal to wanted_size give 2, else give 1
    if len(words_genome) == wanted_size:
        size_fitness_score = 2

    # 3.If last can_connect_to is zero, give 0
    if words_genome.iloc[len(words_genome) - 1].can_connect_with > 0:
        continue_fitness = 1

    final_fitness_score = std_score * size_fitness_score * continue_fitness
    print(f"Fitness = {final_fitness_score}")
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


def plot_each_letters_group(words_table):
    gr_start = words_table.groupby(['first_two_letters'])[['first_two_letters', 'last_two_letters']]
    all_starting_letters = gr_start.first()['first_two_letters']  # This is Series

    for w in all_starting_letters:
        x_value = range(len(words_table[words_table["first_two_letters"] == w]))
        y_value_ending = words_table[words_table["first_two_letters"] == w]['words_with_same_ending']
        y_value_connecting = words_table[words_table["first_two_letters"] == w]['can_connect_with']

        plt.plot(words_table[words_table["first_two_letters"] == w].index, y_value_ending,
                 label='words_with_same_ending')
        plt.plot(words_table[words_table["first_two_letters"] == w].index, y_value_connecting, label='can_connect_with')

        plt.title(w)
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
    first_part_a.iloc[:-1, :]  # remove the last word because it will be included in the next line

    second_part_a = a.loc[cutting_a_at_index:, :]

    cutting_b_at_index = b.loc[(b["first_two_letters"] == a.loc[cutting_a_at_index].first_two_letters)].index[0]
    first_part_b: DataFrame = b.loc[:cutting_b_at_index, :]
    first_part_b.iloc[:-1, :]
    second_part_b = b.loc[cutting_b_at_index:, :]

    a_b = first_part_a.append(second_part_b)
    b_a = first_part_b.append(second_part_a)

    return a_b, b_a


def run_evolution(population_limit: int, genome_size: int, words_table: WordsTable,
                  last_used_word: Word):
    population = generate_population_words(population_limit, genome_size, words_table, last_used_word)
    genome_sum = population_limit * genome_size
    for i in range(genome_sum):
        population = sorted(
            population,
            key=lambda genome: get_fitness_score_for_words(genome, wanted_size=genome_size),
            reverse=True
        )
        both = single_point_crossover(population[0], population[1])
        print(population)
    pass


if __name__ == '__main__':
    df_words = pd.read_csv('words_filled.csv')
    last_word: Word = df_words.loc[16, :]
    run_evolution(10, 50, df_words, last_word)
    generate_genome_words(50, df_words, last_word)
