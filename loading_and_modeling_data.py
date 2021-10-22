if False:
    def first_approach() -> DataFrame:
        words_table = pd.read_csv('words.csv')

        first_two_letters = words_table.loc[0:, 'word'].str[:2:]
        last_two_letters = words_table.loc[0:, 'word'].str[::-1].str[:2:].str[::-1]

        words_table['first_two_letters'] = first_two_letters
        words_table['last_two_letters'] = last_two_letters
        words_table['words_with_same_start'] = 1
        words_table['words_with_same_ending'] = 1
        words_table['can_connect_with'] = 0
        words_table['is_used'] = False

        gr_start = words_table.groupby(['first_two_letters'])[['first_two_letters', 'last_two_letters']]
        all_starting_letters = gr_start.first()['first_two_letters']  # This is Series

        for w in all_starting_letters:
            filter_words = words_table[words_table["first_two_letters"] == w]
            words_table.loc[words_table["first_two_letters"] == w, 'words_with_same_start'] = len(filter_words)

        gr_end = words_table.groupby(['last_two_letters'])[['first_two_letters', 'last_two_letters']]
        all_ending_letters = gr_end.first()['last_two_letters']  # This is Series

        for w in all_ending_letters:
            all_first_two_same = words_table[words_table["first_two_letters"] == w]
            words_table.loc[words_table["last_two_letters"] == w, 'can_connect_with'] = len(all_first_two_same)
            words_table.loc[words_table["last_two_letters"] == w, 'words_with_same_ending'] = len(
                words_table[words_table["last_two_letters"] == w])

        sum_of_game_ending_words = len(words_table[words_table['can_connect_with'] == 0])
        # words_table.to_csv("words_filled.csv")

        return words_table

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

