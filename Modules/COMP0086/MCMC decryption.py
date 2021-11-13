import numpy as np
import requests
import re

# we import all the data

message = requests.get('http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/message.txt').text
symbols = requests.get('http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/symbols.txt').text
training = requests.get('https://www.gutenberg.org/files/2600/2600-0.txt').text
training = training.lower()

# we need to find the transition probabilities for each symbol.
def decryption(message, symbols, training):

    def clean_up(symbols, training):
        """
        first we clean up our training text so we don't have any extra symbols that aren't in the list
        This is purely cosmetic, it helps to look at the data
        """
        characters = list(set(training))
        symbols_no_nl = re.sub(r"\r", "", symbols)
        symbols_no_nl = re.sub(r"\n", "", symbols_no_nl)
        symbols_no_nl = re.sub(r"\\", "", symbols_no_nl)
        filtered_training = training

        for char in characters:
            if char not in symbols_no_nl:

                # we remove all the characters that we don't care about
                filtered_training = re.sub(r"{}".format(char), "", filtered_training)

                # we want to remove regex special characters as well
                if filtered_training.count(char) != 0:
                    reg_char = '\\' + char
                    filtered_training = re.sub(reg_char, "", filtered_training)

        print(len(training)-len(filtered_training), "were removed")

        return filtered_training, symbols_no_nl

    def evaluate_transitions(training, symbols):

        # create a 53x53 matrix to store the probabilities
        transition_matrix = np.zeros((len(symbols), len(symbols)))

        i = 0
        for previous_char in symbols:
            j = 0
            for next_char in symbols:
                transition_matrix[i, j] = training.count(previous_char+next_char)
                j += 1
            i += 1

        transition_matrix_norm = transition_matrix
        # we have to normalise our matrix as it only includes counts right now
        for i in range(len(symbols)):
            transition_matrix_norm[i, :] = transition_matrix[i, :]/np.sum(transition_matrix[i, :])

        return transition_matrix_norm

    def markov_chain():
        pass

    training_clean, symbols_clean = clean_up(symbols, training)
    transition_matrix = evaluate_transitions(training_clean, symbols_clean)


if __name__== "__main__":
    decryption(message, symbols, training)