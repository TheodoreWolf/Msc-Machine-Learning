import numpy as np
import requests
import re
import random
from tqdm import tqdm

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
                # we add +1 everywhere for stability
                transition_matrix[i, j] = training.count(previous_char+next_char)+1
                j += 1
            i += 1

        transition_matrix_norm = transition_matrix
        # we have to normalise our matrix as it only includes counts right now
        for i in range(len(symbols)):
            transition_matrix_norm[i, :] = transition_matrix[i, :]/np.sum(transition_matrix[i, :])

        return transition_matrix_norm

    def stationary_dist(trans_matrix):

        # we want to find the eigenvector of our transition matrix
        w, v = np.linalg.eig(trans_matrix.T)

        # numpy always returns the largest eigenvalue eigenvector first (index=0)
        dist = np.real(v[:, 0])
        return dist/np.sum(dist)

    def generate_new_proposal(sigma):

        # we generate a new proposal sigma
        index1, index2 = random.sample(range(0, 52), 2)
        new_sigma = sigma.copy()

        # the new sigma is identical except for two indices switched around
        i2, i1 = sigma[index1], sigma[index2]
        new_sigma[index1] = i1
        new_sigma[index2] = i2
        return new_sigma

    def decrypt_message(sigma, message, symbols_clean):
        new_message = ""
        for char in message:
            new_message += symbols_clean[sigma[symbols_clean.find(char)]]

        return new_message

    def evaluate_probs_text(sigma, original_message, transition_matrix, stationary_distribution, symbols_clean):

        new_message = decrypt_message(sigma, original_message, symbols_clean)
        log_prob = np.log(stationary_distribution[symbols_clean.find(new_message[0])])

        for i, char in enumerate(new_message):
            prev_char = new_message[i - 1]
            char_index = symbols_clean.find(char)
            prev_char_index = symbols_clean.find(prev_char)
            log_prob += np.log(transition_matrix[prev_char_index, char_index])

        return log_prob

    def MH_step(pi_dist, tran_matrix, s, m, symbols_clean):

        # we generate a new random sigma
        new_s = generate_new_proposal(s)
        accepted = False

        # we evaluate its log probability
        log_text_prob_new = evaluate_probs_text(new_s, m, tran_matrix, pi_dist, symbols_clean)
        log_text_prob = evaluate_probs_text(s, m, tran_matrix, pi_dist, symbols_clean)

        # we compute the decision kernel
        weighted_prob = log_text_prob_new - log_text_prob
        rejection_kernel = np.exp(min(0, weighted_prob))

        # we accept if it is larger than a random number from [0,1]
        if rejection_kernel > np.random.rand():
            s = new_s
            log_text_prob = log_text_prob_new
            accepted = True

        return s, log_text_prob, accepted

    training_clean, symbols_clean = clean_up(symbols, training)
    transition_matrix = evaluate_transitions(training_clean, symbols_clean)
    stat_dist = stationary_dist(transition_matrix)

    # we create the sigma vector, it is simply 0-52 corresponding to the order of the letters in the file given
    sigma = [i for i in range(53)]
    random.shuffle(sigma)
    acceptation = []
    # we decide the number of iterations
    N = 50000
    for n in tqdm(range(N)):

        # we do a step of MH

        new_sigma, prob, accept = MH_step(stat_dist, transition_matrix, sigma, message, symbols_clean)

        acceptation.append(accept)

        # we use the new sigma
        sigma = new_sigma

        if n % 1000 == 0:
            print("Likelihood: {:.2f}, accept={:.4f}".format(prob,
                                                              sum(acceptation)/len(acceptation)),
                                                              decrypt_message(sigma, message, symbols_clean))

    print(decrypt_message(sigma, message, symbols_clean))

if __name__== "__main__":
    decryption(message, symbols, training)