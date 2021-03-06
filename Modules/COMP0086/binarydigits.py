"""
Assignment 1, Unsupervised Learning, UCL 2003
Author: Zoubin Gahramani
Ported to Python by Raza Habib and Jamie Townsend 2017
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.special as sp


# Python comments use a hash

def main():
    # load the data set
    Y = np.loadtxt(os.path.join(r"C:\Users\theod\Desktop\UCL\Machine learning\COMP0086", 'binarydigits.txt'))
    N, D = Y.shape
    """
    MLE
    """
    prob_vec = []
    for d in range(D):
        pixel_data = Y[:, d]
        # since the entries are either 1 or 0, we can sum all the entries for each pixel and average them
        ones_count = np.sum(pixel_data)
        # we now have a 64 vector where each entry is the average "light" of that specific pixel across all images
        prob_vec.append(ones_count / len(pixel_data))
    print(prob_vec)
    # Showing the average picture
    plt.figure()
    plt.title("MLE estimate")
    np.array(prob_vec)
    plt.imshow(np.reshape(prob_vec, (8, 8)), interpolation="None", cmap="gray")
    plt.axis("off")

    """
    MAP
    """
    a = 2
    b = 2
    # Finding the MAP parameter for each pixel across all images
    prob_vec = []
    for d in range(D):
        pixel_data = Y[:, d]
        # since the entries are either 1 or 0, we can sum all the entries for each pixel and average them
        ones_count = np.sum(pixel_data)
        # From the formula derived in the written exercise
        prob_vec.append((ones_count + a-1) / (len(pixel_data) + a+b-2))
    print(prob_vec)
    # Showing the average picture
    plt.figure()
    plt.title("MAP estimate")
    np.array(prob_vec)
    plt.imshow(np.reshape(prob_vec, (8, 8)), interpolation="None", cmap="gray")
    plt.axis("off")

    # this is how you display one image using matplotlib,
    # e.g. the 4th image:
    y = Y[3, :]
    plt.figure()
    plt.imshow(np.reshape(y, (8, 8)),
               interpolation="None",
               cmap='gray')
    plt.axis('off')

    # now we will display the whole data set:
    plt.figure(figsize=(5, 5))
    for n in range(N):

        plt.subplot(10, 10, n + 1)
        plt.imshow(np.reshape(Y[n, :], (8, 8)),
                   interpolation="None",
                   cmap='gray')
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
