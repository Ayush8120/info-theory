"""
This file uses the methods from entropy_models.py to illustrate simple concepts from the 7480 information theory class
First version: 9/15/2024
This version: 9/15/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
Intention is to put those into a notebook later
"""

import numpy as np
from entropy_models import(calculate_entropy,
                           calculate_entropy_vector,
                           plot_figure,
                           markov_chain_stationary,
                           markov_chain_sample)
# from scipy.stats import entropy
# import matplotlib.pyplot as plt
import timeit


def test_calculate_entropy():
    # tests 1d entropy  calculation

    print("\ntest_calculate_entropy():")
    P1 = np.ones(3) / 3
    P2 = np.array([0.25, 0.25, 0.5])
    P3 = np.array([1/8, 1/8, 3/4])
    P4 = np.array([0, 0, 1])

    print(calculate_entropy(P1))
    print(calculate_entropy(P2))
    print(calculate_entropy(P3))
    print(calculate_entropy(P4))


def test_calculate_entropy_vector():
    # tests 2d entropy calculations

    print("\ntest_calculate_entropy_vector():")
    P_vec = np.array([np.ones(3) / 3,
                  [0.25, 0.25, 0.5],
                  [1/8, 1/8, 3/4],
                  [0, 0, 1]])

    print(calculate_entropy_vector(P_vec))


def test_time_calculate_entropy():
    # shows that calculating entropy with numpy natively is quite a bit faster than the more general scipy function

    print("\ntest_time_calculate_entropy():")
    print("Time to evalute via Scipy:", timeit.timeit(stmt="entropy(P3, base=2)",
                        setup="from scipy.stats import entropy; import numpy; P3 = numpy.array([0, 0, 1])",
                        number=10000))

    print("Time to evalute via Numpy:",
          timeit.timeit(stmt="calculate_entropy(P3)",
                        setup="from entropy_models import calculate_entropy; import numpy; P3 = numpy.array([0, 0, 1])",
                        number=10000))


def test_show_entropy_binary():
    # display binary entropy function

    print("\ntest_show_entropy_binary():")
    P = np.linspace(0, 1, 41)
    P_vec = np.array([P, 1-P]).transpose()      # create 2D distribution

    Y = calculate_entropy_vector(P_vec)
    # print(Y)
    plot_figure(P, Y, low=-0.02, high=1.02, title=r"$H(p)$", show_legend=False, fine_grid=True, xlabel=r'$p$',
                ylabel=None, pdfname="Fig_Binary_Entropy", linewidth=(4,), squaresize=True)


def test_markov_chain():
    # Javed's Restaurant Markov chain example from class

    print("\ntest_markov_chain():")
    P_matrix = np.array([[0.7, 0.2, 0.1],
                         [0.3, 0.6, 0.1],
                         [0.3, 0.2, 0.5]])

    P = markov_chain_stationary(P_matrix)
    print(P_matrix)
    print("Stationary distribution:", P)

    st = markov_chain_sample(P_matrix, 1000, int_to_char={0: 'B', 1: 'M', 2: 'S'})
    print("A random sample:")
    print(st)

    print("Entropy of the MC and the Stationary Distribution (SD)")
    entropy_MC = P.dot(calculate_entropy_vector(P_matrix))
    print(entropy_MC)
    entropy_SD = calculate_entropy(P)
    print(entropy_SD)


def test_markov_chain_binary():
    # Binary example

    print("\ntest_markov_chain_binary():")
    p = 0.95

    P_matrix = np.array([[p, 1-p],
                         [1-p, p]])

    P = markov_chain_stationary(P_matrix)
    print(P_matrix)
    print("Stationary distribution:", P)

    st = markov_chain_sample(P_matrix, 1000)
    print("A random sample:")
    print(st)

    print("Entropy of the MC and the Stationary Distribution (SD)")
    entropy_MC = P.dot(calculate_entropy_vector(P_matrix))
    print(entropy_MC)
    entropy_SD = calculate_entropy(P)
    print(entropy_SD)


if __name__ == '__main__':
    # test_calculate_entropy()
    # test_calculate_entropy_vector()
    # test_time_calculate_entropy()
    # test_show_entropy_binary()
    test_markov_chain()
    # test_markov_chain_binary()