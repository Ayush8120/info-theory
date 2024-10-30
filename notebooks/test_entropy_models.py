"""
This file uses the methods from entropy_models.py to illustrate simple concepts from the 7480 information theory class
First version: 9/15/2024
This version: 10/1/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""

import numpy as np
from entropy_models import(calculate_entropy,
                           calculate_entropy_vector,
                           calculate_relative_entropy,
                           calculate_relative_entropy_vector,
                           plot_figure,
                           markov_chain_stationary,
                           markov_chain_sample)
import timeit

# import matplotlib.pyplot as plt
# from scipy.stats import entropy
# import matplotlib.pyplot as plt


def test_calculate_entropy():
    # tests 1d entropy  calculation
    print("\ntest_calculate_entropy():")

    P = np.ones(3) / 3
    print(calculate_entropy(P))

    P = np.array([0.25, 0.25, 0.5])
    print(calculate_entropy(P))

    P = np.array([1/8, 1/8, 3/4])
    print(calculate_entropy(P))

    P = np.array([0, 0, 1])
    print(calculate_entropy(P))

    P = np.array([1/2, 1/3, 1/6])
    print(calculate_entropy(P))

    P = np.array([0.25, 0.75])
    print(calculate_entropy(P))

    P = np.array([9/14, 5/14])
    print(calculate_entropy(P))

    P = np.array([5 / 14, 4 / 14, 5 / 14])      # Mitchell Tennis example split information Outlook
    print(calculate_entropy(P))

    P = np.array([4 / 14, 6 / 14, 4 / 14])      # Mitchell Tennis example split information Temp
    print(calculate_entropy(P))

    P = np.array([8 / 14, 6 / 14])      # Mitchell Tennis example split information Wind
    print(calculate_entropy(P))

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

    print("Time to evalute via Scipy:",
          timeit.timeit(stmt="entropy(P3, base=2)",
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
    print(Y)
    plot_figure(P, Y, low=-0.02, high=1.02, title=r"$H(p)$", show_legend=False, fine_grid=True, xlabel=r'$p$',
                ylabel=None, pdfname="Fig_Binary_Entropy",
                linewidth=(4,), squaresize=True)
    # TODO: does not display latex correctly


def test_show_coin_flip():
    # Cover, Thomas, Example 2.1
    # A fair coin is flipped until the first head occurs. Let X denote the number of flips required.
    # Find the entropy H(X) in bits, as function of p
    print("\ntest_show_entropy_binary():")

    # P = np.linspace(0.01, 1, 41)
    # P_vec = np.array([P, 1-P]).transpose()      # create 2D distribution
    # Y = calculate_entropy_vector(P_vec)
    # print(Y)
    # Z = Y / P  # actual entropy
    # print(Z)
    # plot_figure(P, Z, low=-0.02, high=1.02, ymax=10, title=r"$H(p)$", show_legend=False, fine_grid=True,
    #             xlabel=r'$p$',
    #             ylabel=None, pdfname="Fig_CoinFlip_Entropy", linewidth=(4,), squaresize=True)

    P = np.logspace(-10, 0, 41)
    # print(P)
    P_vec = np.array([P, 1-P]).transpose()      # create 2D distribution
    Y = calculate_entropy_vector(P_vec)
    # print(Y)
    Z = Y / P  # actual entropy according to formula
    print(Z)

    plot_figure(P, Z, low=-0.02, high=1.02, ymax=10, title=r"$H(p)$", show_legend=False, fine_grid=False,
                xlabel=r'$p$', logscale=True,
                ylabel=None, pdfname="Fig_CoinFlip_Entropy_logscale", linewidth=(4,), squaresize=True)



def test_markov_chain_restaurants():
    # Javed's Restaurant Markov chain example from class
    print("\ntest_markov_chain():")

    P_matrix = np.array([[0.7, 0.2, 0.1],
                         [0.3, 0.6, 0.1],
                         [0.3, 0.2, 0.5]])
    mu = markov_chain_stationary(P_matrix)
    print(P_matrix)

    print("\nRestaurant example:")
    print("Stationary distribution:", mu)

    st = markov_chain_sample(P_matrix, 1000, int_to_char={0: 'B', 1: 'M', 2: 'S'})
    print("A random sample:")
    print(st)

    print("Entropy of the MC and the Stationary Distribution (SD)")
    entropy_MC = mu.dot(calculate_entropy_vector(P_matrix))
    print(entropy_MC)
    entropy_SD = calculate_entropy(mu)
    print(entropy_SD)

    # re-use the code to create a sample with IID sampled letters
    print("\nRestaurant example, assuming IID samples:")
    P_matrix = np.array([mu, mu, mu])
    mu = markov_chain_stationary(P_matrix)
    print(P_matrix)
    print("Stationary distribution:", mu)

    st = markov_chain_sample(P_matrix, 1000, int_to_char={0: 'B', 1: 'M', 2: 'S'})
    print("A random sample:")
    print(st)

    print("Entropy of the MC and the Stationary Distribution (SD)")
    entropy_MC = mu.dot(calculate_entropy_vector(P_matrix))
    print(entropy_MC)
    entropy_SD = calculate_entropy(mu)
    print(entropy_SD)



def test_markov_chain_binary():
    # Binary example

    print("\ntest_markov_chain_binary():")
    p = 0.05

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


def test_calculate_relative_entropy():
    # tests 1d entropy  calculation
    print("\ntest_calculate_relative_entropy():")

    P = np.array([0.5, 0.5])
    Q = np.array([0.5, 0.5])
    print(calculate_relative_entropy(P, Q))

    P = np.array([0.5, 0.5])
    Q = np.array([0.05, 0.95])
    print()
    print(calculate_relative_entropy(P, Q))
    print(calculate_relative_entropy(Q, P))

    P = np.array([0.5, 0.5])
    Q = np.array([0.001, 0.999])
    print()
    print(calculate_relative_entropy(P, Q))
    print(calculate_relative_entropy(Q, P))

    P = np.array([0.5, 0.5])
    Q = np.array([0, 1])
    print()
    print(calculate_relative_entropy(P, Q))
    print(calculate_relative_entropy(Q, P))

    P = np.array([0, 1])
    Q = np.array([0, 1])
    print()
    print(calculate_relative_entropy(P, Q))


def test_calculate_relative_entropy_vector():
    # tests 2d entropy calculations
    print("\ntest_calculate_relative_entropy_vector():")

    P_vec = np.array([[0.5, 0.5],
                      [0.5, 0.5],
                      [0.5, 0.5],
                      [0.5, 0.5],
                      [0, 1],
                      [0, 1],])
    Q_vec = np.array([[0.5, 0.5],
                      [0.05, 0.95],
                      [0.001, 0.999],
                      [0, 1],
                      [0.5, 0.5],
                      [0, 1]])
    print(calculate_relative_entropy_vector(P_vec, Q_vec))

    P_vec = np.array([np.ones(3) / 3,
                      np.ones(3) / 3,
                      np.ones(3) / 3,
                      [1, 0, 0],
                      [1, 0, 0]])
    Q_vec = np.array([np.ones(3) / 3,
                      [0.998, 0.001, 0.001],
                      [1, 0, 0],
                      np.ones(3) / 3,
                      [1, 0, 0]])
    print(calculate_relative_entropy_vector(P_vec, Q_vec))


def test_assymmetric_relative_entropy():
    # shows the asymmetry of KL divergence (relative entropy) with uniform against biased distribution
    print("\ntest_calculate_relative_entropy_vector():")

    P = np.linspace(0, 1, 301)
    P_vec = np.array([P, 1-P]).transpose()      # create 2D distribution
    Q_vec = np.repeat(np.array([[0.5, 0.5]]), 301, axis=0)

    RE1 = calculate_relative_entropy_vector(P_vec, Q_vec)
    RE2 = calculate_relative_entropy_vector(Q_vec, P_vec)

    plot_figure(P, (RE1, RE2), ymax=3.01, low=-0.02, high=1.02, title=r"$D_{\mathrm{KL}}(p||u)$ and $D_{\mathrm{KL}}(u||p)$", show_legend=False, fine_grid=True, xlabel=r'$p$',
                ylabel=None, pdfname="Fig_Relative_Entropy_Assymmetry",
                linewidth=(4,4), squaresize=True)

    P = np.array([0.5, 0.5])
    Q = np.array([0.001, 0.999])
    print()
    print(calculate_relative_entropy(P, Q))
    print(calculate_relative_entropy(Q, P))


if __name__ == '__main__':
    test_calculate_entropy()
    # test_calculate_entropy_vector()
    # test_time_calculate_entropy()
    # test_show_entropy_binary()
    # test_show_coin_flip()
    # test_markov_chain_restaurants()
    # test_markov_chain_binary()
    # test_calculate_relative_entropy()
    # test_calculate_relative_entropy_vector()
    # test_assymmetric_relative_entropy()