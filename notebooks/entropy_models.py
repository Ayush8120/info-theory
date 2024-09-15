"""
This file defines various methods to illustrate simple concepts from the 7480 information theory class
Also includes a standardized way to show plots
First version: 9/15/2024
This version: 9/15/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# from scipy.special import expit
# import matplotlib.pyplot as plt


def calculate_entropy(P):
    """
    calculate Shannon entropy for a 1-dimensional probability distribution given as Numpy array
    """
    assert type(P).__module__ == "numpy"
    assert P.ndim == 1
    assert np.all(P >= 0)
    assert math.isclose(np.sum(P), 1, abs_tol=1e-5)

    with np.errstate(divide='ignore'):  # most efficient to temporarily ignore the inf error warning, also allows simple generalization to multi-dimensional arrays
        res = -np.log2(P)               # https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
    res[np.isinf(res)] = 0
    return P.dot(res)

    # # IGNORE BELOW: Following is a less efficient code
    # P = P[np.nonzero(P)]                    # remove the 0's, to avoid divide by zero error for log
    # return P.dot(np.log2(1/P))              # log2(1/P) instead of -log2(P) to avoid -0 instead of 0 for deterministic distribution
    # # return np.nansum(P*(-np.log2(P)), axis=0)     # alternative that avoids -0 by ignores -0's / entrywise multiplication


def calculate_entropy_vector(P_vec):
    """
    calculate Shannon entropy for a 2-dimensional vector of probability distribution as Numpy array
    Example P_vec input:
        P_vec = np.array([[0.25, 0.25, 0.5],
                          [1/8, 1/8, 3/4]])
    """
    assert type(P_vec).__module__ == "numpy"
    assert P_vec.ndim == 2
    assert np.all(P_vec >= 0)
    np.testing.assert_allclose(np.sum(P_vec, axis=1), 1, atol=1e-5)

    with np.errstate(divide='ignore'):  # https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
        res = -np.log2(P_vec)  # most efficient to temporarily ignore the inf error warning, also allows simple generalization to multi-dimensional arrays
    res[np.isinf(res)] = 0
    return np.nansum(P_vec * res, axis=1)  # allows generalization to 2D arrays


def markov_chain_stationary(P_matrix):
    """
    calculate stationary distribution for k-k-dimensional stochastic row matrix
        P_matrix = np.array([[0.8, 0.2],
                             [0.7, 0.3]])
    """
    assert type(P_matrix).__module__ == "numpy"
    assert P_matrix.ndim == 2
    assert np.all(P_matrix >= 0)
    (k, m) = P_matrix.shape
    assert k == m
    np.testing.assert_allclose(np.sum(P_matrix, axis=1), 1, atol=1e-5)

    P = np.ones(k) / k
    for i in range(30):
        P = P_matrix.transpose().dot(P)
    return P


def markov_chain_sample(P_matrix, n, int_to_char=None):
    """
    create a string of choices given a Markov Chain
    int_to_char by default maps 0->A, 1->B, 2->C, ...
    """
    P = markov_chain_stationary(P_matrix)   # stationary distribution

    st = []
    k = P.size
    s = np.random.choice(k, p=P)
    st.append(s)
    for i in range(n):
        s = np.random.choice(k, p=P_matrix[s, :])   # sample from the s-th row in the transition matrix
        st.append(s)

    if int_to_char is None:
        int_to_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    char_list = [int_to_char[num] for num in st]
    result_string = ''.join(char_list)

    return result_string



def plot_figure(x, y, low, high,
                title=None, pdfname=None, linewidth=None, label=None,
                logscale=False, linestyle=None, marker=None, markevery=None,
                xmin=None, show_legend=True, fine_grid=False,
                xlabel=r'User Ability', ylabel=r'Probability', color=None, squaresize=False):
    """
    Unified plot function. Allows to input y as tuple of 1D and 2D numpy arrays
    Optional linewidth, label tuples

    Example usage:
    plot_figure(theta, (y_nb, y_bi), low, high, title =r"Samejima-IRT vs. Bock-IRT",
               label=("1", "2", "3", "4", "1", "2", "3", "4"), linewidth=(5, 5, 5, 5, 1, 1, 1, 1))

    Adapted from: https://github.com/northeastern-datalab/HITSnDIFFs/blob/main/IRT/irt_models.py
    """
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14  # 6
    mpl.rcParams['grid.color'] = '777777'  # grid color
    mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
    mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['axes.titlesize'] = 16
    if squaresize:
        fig = plt.figure(figsize=(4, 4))
    else:
        fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.18, 0.17, 0.76, 0.75])   # [left, bottom, width, height]

    if type(y) == tuple:
        y = np.vstack(y)
    y = np.atleast_2d(y)
    for i, yi in enumerate(y):
        plt.plot(x, yi,
                 label='{}'.format(i) if label is None else label[i],
                 linewidth=2 if linewidth is None else linewidth[i],
                 linestyle="-" if linestyle is None else linestyle[i],
                 marker=None if marker is None else marker[i],
                 markevery=None if markevery is None else markevery[i],
                 color=color[i] if color is not None else plt.rcParams['axes.prop_cycle'].by_key()['color'][i]) # assign colors from default color cycle unless explicitly specified

    plt.xlim(low, high)
    if logscale:
        plt.yscale("log")
        plt.grid(True, which="both")
        if xmin:
            plt.ylim(xmin, 2)
    else:
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if title:
        plt.title(title, fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels,
                        # loc=legend_location,  # 'upper right'
                        handlelength=2,
                        labelspacing=0,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        # title='Variants',
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        )
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_alpha(0.9)  # 0.8

    if not show_legend:
        legend.remove()

    if fine_grid:                                           # Change major and minor ticks
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))     # multiplicator for number of minor gridlines
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))     # multiplicator for number of minor gridlines
        ax.grid(which='major', alpha=0.7)
        ax.grid(which='minor', alpha=0.2)



    if pdfname:
        plt.savefig("figures/" + pdfname + ".pdf",
                    format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05)
    plt.show()


