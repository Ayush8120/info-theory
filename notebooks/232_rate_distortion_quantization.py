"""
Simple implementation of the Lloyd algorithm mentioned in section 10 on rate distorion from [Cover, Thomas'06]
First version: 11/19/2024
This version: 11/19/2024
https://northeastern-datalab.github.io/cs7840/fa24/calendar.html
"""

import numpy as np
import math
import scipy.integrate as integrate


np.set_printoptions(precision=4, suppress=True)         # show fewer digits


def fct(x, mean=0, std_dev=1):
    # PDF for Gaussian normal distribution
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    exponent = -((x - mean) ** 2) / (2 * (std_dev ** 2))
    return coefficient * math.exp(exponent)


def conditional_mean(f = fct, x1 = -1, x2 = 1):
    # Conditional mean value (centroid) of f in the region [x1, x2] (conditioned on chosing a point in that interval)
    total, _ = integrate.quad(fct, x1, x2)  # total integral of fct in [x1, x2]

    def g(x):
        return x * f(x)                     # 1st moment

    first_moment, _ = integrate.quad(g, x1, x2) # integral of "x * fct" in [x1, x2]
    return first_moment / total


# Setup
x_min = -10
x_max = 10
max_it = 100
CASE = 6

if CASE == 1:
    points = np.array([-1, 0, 2, 3])
    max_it = 80

elif CASE == 2:
    points = np.array([1, 2])
    max_it = 30

elif CASE == 3:
    points = np.array([2, 3])
    max_it = 30

elif CASE == 4:
    points = np.arange(-3, 4)
    max_it = 50

elif CASE == 5:
    points = np.arange(-3.5, 4.5)
    max_it = 200

elif CASE == 6:
    points = np.arange(-2.5, 3.5)
    max_it = 50


# print(f"Points: {points}")


def quantization_error(f = fct, x_rep = 1, x1 = -1, x2 = 1):
    # Mean squared error for approximating fct with representation x_rep in the interval [x1, x2]

    def squared_error(x):
        return f(x) * ((x - x_rep) ** 2)

    result, _ = integrate.quad(squared_error, x1, x2)  # integral
    return result


# Loop for Lloyd algorithm
for i in range(max_it):

    thresholds = (points[:-1] + points[1:]) / 2
    thresholds = np.concatenate(([x_min], thresholds, [x_max]))

    distortion = 0
    for index, _ in enumerate(points):
        distortion += quantization_error(f = fct, x_rep = points[index], x1 = thresholds[index], x2 = thresholds[index + 1])

    # print(f"{i:2d}: thresholds: {thresholds[1:-1]}, d: {distortion:.4f}")         # show the threshold
    print(f"{i:2d}: representatives: {points}, d: {distortion:.4f}")                    # show representatives

    new_points = []
    for index, _ in enumerate(points):
        point = conditional_mean(f = fct, x1 = thresholds[index], x2 = thresholds[index + 1])
        new_points.append(point)

    points = np.array(new_points)








