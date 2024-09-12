import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt


def compute_function_0(x):
    return x + 0.95 * np.sin(x)


def compute_function_1(x):
    t = np.pi * (0.5 + x / 5.0 / np.pi)
    return np.cos(t) / np.sin(t) / np.sin(t)


def make_plots_problem_1():
    numx = 10001
    x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, numx)
    y0 = compute_function_0(x)
    y1 = compute_function_1(x)
    plt.figure()
    plt.plot(x, y0, c='r', label='$f_0(x)$')
    plt.plot(x, y1, c='b', label='$f_1(x)$')
    plt.legend()
    plt.grid()
    plt.show()


make_plots_problem_1()


def binary_search(left_bound, right_bound, target, func, eps=1.0e-10):
    center = (left_bound + right_bound)/2
    if right_bound - left_bound < eps:
        return center
    y_left = func(left_bound)
    y_center = func(center)
    y_right = func(right_bound)
    if (y_left - target)*(y_center - target) >= 0:
        return binary_search(center, right_bound, target, func, eps)
    if (y_center - target)*(y_right - target) >= 0:
        return binary_search(left_bound, center, target, func, eps)


def solve_problem_1():
    target = 4.0
    left = -2*np.pi
    right = 2*np.pi

    x_0 = binary_search(left, right, target, compute_function_0)
    x_1 = binary_search(left, right, target, compute_function_1)
    assert abs(compute_function_0(x_0) - target) < 1e-10
    assert abs(compute_function_1(x_1) - target) < 1e-10
    print(x_0)
    print(x_1)


solve_problem_1()