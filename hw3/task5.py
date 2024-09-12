import numpy as np
import scipy.special
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
from scipy.special import eval_hermitenorm
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_metric_tensor(x, r0=1.0):
    eps = 1.0e-2
    gmat = np.eye(x.size)
    factor = (np.dot(x, x) + eps * r0 * r0) / (np.dot(x, x) + r0 * r0)
    return factor * gmat


def compute_segment_length(x0, x1, nsample=100):
    t = np.linspace(0.0, 1.0, nsample + 1)
    dx = (x1 - x0) / nsample
    s = 0.0
    for idx in range(nsample):
        x = x0 + 0.5 * dx * (t[idx] + t[idx + 1])
        gmat = compute_metric_tensor(x)
        s += np.sqrt(np.dot(dx, np.dot(gmat, dx)))
    return s


def compute_pwl_curve_length(xpwl):
    # length of piecewise linear curve
    nseg, dimx = xpwl.shape
    s = 0.0
    for kseg in range(1, nseg):
        s += compute_segment_length(xpwl[kseg - 1, :], xpwl[kseg, :])
    return s


def compute_objective_function(
    x,
    x0=np.asarray([-1.0, 0.0]),
    x1=np.asarray([0.3, 0.1])
):
    dims = x0.size
    nseg = np.int64(x.size / dims)
    xdata = np.zeros((2 + nseg, dims))
    xdata[0, :] = x0.copy()
    xdata[-1, :] = x1.copy()
    xdata[1: (-1), :] = np.reshape(x, (nseg, dims)).copy()
    return compute_pwl_curve_length(xdata)


def compute_objective_function_gradient(x):
    dx = 1.0e-6
    dimx = x.size
    x0 = x.copy()
    l0 = compute_objective_function(x0)
    grad = np.zeros(dimx)
    for idx in range(dimx):
        x1 = x0.copy()
        x1[idx] += dx
        l1 = compute_objective_function(x1)
        grad[idx] = (l1 - l0) / dx
    return grad


def init_optimization_problem_5(
    nseg,
    x0=np.asarray([-1.0, 0.0]),
    x1=np.asarray([0.3, 0.1])
):
    x = np.zeros((nseg, x0.size))
    for kseg in range(nseg):
        x[kseg, :] = (1 + kseg) / (nseg + 2) * x1 + (nseg - kseg + 1) / (nseg + 2) * x0
    return x.flatten()


def single_gradient_descent_step_problem_5(x, dt):
    grad = compute_objective_function_gradient(x)
    return x - dt * grad


def single_step_momentum(x, x_prev, dt, b):
    grad = compute_objective_function_gradient(x)
    return x - dt * grad + b*(x - x_prev)


def compute_optimal_x_problem_5(nseg, dt, b, num_iter):
    x0 = init_optimization_problem_5(nseg)
    x0_prev = x0
    for _ in tqdm(range(num_iter)):
        # x1 = single_gradient_descent_step_problem_5(x0, dt)
        x1 = single_step_momentum(x0, x0_prev, dt, b)
        x0_prev = x0.copy()
        x0 = x1.copy()
    return x0


def make_plot_problem_5():
    nseg = 50
    dt = 0.01
    b = 0.1
    niter = 100
    x = compute_optimal_x_problem_5(nseg, dt, b, niter)
    x = np.reshape(x, (nseg, -1))

    plt.figure()
    plt.plot([-1.0, x[0, 0]], [0.0, x[0, 1]], c='g')
    for idx in range(1, x.shape[0]):
        plt.plot([x[idx - 1, 0], x[idx, 0]], [x[idx - 1, 1], x[idx, 1]], c='b')
    plt.plot([0.3, x[-1, 0]], [0.1, x[-1, 1]], c='r')
    plt.grid()
    plt.show()


make_plot_problem_5()
