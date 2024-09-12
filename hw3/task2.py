import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from tqdm import tqdm

from ode_solver import compute_numerical_solution_rk_5, compute_numerical_trajectory_rk_5


def ffun(v, t_final=2.0):
    """
    This function computes the position by velocity vector
    """
    t0 = 0.0
    x = np.zeros(4)
    x[0] = 1.0
    x[2:] = v.copy()
    nstep = 1000
    return compute_numerical_solution_rk_5(t0, x, t_final, nstep)[:-2]


def initial_guess_problem_2(x_final, t_final=2.0):
    x_start = np.asarray([1.0, 0.0])
    return (x_final - x_start) / t_final


def compute_jacoby_matrix_problem_2(x):
    dx = 1.0e-6
    dimx = x.size
    jmat = np.zeros((dimx, dimx))
    f0 = ffun(x)
    for idx in range(dimx):
        x_pert = x.copy()
        x_pert[idx] += dx
        jmat[:, idx] = (ffun(x_pert) - f0) / dx
    return jmat


def compute_trajectory_problem_2(v, t_final=2.0, nstep=1000):
    x_start = np.asarray([1.0, 0.0])
    x = np.zeros(4)
    x[:2] = x_start.copy()
    x[2:] = v.copy()
    x_numer = compute_numerical_trajectory_rk_5(0.0, x, t_final, nstep)
    return x_numer[:, :2]


def make_plots_problem_2():
    v = np.random.normal(0.0, 1.0, 2)
    print(f'Velocity: ({v[0]}, {v[1]})')
    x_trajectory = compute_trajectory_problem_2(v, t_final=2.0, nstep=1000)

    plt.plot(x_trajectory[:, 0], x_trajectory[:, 1])
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$X_1$')
    plt.grid()
    plt.show()


make_plots_problem_2()


def newton_solver(y, x0, num_iter=400):
    """
    Newton method
    You can use np.linalg.solve() function
    """
    x = x0
    for _ in tqdm(range(num_iter)):
        J = compute_jacoby_matrix_problem_2(x)
        x += np.linalg.solve(J, y - compute_trajectory_problem_2(x)[-1])
    return x


def solve_problem_2():
    x_final = np.array([-0.8, 0.3])
    v = initial_guess_problem_2(x_final)
    v = newton_solver(x_final, v)

    x_final_pred = ffun(v)
    print(f'Start velocity: ({v[0]}, {v[1]})')
    print(f'Final point: ({x_final_pred[0]}, {x_final_pred[1]})')
    assert np.max(np.absolute(x_final - x_final_pred)) < 1.0e-2

    x_trajectory = compute_trajectory_problem_2(v, t_final=2.0, nstep=1000)
    plt.plot(x_trajectory[:, 0], x_trajectory[:, 1])
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$X_1$')
    plt.grid()
    plt.show()


solve_problem_2()