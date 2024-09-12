import numpy as np
import matplotlib.pyplot as plt
from task0 import Sparse


def init_system_matrix_1d_problem_task_1(nx, nt):
    # инициализация тридиагональной матрицы системы линейных уравнений
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    dt = 1.0 / nt
    dx = 1.0 / nx
    main_diag = np.hstack((1 + 0.5*dt/dx**2, [1 + dt/dx**2]*(nx - 2), 1 + 0.5*dt/dx**2))
    upper_diag = np.hstack([-0.5*dt/dx**2]*(nx - 1))
    lower_diag = np.hstack([-0.5*dt/dx**2]*(nx - 1))
    values = np.hstack((main_diag, upper_diag, lower_diag))
    rows = np.hstack((np.arange(nx), np.arange(nx - 1), np.arange(1, nx)))
    cols = np.hstack((np.arange(nx), np.arange(1, nx), np.arange(nx - 1)))
    return Sparse(rows, cols, values)


def init_right_part_task_1(nx):
    # инициализация правой части
    xnode = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    b = 0.2 + 1.6 * np.cos(2.0 * np.pi * xnode) - 0.2 * np.sin(1.0 * np.pi * xnode)
    return b


def init_bcg(smat, y):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0 = np.zeros(y.shape)
    r0 = y - smat.multiply(x0)
    rh0 = r0.copy()
    d0 = r0.copy()
    dh0 = d0.copy()
    return x0, r0, rh0, d0, dh0


def single_step_bcg(smat, x0, r0, d0, rh0, dh0):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    v0 = smat.multiply(d0)
    vh0 = smat.transpose().multiply(dh0)
    alpha = np.dot(rh0, r0) / np.dot(v0, dh0)
    x1 = x0 + alpha * d0

    r1 = r0 - alpha * v0
    rh1 = rh0 - alpha * vh0

    beta = np.dot(rh1, r1) / np.dot(rh0, r0)

    d1 = r1 + beta * d0
    dh1 = rh1 + beta * dh0
    return x1, r1, rh1, d1, dh1


def compute_solution_bcg(smat, bvec, niter):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0, r0, d0, rh0, dh0 = init_bcg(smat, bvec)
    res = np.zeros((niter + 1))
    for kiter in range(niter):
        res[kiter] = np.sqrt(np.dot(r0, r0))
        x1, r1, d1, rh1, dh1 = single_step_bcg(smat, x0, r0, d0, rh0, dh0)
        x0 = x1.copy()
        r0 = r1.copy()
        d0 = d1.copy()
        rh0 = rh1.copy()
        dh0 = dh1.copy()
    res[niter] = np.sqrt(np.dot(r0, r0))
    return x0, r0, res


def test_problem_1():
    print('test_problem_1')
    nx = 80  # размерность пространства
    nt = 600  # параметр, контролирующий вид матрицы системы линейных уравнений
    niter = 60  # число итераций по методу би-сопряженнх градиентов
    smat = init_system_matrix_1d_problem_task_1(nx, nt)
    bvec = init_right_part_task_1(nx)

    # данную функцию надо реализовать
    u, r, res_norm = compute_solution_bcg(smat, bvec, niter)

    # график с десятичным логарифмом модуля вектора невязки
    plt.figure()
    plt.scatter(range(len(res_norm)), np.log10(res_norm), c='r')
    plt.xlabel('iteration')
    plt.ylabel(r'$\log_{10}(|r|)$')
    plt.grid()
    plt.show()

test_problem_1()