import numpy as np
import matplotlib.pyplot as plt
from task0 import Sparse


def init_system_matrix_1d_problem_task_2(nx, nt):
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


def init_right_part_task_2(nx):
    # инициализация правой части
    xnode = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    b = 0.2 + 1.6 * np.cos(2.0 * np.pi * xnode) - 0.2 * np.sin(1.0 * np.pi * xnode)
    return b


def init_gd(smat, bvec):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0 = np.zeros(bvec.shape)
    r0 = bvec - smat.multiply(x0)
    d0 = r0.copy()
    return x0, r0, d0


def single_step_gd(smat, x0, r0, d0):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    eps = 1.0e-10
    v0 = smat.multiply(d0)
    alpha = np.dot(r0, r0) / max(eps, np.dot(d0, v0))
    x1 = x0 + alpha * d0
    r1 = r0 - alpha * v0
    d1 = r1.copy()
    return x1, r1, d1


def compute_solution_gd(smat, bvec, niter):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0, r0, d0 = init_gd(smat, bvec)
    res = np.zeros((niter + 1))
    for kiter in range(niter):
        res[kiter] = np.sqrt(np.dot(r0, r0))
        x1, r1, d1 = single_step_gd(smat, x0, r0, d0)
        x0 = x1.copy()
        r0 = r1.copy()
        d0 = d1.copy()
    res[niter] = np.sqrt(np.dot(r0, r0))
    return x0, r0, res


def solve_l_system(lmat, bvec):
    # решение системы линейных уравнений с нижнетреугольной матрицей
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    nx = bvec.size
    x = np.zeros(nx)
    x[0] = bvec[0] / lmat.get_element(0, 0)
    for idx in range(1, nx):
        x[idx] = (bvec[idx] - np.dot(lmat.get_row(idx, (0, idx)), x[:idx])) \
            / lmat.get_element(idx, idx)
    return x


def solve_u_system(umat, bvec):
    # решение системы линейных уравнений с верхнетреугольной матрицей
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    nx = bvec.size
    x = np.zeros(nx)
    x[nx - 1] = bvec[nx - 1] / umat.get_element(nx - 1, nx - 1)
    for idx in range(1, nx):
        j = nx - 1 - idx
        x[j] = (bvec[j] - np.dot(umat.get_row(j, (j + 1, nx)), x[(j + 1):])) \
            / umat.get_element(j, j)
    return x


def init_gd_prec(smat, lmat, bvec):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0 = np.zeros(bvec.shape)
    r0 = bvec - smat.multiply(x0)
    d0 = r0.copy()
    d0 = solve_u_system(lmat.transpose(), d0)
    d0 = solve_l_system(lmat, d0)
    return x0, r0, d0


def single_step_gd_prec(smat, lmat, x0, r0, d0):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    eps = 1.0e-10
    v0 = smat.multiply(d0)
    alpha = np.dot(r0, d0) / max(eps, np.dot(d0, v0))
    x1 = x0 + alpha * d0
    r1 = r0 - alpha * v0
    d1 = solve_u_system(lmat.transpose(), r1)
    d1 = solve_l_system(lmat, d1)
    return x1, r1, d1


def compute_solution_gd_prec(smat, lmat, bvec, niter):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    x0, r0, d0 = init_gd_prec(smat, lmat, bvec)
    res = np.zeros((niter + 1))
    for kiter in range(niter):
        res[kiter] = np.sqrt(np.dot(r0, r0))
        x1, r1, d1 = single_step_gd_prec(smat, lmat, x0, r0, d0)
        x0 = x1.copy()
        r0 = r1.copy()
        d0 = d1.copy()
    res[niter] = np.sqrt(np.dot(r0, r0))
    return x0, r0, res


def compute_incomplete_cholesky_factorization(smat):
    # НЕОБХОДИМО ПРИВЕСТИ К ФОРМАТУ РАЗРЕЖЕННЫХ МАТРИЦ
    nx = int(np.max(smat.rows)) + 1
    lmat = Sparse(np.zeros(0), np.zeros(0), np.zeros(0))
    for idx in range(nx):
        # compute diagonal element
        col = lmat.transpose().get_row(idx, (0, idx))
        prod = np.dot(col, col)
        lmat.values = np.append(lmat.values, np.sqrt(smat.get_element(idx, idx) - prod))
        lmat.rows = np.append(lmat.rows, idx)
        lmat.cols = np.append(lmat.cols, idx)
        # compute off-diagonal element
        if idx > 0:
            lmat.values = np.append(lmat.values, \
                smat.get_element(idx, idx - 1)/lmat.get_element(idx - 1, idx - 1))
            lmat.rows = np.append(lmat.rows, idx)
            lmat.cols = np.append(lmat.cols, idx - 1)
    return lmat


def test_problem_2():
    print('test_problem_2')
    nx = 80  # размерность пространства
    nt = 600  # параметр, контролирующий вид матрицы системы линейных уравнений
    niter = 60  # число итераций по методу наискорейшего спуска
    
    # инициализация системы линеййных уравнений
    smat = init_system_matrix_1d_problem_task_2(nx, nt)
    # вычиление нижнетреуголбной матрицы для предобуславливания
    lmat = compute_incomplete_cholesky_factorization(smat)
    # правая часть
    bvec = init_right_part_task_2(nx)
    # решение системы линейных уравнений без предобуславливания
    u, r, res_norm = compute_solution_gd(smat, bvec, niter)
    # решение системы линейных уравнений с предобуславливанием
    u, r, res_prec = compute_solution_gd_prec(smat, lmat, bvec, niter)
    
    # график с десятичным логарифмом модуля вектора невязки
    plt.figure()
    plt.scatter(range(len(res_norm)), np.log10(res_norm), c='r', label='norm')
    plt.scatter(range(len(res_norm)), np.log10(res_prec), c='b', label='prec')
    plt.xlabel('iteration')
    plt.ylabel(r'$\log_{10}(|r|)$')
    plt.legend()
    plt.grid()
    plt.show()

test_problem_2()