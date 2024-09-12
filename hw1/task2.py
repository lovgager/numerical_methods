import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from numba import jit


def generate_random_vector(n):
    return np.random.normal(0.0, 1.0, n)


@jit(nopython=True, fastmath=True)
def generate_diagonal_matrix(n):
    dmat = np.zeros((n, n))
    dval = np.exp(np.random.normal(0.0, 1.0, n))
    for idx in range(n):
        dmat[idx, idx] = dval[idx]
    return dmat


@jit(nopython=True, fastmath=True)
def generate_skew_symmetric_matrix(n):
    # smat = np.zeros((n, n))
    smat = np.random.normal(0.0, 1.0, (n, n))
    smat = 0.5 * smat - 0.5 * np.transpose(smat)
    return smat


@jit(nopython=True, fastmath=True)
def compute_rotation_matrix(smat, niter=100):
    n = smat.shape[0]
    pmat = np.eye(n)
    rmat = 0.0 + pmat
    for kiter in range(niter):
        pmat = np.dot(pmat, smat) / (kiter + 1)
        rmat = rmat + pmat
    return rmat


@jit(nopython=True, fastmath=True)
def generate_spd_matrix(n):
    dmat = generate_diagonal_matrix(n)
    smat = generate_skew_symmetric_matrix(n)
    rmat = compute_rotation_matrix(smat, niter=1000)
    spd_mat = np.dot(np.transpose(rmat), np.dot(dmat, rmat))
    spd_mat = 0.5 * spd_mat + 0.5 * np.transpose(spd_mat)
    return spd_mat


def generate_symmetric_dense_system(n):
    smat = generate_spd_matrix(n)
    x = generate_random_vector(n)
    y = np.dot(smat, x)
    return (smat, x, y)


def compute_ch_factorization(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j]))/L[j, j]
        L[i, i] = np.sqrt(A[i, i] - np.dot(L[i, :i], L[i, :i]))
    return L


def solve_l_system(lmat, y):
    n = y.size
    x = np.zeros((n))
    for idx in range(n):
        x[idx] = (y[idx] - np.dot(lmat[idx, :idx], x[:idx]))/lmat[idx, idx]
    return x


def solve_u_system(umat, y):
    n = y.size
    x = np.zeros((n))
    for idx in range(n):
        k = n - 1 - idx
        x[k] = (y[k] - np.dot(umat[k, (k + 1):], x[(k + 1):]))/umat[k, k]
    return x


def compute_solution_spd(amat, yvec):
    lmat = compute_ch_factorization(amat)
    zvec = solve_l_system(lmat, yvec)
    xvec = solve_u_system(np.transpose(lmat), zvec)
    return xvec


# %% Cholesky decomposition solver

trials = 10
n_array_ch = np.array([10, 30, 100, 300, 1000])
R_array_ch = np.zeros(n_array_ch.shape)
t_array_ch = np.zeros(n_array_ch.shape)
for i, n in enumerate(n_array_ch):
    errors = np.zeros(trials)
    times  = np.zeros(trials)
    for m in range(trials):
        A, x, y = generate_symmetric_dense_system(n)
        start = perf_counter()
        x_sol = compute_solution_spd(A, y)
        times[m] = perf_counter() - start
        errors[m] = np.max(np.abs(x - x_sol))
    R_array_ch[i] = np.max(errors)
    t_array_ch[i] = np.mean(times)

# %% print results Cholesky

print('Cholesky')
print(f'n = {n_array_ch}')
print(f'R(n) = {R_array_ch}')
print(f't(n) = {t_array_ch}')
print()

# %% Gradient descent


def init_gradient_descent(y):
    x0 = np.zeros(y.shape)
    r0 = y.copy()
    return (x0, r0)


def make_single_step_gradient_descent(x, r, smat, eps=1.0-10):
    s = np.dot(smat, r)
    alpha = np.dot(r, r) / max(eps, np.dot(r, s))
    return (x + alpha * r, r - alpha * s)


def compute_solution_gradient_descent(smat, y, niter):
    x0, r0 = init_gradient_descent(y)
    for kiter in range(niter):
        x1, r1 = make_single_step_gradient_descent(x0, r0, smat)
        x0 = x1.copy()
        r0 = r1.copy()
    return (x0, r0)


trials = 10
niter = 100
n_array_gd = np.array([10, 30, 100, 300, 1000])
R_array_gd = np.zeros(n_array_gd.shape)
t_array_gd = np.zeros(n_array_gd.shape)
for i, n in enumerate(n_array_gd):
    errors = np.zeros(trials)
    times  = np.zeros(trials)
    for m in range(trials):
        A, x, y = generate_symmetric_dense_system(n)
        start = perf_counter()
        x_sol, r0 = compute_solution_gradient_descent(A, y, niter)
        times[m] = perf_counter() - start
        errors[m] = np.max(np.abs(x - x_sol))
    R_array_gd[i] = np.max(errors)
    t_array_gd[i] = np.mean(times)

# %% print results gradient descent

print('gradient descent')
print(f'n = {n_array_gd}')
print(f'R(n) = {R_array_gd}')
print(f't(n) = {t_array_gd}')
print()


# %% conjugate gradient

def init_cgd(y):
    x0 = np.zeros(y.shape)
    r0 = y.copy()
    d0 = y.copy()
    return (x0, r0, d0)


def make_single_step_cgd(x, r, d, smat, eps=1.0-10):
    s = np.dot(smat, d)
    alpha = np.dot(r, r) / max(eps, np.dot(r, s))
    r_up = r - alpha * s
    bbeta = np.dot(r_up, r_up) / max(eps, np.dot(r, r))
    d_up = r_up + bbeta * d
    return (x + alpha * d, r_up, d_up)


def compute_solution_cgd(smat, y, niter):
    x0, r0, d0 = init_cgd(y)
    for kiter in range(niter):
        x1, r1, d1 = make_single_step_cgd(x0, r0, d0, smat)
        x0 = x1.copy()
        r0 = r1.copy()
        d0 = d1.copy()
    return (x0, r0, d0)


trials = 10
niter = 100
n_array_cg = np.array([10, 30, 100, 300, 1000])
R_array_cg = np.zeros(n_array_cg.shape)
t_array_cg = np.zeros(n_array_cg.shape)
for i, n in enumerate(n_array_cg):
    errors = np.zeros(trials)
    times  = np.zeros(trials)
    for m in range(trials):
        A, x, y = generate_symmetric_dense_system(n)
        start = perf_counter()
        x_sol, _, _ = compute_solution_cgd(A, y, niter)
        times[m] = perf_counter() - start
        errors[m] = np.max(np.abs(x - x_sol))
    R_array_cg[i] = np.max(errors)
    t_array_cg[i] = np.mean(times)


# %% plot results


plt.plot(n_array_ch, t_array_ch, marker='.', label='Cholesky')
plt.plot(n_array_gd, t_array_gd, marker='.', label='gd')
plt.plot(n_array_cg, t_array_cg, marker='.', label='cg')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('t(n)')
plt.grid(True)
plt.title('time of solving nxn system')
plt.legend()
plt.show()
