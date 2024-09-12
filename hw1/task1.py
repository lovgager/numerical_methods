import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def generate_diagonally_dominant_matrix(n):
    eps = 1.0e-3
    smat = np.random.normal(0.0, 1.0, (n, n))
    for a in range(n):
        smat[a, a] = np.sum(np.absolute(smat[a, :])) + eps
    return smat


def generate_random_vector(n):
    return np.random.normal(0.0, 1.0, n)


def generate_random_system_of_linear_equations(n):
    smat = generate_diagonally_dominant_matrix(n)
    perm = np.random.permutation(n)
    smat = smat[:, perm[:]]
    x = generate_random_vector(n)
    y = np.dot(smat, x)
    return (smat, x, y)


def compute_lu_factorization(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j]))/U[j, j]
        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - np.dot(L[i, :i], U[:i, j]))/L[i, i]
    return (L, U)


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


def compute_solution_gen(amat, yvec):
    lmat, umat = compute_lu_factorization(amat)
    zvec = solve_l_system(lmat, yvec)
    xvec = solve_u_system(umat, zvec)
    return xvec


trials = 10
n_array = np.array([10, 30, 100, 300, 1000])
R_array = np.zeros(n_array.shape)
t_array = np.zeros(n_array.shape)
for i, n in enumerate(n_array):
    errors = np.zeros(trials)
    times  = np.zeros(trials)
    for m in range(trials):
        A, x, y = generate_random_system_of_linear_equations(n)
        start = perf_counter()
        x_sol = compute_solution_gen(A, y)
        times[m] = perf_counter() - start
        errors[m] = np.max(np.abs(x - x_sol))
    R_array[i] = np.max(errors)
    t_array[i] = np.mean(times)

# %%

print(f'n = {n_array}')
print(f'R(n) = {R_array}')
print(f't(n) = {t_array}')

plt.plot(n_array, t_array, marker='.', markersize=3, lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('t(n)')
plt.grid(True)
plt.title('time of solving nxn system')
plt.show()
