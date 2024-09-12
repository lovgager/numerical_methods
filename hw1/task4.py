import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True, fastmath=True)
def init_gradient_descent(y):
    x0 = np.zeros(y.shape)
    r0 = y.copy()
    return (x0, r0)


@jit(nopython=True, fastmath=True)
def make_single_step_gradient_descent(x, r, smat, eps=1.0-10):
    s = np.dot(smat, r)
    alpha = np.dot(r, r) / max(eps, np.dot(r, s))
    return (x + alpha * r, r - alpha * s)


@jit(nopython=True, fastmath=True)
def compute_solution_gradient_descent(smat, y, niter):
    x0, r0 = init_gradient_descent(y)
    for kiter in range(niter):
        x1, r1 = make_single_step_gradient_descent(x0, r0, smat)
        x0 = x1.copy()
        r0 = r1.copy()
    return (x0, r0)


@jit(nopython=True, fastmath=True)
def main():
    n_array = np.array([5, 10, 20, 40, 80])
    r_array = np.zeros(n_array.shape)
    for i, n in enumerate(n_array):
        niter = 100
        h = 1/n
        x = np.arange(h/2, 1+h/2, h)
        y = np.arange(h/2, 1+h/2, h).reshape(-1, 1)
        A = (4 + h**2)*np.eye(n**2)-np.eye(n**2, k=1)-np.eye(n**2, k=-1) - \
            np.eye(n**2, k=n) - np.eye(n**2, k=-n)
        A[0, -1] = -1
        A[-1, 0] = -1
        f = ((1 + 13*4*np.pi**2)*np.cos(4*np.pi*x)*np.sin(6*np.pi*y)).flatten()
        u_exact = (np.cos(4*np.pi*x)*np.sin(6*np.pi*y)).flatten()
        u, r = compute_solution_gradient_descent(A, f*h**2, niter)
        r_array[i] = np.max(np.abs(u - u_exact))
    return n_array, r_array


n_array, r_array = main()

# %% plot results

plt.plot(n_array, r_array, marker='.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('r(n)')
plt.title('n vs error')
plt.show()
