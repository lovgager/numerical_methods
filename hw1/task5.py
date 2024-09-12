import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from functools import partial


def compute_test_function_value(xdata: np.ndarray, nstep: int = 100) -> np.ndarray:
    y = np.ones((xdata.shape[0]))
    for idx in range(nstep):
        y = y - np.cos(np.pi * (idx + 1) * xdata.flatten()) / (idx + 1)
        y = y - np.sin(np.pi * (idx + 1) * xdata.flatten()) / (idx + 1)
        y = y + np.cos(np.pi * (idx + 2) * xdata.flatten()) / (idx + 2)
        y = y + np.sin(np.pi * (idx + 2) * xdata.flatten()) / (idx + 2)
    return y


def generate_data(ndata: int) -> np.ndarray:
    x = np.random.rand(ndata).reshape(ndata, 1)
    return x


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


def main(N, n):
    """
    :param N: number of nodes
    :param n: number of basis functions
    """
    x = generate_data(N)
    y = compute_test_function_value(x)
    phi = [lambda x: np.ones(len(x))]
    cosines = [partial(lambda x,k: np.cos(np.pi*x*k), k=k) for k in range(1,n//2+1)]
    sines   = [partial(lambda x,k: np.sin(np.pi*x*k), k=k) for k in range(1,n//2+1)]
    pairs = zip(cosines, sines)
    phi.extend([f for p in pairs for f in p])
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            A[i, j] = np.dot(phi[i](x).flatten(), phi[j](x).flatten())/N
        b[i] = np.dot(phi[i](x).flatten(), y)/N
        
    # using conjugate gradients because the matrix is symmetric and positive
    c, _, _ = compute_solution_cgd(A, b, 100)
    f = lambda x: c[0] + np.dot(np.array([p(x) for p in phi[1:]]).T, c[1:])
    x_plt = np.arange(0, 1, 0.001)
    f_reg = f(x_plt)
    f_exact = compute_test_function_value(x_plt)
    plt.plot(x_plt, f_reg, label='regression')
    plt.plot(x_plt, f_exact, label='exact')
    plt.legend()
    plt.grid(True)
    print(f'N = {N}, n = {n}')
    print(f'error = {np.linalg.norm(f_reg - f_exact, ord=np.inf)}')

# N - number of nodes (points)
# n - number of basis functions (must be odd)
main(N=400, n=101)
# Conclusion: 
# if N < n then the approximation is bad because of strong oscillations
# Must be N >> n
