import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_data(ndata):
    # random angles
    phi = 2.0 * np.pi * np.random.rand(3 * ndata)
    # coordinates
    xdata = np.zeros((phi.size, 2))
    # circles of different radius
    r0 = 1.0
    r1 = 2.0
    r2 = 3.0
    # data for cluster 0:
    idx_0 = 0 * ndata
    idx_1 = 1 * ndata
    xdata[idx_0 : idx_1, 0] = r0 * np.cos(phi[idx_0 : idx_1])
    xdata[idx_0 : idx_1, 1] = r0 * np.sin(phi[idx_0 : idx_1])
    # data for cluster 1:
    idx_0 = 1 * ndata
    idx_1 = 2 * ndata
    xdata[idx_0 : idx_1, 0] = r1 * np.cos(phi[idx_0 : idx_1])
    xdata[idx_0 : idx_1, 1] = r1 * np.sin(phi[idx_0 : idx_1])
    # data for cluster 2:
    idx_0 = 2 * ndata
    idx_1 = 3 * ndata
    xdata[idx_0 : idx_1, 0] = r2 * np.cos(phi[idx_0 : idx_1])
    xdata[idx_0 : idx_1, 1] = r2 * np.sin(phi[idx_0 : idx_1])
    return xdata


def init_qr_method(dim: int, num_vec: int):
    vvecs = np.random.normal(loc=0.0, scale=1.0, size=(dim, num_vec))
    return vvecs / np.linalg.norm(vvecs, axis=0)


def compute_gram_schmidt_ortho(vvecs: np.ndarray):
    # столбцы vvecs уже ортонормированы в функции init_qr_merthod
    _, num_vec = vvecs.shape
    uvecs = vvecs.copy()
    for i in range(num_vec):
        v_curr = uvecs[:, i]
        v_prevs = uvecs[:, :i]
        v_curr -= np.dot(v_prevs, np.dot(v_prevs.T, v_curr))
        uvecs[:, i] = v_curr / np.linalg.norm(v_curr)
    return uvecs


# вычисление МИНИМАЛЬНЫХ собственных значений
def compute_eigen_pair_qr(smat: np.ndarray, num_vec: int, num_iter: int = 10000):
    dim, _ = smat.shape
    vvecs = init_qr_method(dim=dim, num_vec=num_vec)
    for _ in tqdm(range(num_iter)):
        for i in range(num_vec):
            vvecs[:, i] = compute_solution_gen(smat, vvecs[:, i])
        vvecs = compute_gram_schmidt_ortho(vvecs)
    eigen_vectors = vvecs
    eigen_valus = (np.dot(smat, eigen_vectors) / eigen_vectors)[0]
    return eigen_valus, eigen_vectors


def solve_l_system(lmat: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = y.size
    x = np.zeros(n)
    for i in range(n):
        x[i] = (y[i] - np.sum(lmat[i, :i] * x[:i])) / lmat[i, i]
    return x


def solve_u_system(umat: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = y.size
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(umat[i, (i + 1):] * x[(i + 1):])) / umat[i, i]
    return x

    
def compute_lu_factorization(mat: np.ndarray) -> (np.ndarray, np.ndarray):
    n, _ = mat.shape
    lmat = np.zeros((n, n))
    umat = np.eye(n)
    for i in range(n):
        for j in range(i + 1):
            lmat[i, j] = (mat[i, j] - np.sum(lmat[i, :j] * umat[:j, j])) / umat[j, j]
        for j in range(i + 1, n):
            umat[i, j] = (mat[i, j] - np.sum(lmat[i, :i] * umat[:i, j])) / lmat[i, i]
    return lmat, umat


def compute_solution_gen(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    lmat, umat = compute_lu_factorization(mat=mat)
    z = solve_l_system(lmat=lmat, y=y)
    x = solve_u_system(umat=umat, y=z)
    return x


n1 = 20 # число точек в одном кластере
n = n1 * 3
gamma = 1
W = np.empty((n, n))
data = generate_data(n1) # вектор размера (n, 2)
x = data[:, 0]
y = data[:, 1]
x1, x2 = np.meshgrid(x, x)
y1, y2 = np.meshgrid(y, y)
W = np.exp(-gamma*((x1 - x2)**2 + (y1 - y2)**2))
d = np.sum(W, axis=0) # диагональ матрицы D
D = np.diag(d)
D_sqrt_inv = np.diag(1/np.sqrt(d)) # D в степени -1/2
L = D_sqrt_inv @ (D - W) @ D_sqrt_inv

np.random.seed(1)
vals, vecs = compute_eigen_pair_qr(L, 4, 100)
index = np.argsort(vals)
vals = vals[index]
vecs = vecs[:, index]

plt.grid()
plt.title('Eigenfunctions with minimal eigenvalues')
plt.xlabel('n')
plt.scatter(np.arange(n), vecs[:, 0], label='1')
plt.scatter(np.arange(n), vecs[:, 1], label='2')
plt.scatter(np.arange(n), vecs[:, 2], label='3')
plt.scatter(np.arange(n), vecs[:, 3], label='4')
plt.legend()
plt.savefig('eigfunctions.png')

def test_problem_5():
    print('inside the main function')
    ndata = 200
    xdata = generate_data(ndata)
    plt.figure()
    idx_0 = 0 * ndata
    idx_1 = 1 * ndata
    plt.scatter(xdata[idx_0:idx_1, 0], xdata[idx_0:idx_1, 1], c='r', label='cluster 0')
    idx_0 = 1 * ndata
    idx_1 = 2 * ndata
    plt.scatter(xdata[idx_0:idx_1, 0], xdata[idx_0:idx_1, 1], c='g', label='cluster 1')
    idx_0 = 2 * ndata
    idx_1 = 3 * ndata
    plt.scatter(xdata[idx_0:idx_1, 0], xdata[idx_0:idx_1, 1], c='b', label='cluster 2')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('clusters.png')
    return 0

test_problem_5()
