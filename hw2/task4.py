import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm


@njit
def generate_grid(nx, ny):
    mesh_data = np.zeros((nx, ny, 2))
    for kx in range(nx):
        for ky in range(ny):
            mesh_data[kx, ky, 0] = (0.5 + kx) / nx
            mesh_data[kx, ky, 1] = (0.5 + ky) / ny
    return mesh_data


@njit
def distance_matrix(mesh_data):
    nx, ny, _ = mesh_data.shape
    n_block = nx * ny
    dist_mat = np.zeros((n_block, n_block))
    for kx0 in range(nx):
        for kx1 in range(nx):
            for ky0 in range(ny):
                for ky1 in range(ny):
                    idx_0 = ny * kx0 + ky0
                    idx_1 = ny * kx1 + ky1
                    diff = mesh_data[kx1, ky1, :] - mesh_data[kx0, ky0, :]
                    dist_mat[idx_0, idx_1] = np.sqrt(np.sum(diff * diff))
    return dist_mat


@njit
def generate_gram_matrix(n, sigma):
    mesh_data = generate_grid(n, n)
    dist_mat = distance_matrix(mesh_data)
    gram_mat = np.exp(- dist_mat / sigma)
    return gram_mat


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


def compute_eigen_pair_qr(smat: np.ndarray, num_vec: int, num_iter: int = 10000):
    dim, _ = smat.shape
    vvecs = init_qr_method(dim=dim, num_vec=num_vec)
    for _ in tqdm(range(num_iter)):
        vvecs = compute_gram_schmidt_ortho(np.dot(smat, vvecs))
    eigen_vectors = vvecs
    eigen_valus = (np.dot(smat, eigen_vectors) / eigen_vectors)[0]
    return eigen_valus, eigen_vectors


num_vec = 10
sigmas = [0.1, 0.3, 0.9]
np.random.seed(1)
for n in [10, 30, 100]:
    dim = n**2
    # графики собств.значений
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('k')
    ax.set_ylabel('lambda_k')
    ax.set_title(f'eigenvalues\ndim: {n}')
    ax.grid()
    vecs_to_plot = np.empty((len(sigmas), dim, num_vec))
    for i, sigma in enumerate(sigmas):
        G = generate_gram_matrix(n, sigma)
        vals, vecs = compute_eigen_pair_qr(G, num_vec, 50)
        index = np.argsort(vals)[::-1]
        vals = vals[index]
        vecs = vecs[:, index]
        vecs_to_plot[i, :, :] = vecs[:, :num_vec]
        ax.scatter(np.arange(len(vals)), vals, label=f'sigma: {sigma}')
    ax.legend()
    fig.savefig(f'eigvalues{n}')
    
    # графики собств.функций
    fig, ax = plt.subplots(nrows=len(sigmas), ncols=num_vec, figsize=(16,8), dpi=70)
    fig.subplots_adjust(wspace=0)
    for i, sigma in enumerate(sigmas):
        for k in range(num_vec):
            vec_to_plot = vecs_to_plot[i, :, k]
            ax[i, k].imshow(vec_to_plot.reshape((n, n)))
            ax[i, k].set_title(f'dim: {n}\nsigma: {sigma}\neignumber: {k+1}')
            ax[i, k].set_xticks([])
            ax[i, k].set_yticks([])
    fig.savefig(f'eigvectors{n}')
