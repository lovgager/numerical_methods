import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

def test_function_problem_5(x):
    term0 = np.cos(2.0 * np.pi * x[:, 0])
    term1 = np.sin(2.0 * np.pi * x[:, 1])
    term2 = np.sinh(x[:, 0] - x[:, 1] * x[:, 1] * x[:, 1])
    return term0 - term1 + term2


def generate_mesh(nx0, nx1):
    """
    Generating a mesh - interpolation nodes in the case of bilinear interpolation
    nx0, nx1 - mesh sizes along axes 0 and 1, respectively
    """
    x_nodes = np.linspace(0, 1, nx0)
    y_nodes = np.linspace(0, 1, nx1)
    x_mesh, y_mesh = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    mesh_node = np.stack([x_mesh, y_mesh], axis=-1)
    return mesh_node


def compute_mesh_value(mesh_node):
    """
    Computing values at interpolation nodes
    """
    nx0, nx1, dim = mesh_node.shape
    y = test_function_problem_5(np.reshape(mesh_node, (nx0 * nx1, dim)))
    return np.reshape(y, (nx0, nx1))


def generate_test_data(num_samples):
    """
    Generating data for testing the interpolation algorithm
    """
    x = np.random.rand(num_samples, 2)
    y = test_function_problem_5(x)
    return x, y


class BilinearInterpolation:
    def __init__(self, nodes: np.ndarray, values: np.ndarray):
        self.nodes = nodes.copy()
        self.values = values.copy()
        self.n0, self.n1, _ = self.nodes.shape
        self.h0 = 1/(self.n0 - 1)
        self.h1 = 1/(self.n1 - 1)

    def predict(self, x: np.ndarray):
        res = np.zeros(x.shape[0])
        w = self.bilinear_compute_weights(x)
        for k, point in enumerate(x):
            i = int(point[0]/self.h0)
            j = int(point[1]/self.h1)
            res[k] = w[k,0]*w[k,1]*self.values[i,j] + (1-w[k,0])*w[k,1]*self.values[i+1,j] + \
                w[k,0]*(1-w[k,1])*self.values[i,j+1] + (1-w[k,0])*(1-w[k,1])*self.values[i+1,j+1]
        return res

    def bilinear_compute_weights(self, x: np.ndarray):
        w = np.empty(x.shape)
        for k, point in enumerate(x):
            i = int(point[0]/self.h0)
            j = int(point[1]/self.h1)
            w[k,0] = (self.nodes[i+1,j,0] - point[0])/(self.nodes[i+1,j,0] - self.nodes[i,j,0])
            w[k,1] = (self.nodes[i,j+1,1] - point[1])/(self.nodes[i,j+1,1] - self.nodes[i,j,1])
        return w
    
    
class RadialBasisFunctionInterpolation:
    def __init__(self, nodes: np.ndarray, values: np.ndarray, dim: int, s1: int, num_samples: int):
        self.nodes = nodes.copy()
        self.values = values.copy()
        self.dim = dim
        self.s1 = s1
        self.num_samples = num_samples
        self.s = self.s1 / (self.num_samples * self.dim)

    def predict(self, x: np.ndarray):
        w = self.radial_basis_compute_weight(x)
        return w @ self.values
    
    def radial_basis_compute_weight(self, x: np.ndarray):
        w = np.empty((len(x), self.num_samples))
        for k, point in enumerate(x):
            denominator = np.sum([np.exp(-np.linalg.norm(point - xj)**2/self.s**2)
                                  for xj in self.nodes])
            w[k] = np.exp(-np.linalg.norm(point - self.nodes, axis=1)**2/self.s**2)/denominator
        return w

def solve_problem_5():
    # bilinear
    
    nx_arr = np.array([4, 10, 20, 40, 80, 160, 320, 640, 1280])
    r_arr = np.zeros(nx_arr.shape)
    for k, nx0 in enumerate(nx_arr):
        nx1 = nx0
        N = nx0*nx1
        nodes = generate_mesh(nx0, nx1)
        values = compute_mesh_value(nodes)
        b = BilinearInterpolation(nodes, values)
        x, y = generate_test_data(100)
        r_arr[k] = np.linalg.norm(y - b.predict(x), ord=np.inf)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
    ax[0].plot(np.log2(nx_arr**2), np.log2(r_arr), '.-')
    ax[0].set_xlabel('log2(N)')
    ax[0].set_ylabel('log2(r)')
    ax[0].set_title('Bilinear')
    ax[0].grid()
    
    # radial uniform
    
    nx_arr = 2**np.arange(1, 7)
    r_arr = np.zeros(nx_arr.shape)
    for k, nx0 in enumerate(nx_arr):
        nx1 = nx0
        N = nx0*nx1
        nodes = generate_mesh(nx0, nx1)
        values = compute_mesh_value(nodes)
        rb = RadialBasisFunctionInterpolation(nodes.reshape((N,2)), values.reshape(N), 2, 10, N)
        x, y = generate_test_data(100)
        r_arr[k] = np.linalg.norm(y - rb.predict(x), ord=np.inf)
    
    ax[1].plot(np.log2(nx_arr**2), np.log2(r_arr), '.-', c='C1', label='uniform')
    ax[1].set_xlabel('log2(N)')
    ax[1].set_title('Radial')
    
    # radial random
    
    N_arr = 2**np.arange(2, 13)
    r_arr = np.zeros(N_arr.shape)
    for k, N in enumerate(N_arr):
        nodes = np.random.random((N, 2))
        values = test_function_problem_5(nodes)
        rb = RadialBasisFunctionInterpolation(nodes, values, 2, 10, N)
        x, y = generate_test_data(100)
        r_arr[k] = np.linalg.norm(y - rb.predict(x), ord=np.inf)
    
    ax[1].plot(np.log2(N_arr), np.log2(r_arr), '.-', c='C2', label='random')
    ax[1].legend()
    ax[1].grid()

solve_problem_5()
