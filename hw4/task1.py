import numpy as np
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt


def compute_target_function_pr1(x):
    return np.cos(np.pi * (x - 0.25 * x * x))


def compute_target_first_derivative_function_pr1(x):
    return np.pi * (x - 2.0) * np.sin(np.pi * (x - 0.25 * x * x)) / 2.0


def compute_target_second_derivative_function_pr1(x):
    term1 = np.pi * (x - 2.0) * (x - 2.0) * np.cos(np.pi * (x - 0.25 * x * x))
    term2 = 2.0 * np.sin(np.pi * (x - 0.25 * x * x))

    return -np.pi * (term1 - term2) / 4.0


def compute_target_function_derivative(x, ord_der: int):
    if ord_der == 0:
        return compute_target_function_pr1(x)
    if ord_der == 1:
        return compute_target_first_derivative_function_pr1(x)
    if ord_der == 2:
        return compute_target_second_derivative_function_pr1(x)
    return None


def generate_interpolation_nodes_uniform(segment: list[float], count_nodes: int) -> np.ndarray:
    return np.linspace(segment[0], segment[1], count_nodes)


def generate_interpolation_nodes_random(segment: list[float], count_nodes: int) -> np.ndarray:
    return np.sort(np.random.uniform(segment[0], segment[1], count_nodes))


def generate_test_points(segment: list[float], count_nodes: int = 1000) -> (np.ndarray, np.ndarray):
    x = np.linspace(segment[0], segment[1], count_nodes + 1)
    y = compute_target_function_pr1(x)
    return x, y


def compute_inaccuracy(true_values, predict_values):
    return np.sqrt(np.mean((true_values - predict_values) ** 2))
    

class LagrangePoly:
    """
    You are not allowed to use a for-loop over x
    """

    def __init__(self, nodes: np.ndarray, values: np.ndarray):
        self.nodes = nodes.copy()
        self.values = values.copy()

    def compute_lagrange_basis(self, x: np.ndarray):
        """
        x: points for which to compute the basis polynomial
        return: values of the basis polynomial at the point x
        """
        n = len(self.nodes)
        L_basis = np.zeros((n, len(x))) # базис из n полиномов
        for i in range(n):
            p = [(x - self.nodes[j])/(self.nodes[i] - self.nodes[j]) 
                 for j in range(n) if j != i]
            L_basis[i] = np.prod(p, axis=0)
        return L_basis

    def predict(self, x: np.ndarray):
        """
        x: points for which to compute the interpolation value
        return: interpolation values at the point x
        """
        L_basis = self.compute_lagrange_basis(x)
        return self.values @ L_basis


def solve_problem_1():
    segment = [-1, 1]
    x, y = generate_test_points(segment) # для валидации
    xx = np.linspace(-1, 2, 100) # отрезок [-1, 2] для графиков
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for i, n in enumerate([3, 5, 10, 15]):
        # равномерные узлы
        nodes = generate_interpolation_nodes_uniform(segment, n)
        values = compute_target_function_pr1(nodes)
        LP = LagrangePoly(nodes, values)
        r = compute_inaccuracy(y, LP.predict(x))
        print(f'n = {n}, uniform: r = {r}')
        
        ax[i].set_title(f'n = {n}')
        ax[i].plot(xx, compute_target_function_pr1(xx), label='f(x)')
        ax[i].plot(xx, LP.predict(xx), label='LP uniform')
        
        # случайные узлы
        nodes = generate_interpolation_nodes_random(segment, n)
        values = compute_target_function_pr1(nodes)
        LP = LagrangePoly(nodes, values)
        r = compute_inaccuracy(y, LP.predict(x))
        print(f'n = {n},  random: r = {r}')
        
        ax[i].plot(xx, LP.predict(xx), label='LP random')
        ax[i].legend()
        ax[i].grid()


solve_problem_1()
