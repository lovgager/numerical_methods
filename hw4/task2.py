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


class NewtonPoly:
    """
    You are not allowed to use a for-loop over x
    """

    def __init__(self, nodes: np.ndarray, values: np.ndarray):
        self.nodes = nodes.copy()
        self.values = values.copy()
        self.coef_ = self.divided_differences()

    def divided_differences(self):
        """
        Compute divided differences
        :return: polynomial coefficients
        """
        def f(i, j): # f[xi, ..., xj]
            if i == j:
                return self.values[i]
            return (f(i+1, j) - f(i, j-1))/(self.nodes[j] - self.nodes[i])
        return [f(0, j) for j in range(len(self.nodes))]

    def predict(self, x: np.ndarray):
        """
        x: points for which to compute the interpolation value
        return: interpolation values at the point x
        """
        res = self.coef_[0]
        for i in range(1, len(self.coef_)):
            res += self.coef_[i]*np.prod([x-self.nodes[j] for j in range(i)], axis=0)
        return res
            
    
def solve_problem_2():
    segment = [-1, 1]
    x, y = generate_test_points(segment) # для валидации
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for i, n in enumerate([3, 5, 10, 15]):
        # равномерные узлы
        nodes = generate_interpolation_nodes_uniform(segment, n)
        values = compute_target_function_pr1(nodes)
        NP = NewtonPoly(nodes, values)
        r = compute_inaccuracy(y, NP.predict(x))
        print(f'n = {n}, uniform: r = {r}')
        
        ax[i].set_title(f'n = {n}')
        ax[i].plot(x, compute_target_function_pr1(x), label='f(x)')
        ax[i].plot(x, NP.predict(x), label='NP uniform')
        
        # случайные узлы
        nodes = generate_interpolation_nodes_random(segment, n)
        values = compute_target_function_pr1(nodes)
        NP = NewtonPoly(nodes, values)
        r = compute_inaccuracy(y, NP.predict(x))
        print(f'n = {n},  random: r = {r}')
        
        ax[i].plot(x, NP.predict(x), label='NP random')
        ax[i].legend()
        ax[i].grid()


solve_problem_2()