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
    res = np.linspace(segment[0], segment[1], count_nodes)
    return np.sort(np.append(res, [res[1], res[2], res[2]]))


def generate_test_points(segment: list[float], count_nodes: int = 1000) -> (np.ndarray, np.ndarray):
    x = np.linspace(segment[0], segment[1], count_nodes + 1)
    y = compute_target_function_pr1(x)
    return x, y


def compute_inaccuracy(true_values, predict_values):
    return np.sqrt(np.mean((true_values - predict_values) ** 2))


class NewtonDerivativePoly:
    """
    You are not allowed to use a for-loop over x
    """

    def __init__(self, nodes: np.ndarray, values: np.ndarray, max_der_order: int):
        self.nodes = nodes
        self.values = values # матрица из значений f(x) и её первой и второй производных
        self.coef_ = self.divided_differences(max_der_order=max_der_order)

    def divided_differences(self, max_der_order: int):
        """
        Compute divided differences
        :return: polynomial coefficients
        """
        def f(i, j): # f[xi, ..., xj]
            if self.nodes[i] == self.nodes[j]:
                return self.values[j-i, i]
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
            
    
def solve_problem_3():
    segment = [-1, 1]
    x, y = generate_test_points(segment) # для валидации
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for i, n in enumerate([3, 5, 10, 15]):
        # равномерные узлы
        nodes = generate_interpolation_nodes_uniform(segment, n)
        values = np.empty((3, len(nodes)))
        for k in range(3):
            values[k] = compute_target_function_derivative(nodes, k)
        NDP = NewtonDerivativePoly(nodes, values, 2)
        r = compute_inaccuracy(y, NDP.predict(x))
        print(f'n = {n}: r = {r}')
        
        ax[i].set_title(f'n = {n}')
        ax[i].plot(x, compute_target_function_pr1(x), label='f(x)')
        ax[i].plot(x, NDP.predict(x), label='NDP uniform')
        ax[i].legend()
        ax[i].grid()
        

solve_problem_3()
