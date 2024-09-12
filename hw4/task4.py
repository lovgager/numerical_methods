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
    nodes = np.empty(count_nodes)
    nodes[0] = segment[0]
    nodes[-1] = segment[-1]
    nodes[1:-1] = np.sort(np.random.uniform(segment[0], segment[1], count_nodes-2))
    return nodes


def generate_test_points(segment: list[float], count_nodes: int = 1000) -> (np.ndarray, np.ndarray):
    x = np.linspace(segment[0], segment[1], count_nodes + 1)
    y = compute_target_function_pr1(x)
    return x, y


def compute_inaccuracy(true_values, predict_values):
    return np.sqrt(np.mean((true_values - predict_values) ** 2))


class Spline:
    def __init__(self, nodes: np.ndarray, values: np.ndarray, left_first_der: float, right_first_der: float):
        self.nodes = nodes.copy()
        self.values = values.copy()
        self.left_first_derivative = left_first_der
        self.right_first_derivative = right_first_der
        
    @staticmethod
    def compute_cubic_spline_value(
        x_prev, x_curr, second_derivative_prev, second_derivative_curr, y_prev, y_curr, x
    ):
        h = x_curr - x_prev
        return second_derivative_curr/(6*h)*(x - x_prev)**3 + \
            second_derivative_prev/(6*h)*(x_curr - x)**3 + \
            1/h*(y_curr - second_derivative_curr*h**2/6)*(x - x_prev) + \
            1/h*(y_prev - second_derivative_prev*h**2/6)*(x_curr - x)

    def predict(self, x: np.ndarray):
        z = self.compute_second_deriv_vector()
        res = np.empty(x.shape)
        for k in range(len(self.nodes) - 1):
            x_curr = self.nodes[k+1]
            x_prev = self.nodes[k]
            y_curr = self.values[k+1]
            y_prev = self.values[k]
            second_derivative_curr = z[k+1]
            second_derivative_prev = z[k]
            slice_index = (x_prev <= x)*(x <= x_curr)
            x_slice = x[slice_index]
            res[slice_index] = self.compute_cubic_spline_value(x_prev, x_curr,
                second_derivative_prev, second_derivative_curr, y_prev, y_curr, x_slice)
        return res
            
    def compute_system_matrix(self):
        h = np.diff(self.nodes)
        u = np.empty(len(self.nodes))
        u[1:-1] = 2*(h[1:] + h[:-1])
        u[0] = 2*h[0]
        u[-1] = 2*h[-1]
        return np.diag(u) + np.diag(h, k=1) + np.diag(h, k=-1)

    def compute_right_part(self):
        h = np.diff(self.nodes)
        b = 6/h*np.diff(self.values)
        v = np.empty(len(self.nodes))
        v[1:-1] = np.diff(b)
        v[0] = b[0] - 6*self.left_first_derivative
        v[-1] = 6*self.right_first_derivative - b[-1]
        return v

    def compute_second_deriv_vector(self):
        A = self.compute_system_matrix()
        f = self.compute_right_part()
        return np.linalg.solve(A, f)
    

def solve_problem_4():
    segment = [-1, 1]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for i, count_nodes in enumerate([3, 5, 8, 10]):
        nodes = generate_interpolation_nodes_random(segment, count_nodes)
        values = compute_target_function_pr1(nodes)
        left_der = compute_target_first_derivative_function_pr1(-1)
        right_der = compute_target_first_derivative_function_pr1(1)
        s = Spline(nodes, values, left_der, right_der)
        x, y = generate_test_points(segment)
        r = compute_inaccuracy(y, s.predict(x))
        print(f'n = {count_nodes}, random: r = {r}')
        
        ax[i].set_title(f'n = {count_nodes}')
        ax[i].scatter(nodes, values)
        ax[i].plot(x, compute_target_function_pr1(x), label='f(x)')
        ax[i].plot(x, s.predict(x), label='spline random')
        ax[i].legend()
        ax[i].grid()
        
solve_problem_4()