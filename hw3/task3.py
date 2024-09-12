import numpy as np
import scipy.special
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from sklearn import linear_model
from tqdm import tqdm


def compute_poly_norm(poly_degree):
    log_factor = 0.5 * np.log(2.0 * np.pi)
    if poly_degree > 1:
        log_factor += np.sum(np.log(np.linspace(1.0, poly_degree, poly_degree)))
    return np.exp(0.5 * log_factor)


def make_regression_problem_3(npoly, ndata, top_fraction=0.3, noise_std=0.01):
    poly_coef = np.random.normal(0.0, 1.0, npoly)
    xdata = np.random.normal(0.0, 1.0, ndata)
    threshold = np.quantile(np.absolute(poly_coef), 1.0 - top_fraction)
    for idx in range(poly_coef.size):
        if np.absolute(poly_coef[idx]) < threshold:
            poly_coef[idx] = 0.0
    feature_matrix = np.zeros((ndata, npoly))
    for idx in range(npoly):
        feature_matrix[:, idx] = scipy.special.eval_hermitenorm(idx, xdata) / compute_poly_norm(idx)
    ydata = np.dot(feature_matrix, poly_coef)
    ydata += noise_std * np.random.normal(0.0, 1.0, ndata)
    return feature_matrix, ydata, poly_coef


class CoordinateDescent:
    def __init__(self, alpha=1.0e-6, num_iter=1000):
        self.alpha = alpha
        self.num_iter = num_iter
        self.coef_ = None


    def minimize(self, func, initial_guess, tol=1e-6, step=0.1):
        coefs = initial_guess.copy()
        prev_loss = func(coefs)
        num_coefs = len(coefs)
        for it in tqdm(range(self.num_iter)):
            for i in range(num_coefs):
                coefs_try = coefs.copy()
                coefs_try[i] += step
                loss_try = func(coefs_try)
                loss_cur = func(coefs)
                if loss_try < loss_cur:
                    coefs[i] = coefs_try[i]
                else:
                    coefs_try[i] -= 2*step
                    loss_try = func(coefs_try)
                    if loss_try < loss_cur:
                        coefs[i] = coefs_try[i]
            if np.abs(prev_loss - func(coefs)) < tol:
                step *= 0.5
            prev_loss = func(coefs)
        return coefs


    def fit(self, xdata, ydata):
        c = np.zeros(xdata.shape[1])
        L = lambda c: 0.5*np.mean((ydata - np.dot(xdata, c))**2) + \
            self.alpha*np.linalg.norm(c, ord=1)
        self.coef_ = self.minimize(L, c)
                

    def predict(self, xdata):
        return np.dot(xdata, self.coef_)
    
    
def make_plot_problem_3():
    npoly = 11
    ndata = 500
    xdata, ydata, cdata = make_regression_problem_3(npoly, ndata, top_fraction=0.2, noise_std=0.001)
    print(cdata)
    baseline_model = linear_model.Lasso(alpha=1.0e-6, fit_intercept=False, tol=1.0e-8, max_iter=100000)
    baseline_model.fit(xdata, ydata)
    y_baseline_pred = baseline_model.predict(xdata)

    custom_model = CoordinateDescent()
    custom_model.fit(xdata, ydata)
    y_custom_pred = custom_model.predict(xdata)

    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(12,6))

    ax[0].scatter(ydata, y_baseline_pred, label='baseline model vs ground truth')
    ax[0].scatter(ydata, y_custom_pred, label='custom model vs ground truth')
    ax[0].set_title('Target vs baseline \n and custom model prediction')
    ax[0].grid()
    ax[0].legend()

    ax[1].scatter(cdata, baseline_model.coef_, label='baseline model vs ground truth')
    ax[1].scatter(cdata, custom_model.coef_, label='custom model vs ground truth')
    ax[1].set_title('Coefficient vs baseline \n and custom model coefficient')
    ax[1].grid()
    ax[1].legend()
    plt.show()

    print(f'Baseline model inaccuracy: {np.max(np.absolute(cdata - baseline_model.coef_))}')
    print(f'Custom model inaccuracy: {np.max(np.absolute(cdata - custom_model.coef_))}')

np.random.seed(1)
make_plot_problem_3()

