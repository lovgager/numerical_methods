import numpy as np
import scipy.special
import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
from scipy.special import eval_hermitenorm
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_poly_norm(poly_degree):
    log_factor = 0.5 * np.log(2.0 * np.pi)
    if poly_degree > 1:
        log_factor += np.sum(np.log(np.linspace(1.0, poly_degree, poly_degree)))
    return np.exp(0.5 * log_factor)


def make_regression_problem_4(npoly, ndata, top_fraction=0.3, noise_std=0.01):
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


def sample_from_posterior_distribution(feature_matrix, ydata, sigma_pr, sigma_lh, nsample):
    ndata, nfeat = feature_matrix.shape
    hmat = np.dot(np.transpose(feature_matrix), feature_matrix) / ndata
    hmat += sigma_lh * sigma_lh / sigma_pr / sigma_pr / ndata * np.eye(nfeat)
    bvec = np.dot(ydata, feature_matrix) / ndata
    cmean = np.linalg.solve(hmat, bvec)

    eigh_val, eigh_vec = np.linalg.eigh(hmat)
    eigh_val = sigma_lh / np.sqrt(ndata * eigh_val)
    eigh_vec = np.transpose(eigh_vec)
    dmat = np.zeros((nfeat, nfeat))
    for idx in range(nfeat):
        dmat[idx, idx] = eigh_val[idx]
    csample = np.random.normal(0.0, 1.0, (nsample, nfeat))
    csample = np.dot(csample, dmat)
    csample = np.dot(csample, eigh_vec)
    for idx in range(nsample):
        csample[idx, :] += cmean
    return csample


def compute_log_likelihood_gradient(feature_matrix, ydata, cdata, sigma_lh):
    ndata, nfeat = feature_matrix.shape
    rdata = ydata - np.dot(feature_matrix, cdata)
    mse = np.dot(rdata, rdata)
    llh_grad = ndata / sigma_lh * (mse / sigma_lh / sigma_lh / ndata - 1.0)
    return llh_grad


def compute_log_evidence_grad(feature_matrix, ydata, sigma_pr, sigma_lh, num_sample):
    csample = sample_from_posterior_distribution(feature_matrix, ydata, sigma_pr, sigma_lh, num_sample)
    lev_grad = 0.0
    for idx in range(num_sample):
        lev_grad += compute_log_likelihood_gradient(feature_matrix, ydata, csample[idx, :], sigma_lh) / num_sample
    return lev_grad


def init_stochastic_gradient_descent(dim):
    return np.random.normal(0.0, 1.0, dim)


def single_step_stochastic_gradient_descent(
    feature_matrix, ydata, num_samples,
    s_k, dt, alpha, iteration_number
):
    g_k = compute_log_evidence_grad(feature_matrix, ydata, 1.0, 1.0, num_samples)
    return s_k + dt/(1 + iteration_number)**alpha*np.exp(s_k)*g_k
    

def compute_optimium_stochastic_gradient_descent(
    feature_matrix, ydata, num_samples, dt, alpha, num_iter
):
    s = init_stochastic_gradient_descent(feature_matrix.shape[1])
    for k in range(num_iter):
        s = single_step_stochastic_gradient_descent(feature_matrix, ydata, num_samples, s, dt, alpha, k)
    return s


def compute_s_param(dt, alpha, feature_matrix, ydata, num_samples, num_iter):
    s = compute_optimium_stochastic_gradient_descent(feature_matrix, ydata, num_samples, dt, alpha, num_iter)
    sigma = np.exp(s)
    grad = compute_log_evidence_grad(feature_matrix, ydata, 1.0, sigma, num_iter)
    return grad, sigma, s


def solve_problem_4():
    # prior sigma is always 1.0
    dt_array = [0.001, 0.005, 0.0001, 0.0005, 0.0007, 0.00001, 0.00005]
    alpha_array = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    npoly = 3
    ndata = 1000
    num_samples = 100
    num_iter = 1000
    feature_matrix, ydata, _ = make_regression_problem_4(npoly, ndata, noise_std=0.01)
    sigma_opt = 0.01*np.ones(npoly)
    for dt in tqdm(dt_array):
        for alpha in alpha_array:
            evidence_gradient, sigma, s = compute_s_param(dt, alpha, feature_matrix, ydata, num_samples, num_iter)
            if np.linalg.norm(sigma) < np.linalg.norm(sigma_opt):
                sigma_opt[:] = sigma
                alpha_opt = alpha
                dt_opt = dt
    print()
    print(f'Optimal alpha: {alpha_opt}')
    print(f'Optimal dt: {dt_opt}')
    print(f'Optimal sigma: {sigma_opt}')

solve_problem_4()
