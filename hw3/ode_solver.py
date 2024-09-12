import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def compute_velocity(t, x, gamma=2.0):
    eps = 1.0e-6
    v = np.zeros(x.shape)
    v[0] = x[2]
    v[1] = x[3]
    v[2] = - gamma * x[0] / (np.sum(x * x) ** (1.5) + eps)
    v[3] = - gamma * x[1] / (np.sum(x * x) ** (1.5) + eps)
    return v


@jit(nopython=True, fastmath=True)
def compute_flow_rk_4(t, x, dt):
    w = np.zeros(6)
    w[0] = 16.0 / 135.0 - 1.0 / 360.0
    w[1] = 0.0 - 0.0
    w[2] = 6656.0 / 12825.0 + 128.0 / 4275.0
    w[3] = 28561.0 / 56430.0 + 2197.0 / 75240.0
    w[4] = -9.0 / 50.0 - 1.0 / 50.0
    w[5] = 2.0 / 55.0 - 2.0 / 55.0

    c = np.zeros(6)
    c[0] = 0.0
    c[1] = 0.25
    c[2] = 3.0 / 8.0
    c[3] = 12.0 / 13.0
    c[4] = 1.0
    c[5] = 0.5

    dmat = np.zeros((6, 6))
    dmat[1, 0] = 0.25
    dmat[2, 0] = 3.0 / 32.0
    dmat[2, 1] = 9.0 / 32.0
    dmat[3, 0] = 1932.0 / 2197.0
    dmat[3, 1] = -7200.0 / 2197.0
    dmat[3, 2] = 7296.0 / 2197.0
    dmat[4, 0] = 439.0 / 216.0
    dmat[4, 1] = -8.0
    dmat[4, 2] = 3680.0 / 513.0
    dmat[4, 3] = -845.0 / 4104.0
    dmat[5, 0] = -8.0 / 27.0
    dmat[5, 1] = 2.0
    dmat[5, 2] = -3544.0 / 2565.0
    dmat[5, 3] = 1859.0 / 4104.0
    dmat[5, 4] = -11.0 / 40.0

    fmat = np.zeros((w.size, x.size))
    for k in range(w.size):
        tau = t + c[k] * dt
        xxi = x + dt * np.dot(dmat[k, :], fmat)
        fmat[k, :] = compute_velocity(tau, xxi)
    v_numeric = np.dot(w, fmat)
    return v_numeric


@jit(nopython=True, fastmath=True)
def compute_numerical_solution_rk_4(t0, x0, t_final, nstep):
    dt = (t_final - t0) / nstep
    dimx = x0.size  # dimension of the problem
    x_numeric = np.zeros((nstep + 1, dimx))
    t_numeric = np.zeros((nstep + 1))
    x_numeric[0, :] = x0.copy()
    t_numeric[0] = t0
    # iterative procedure
    for kstep in range(nstep):
        v_numeric = compute_flow_rk_4(t_numeric[kstep], x_numeric[kstep, :], dt)
        t_numeric[kstep + 1] = t_numeric[kstep] + dt
        x_numeric[kstep + 1, :] = x_numeric[kstep, :] + dt * v_numeric
    return t_numeric, x_numeric


@jit(nopython=True, fastmath=True)
def compute_flow_rk_5(t, x, dt):
    w = np.zeros(6)
    w[0] = 16.0 / 135.0
    w[1] = 0.0
    w[2] = 6656.0 / 12825.0
    w[3] = 28561.0 / 56430.0
    w[4] = -9.0 / 50.0
    w[5] = 2.0 / 55.0

    c = np.zeros(6)
    c[0] = 0.0
    c[1] = 0.25
    c[2] = 3.0 / 8.0
    c[3] = 12.0 / 13.0
    c[4] = 1.0
    c[5] = 0.5

    dmat = np.zeros((6, 6))
    dmat[1, 0] = 0.25
    dmat[2, 0] = 3.0 / 32.0
    dmat[2, 1] = 9.0 / 32.0
    dmat[3, 0] = 1932.0 / 2197.0
    dmat[3, 1] = -7200.0 / 2197.0
    dmat[3, 2] = 7296.0 / 2197.0
    dmat[4, 0] = 439.0 / 216.0
    dmat[4, 1] = -8.0
    dmat[4, 2] = 3680.0 / 513.0
    dmat[4, 3] = -845.0 / 4104.0
    dmat[5, 0] = -8.0 / 27.0
    dmat[5, 1] = 2.0
    dmat[5, 2] = -3544.0 / 2565.0
    dmat[5, 3] = 1859.0 / 4104.0
    dmat[5, 4] = -11.0 / 40.0

    fmat = np.zeros((w.size, x.size))
    for k in range(w.size):
        tau = t + c[k] * dt
        xxi = x + dt * np.dot(dmat[k, :], fmat)
        fmat[k, :] = compute_velocity(tau, xxi)
    v_numeric = np.dot(w, fmat)
    return v_numeric


@jit(nopython=True, fastmath=True)
def compute_numerical_solution_rk_5(t0, x0, t_final, nstep):
    dt = (t_final - t0) / nstep
    dimx = x0.size  # dimension of the problem
    x_numeric = np.zeros((nstep + 1, dimx))
    t_numeric = np.zeros((nstep + 1))
    x_numeric[0, :] = x0.copy()
    t_numeric[0] = t0
    # iterative procedure
    x = x0.copy()
    t = 0.0 + t0
    for kstep in range(nstep):
        v_numeric = compute_flow_rk_5(t, x, dt)
        t_numeric[kstep + 1] = t_numeric[kstep] + dt
        x += dt * v_numeric
        t += dt
    return x


def compute_numerical_trajectory_rk_5(t0, x0, t_final, nstep):
    dt = (t_final - t0) / nstep
    dimx = x0.size  # dimension of the problem
    x_numeric = np.zeros((nstep + 1, dimx))
    t_numeric = np.zeros((nstep + 1))
    x_numeric[0, :] = x0.copy()
    t_numeric[0] = t0
    # iterative procedure
    x = x0.copy()
    t = 0.0 + t0
    for kstep in range(nstep):
        v_numeric = compute_flow_rk_5(t, x, dt)
        t_numeric[kstep + 1] = t_numeric[kstep] + dt
        x += dt * v_numeric
        t += dt
        x_numeric[kstep + 1, :] = x.copy()
    return x_numeric


@jit(nopython=True, fastmath=True)
def compute_numerical_solution_adaptive(t0, x0, t_final, target_tol, nstep):
    eps = 1.0e-20
    dt = (t_final - t0) / nstep
    dimx = x0.size  # dimension of the problem
    x_numeric = np.zeros((nstep + 1, dimx))
    t_numeric = np.zeros((nstep + 1))
    x_numeric[0, :] = x0.copy()
    t_numeric[0] = t0

    x_numeric = [x0.copy()]
    t_numeric = [t0]

    x_current = x0.copy()
    t_current = t0
    # iterative procedure
    while t_current < t_final:
        v_numeric_0 = compute_flow_rk_4(t_current, x_current, dt)
        v_numeric_1 = compute_flow_rk_5(t_current, x_current, dt)
        diff = dt * np.sqrt(np.sum((v_numeric_1 - v_numeric_0) * (v_numeric_1 - v_numeric_0)))
        dt = 0.9 * dt * np.exp(0.2 * np.log(target_tol / max(eps, diff)))

        t_current = t_current + dt
        x_current = x_current + dt * compute_flow_rk_5(t_current, x_current, dt)
        t_numeric.append(t_current)
        x_numeric.append(x_current)
    return np.asarray(t_numeric), np.asarray(x_numeric)


@jit(nopython=True, fastmath=True)
def compute_exact_solution(tdata, t0, x0):
    dimx = x0.size  # dimension of the problem
    ndata = tdata.size
    xdata = np.zeros((ndata, dimx))
    for kdata in range(ndata):
        xdata[kdata, :] = np.exp(-(tdata[kdata] - t0)) * x0
    return xdata


@jit(nopython=True, fastmath=True)
def compute_residual_precise_rk_4(t0, x0, t_final, nstep):
    t_numeric, x_numeric = compute_numerical_solution_rk_4(t0, x0, t_final, nstep)
    x_precise = compute_exact_solution(t_numeric, t0, x0)
    residual = np.sqrt(np.sum((x_numeric - x_precise) * (x_numeric - x_precise), axis=1))
    res = np.max(residual)
    return res


@jit(nopython=True, fastmath=True)
def compute_residual_precise_rk_5(t0, x0, t_final, nstep):
    t_numeric, x_numeric = compute_numerical_solution_rk_5(t0, x0, t_final, nstep)
    x_precise = compute_exact_solution(t_numeric, t0, x0)
    residual = np.sqrt(np.sum((x_numeric - x_precise) * (x_numeric - x_precise), axis=1))
    res = np.max(residual)
    return res
