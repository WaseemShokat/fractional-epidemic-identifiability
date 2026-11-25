"""Core routines of Experiment A (learning a fractional ODE from data), shared by fde_discovery.py
(real data) and synth_test.py (synthetic recovery test). Same algorithms as the original fde_discovery.py.

Two ways to choose the fractional orders:
  'fit' - as in Yu et al. (arXiv:2412.03970, eq. 14 and steps 1-3): alternate STRidge for the
          coefficients xi at fixed alpha, and differential evolution (DE) for alpha at fixed xi,
          minimising ||D^alpha u - Theta xi||^2 + lam ||xi||^2.
  'val' - our modification: choose alpha (and threshold, library degree) by rolling-origin forecast
          validation, rejecting equations whose forecasts blow up.
"""
import itertools, numpy as np
from scipy.optimize import differential_evolution
from scipy.special import gamma as G


def library(Y, deg, names=None):
    nv = Y.shape[1]; names = names or [f'x{i}' for i in range(nv)]
    cols = [np.ones(len(Y))]; nm = ['1']
    for i in range(nv): cols.append(Y[:, i]); nm.append(names[i])
    if deg >= 2:
        for i, j in itertools.combinations_with_replacement(range(nv), 2):
            cols.append(Y[:, i] * Y[:, j]); nm.append(names[i] + '*' + names[j])
    return np.vstack(cols).T, nm


def caputo_L1(u, alpha, h):
    """Caputo derivative of order alpha in (0,1] at grid points 1..n-1 (L1 scheme)."""
    n = len(u); k = np.arange(n)
    b = (k + 1) ** (1 - alpha) - k ** (1 - alpha); b[0] = 1.0   # avoid 0**0 = 1 when alpha = 1
    du = np.diff(u); out = np.zeros(n - 1)
    for i in range(1, n):
        out[i - 1] = np.dot(b[:i][::-1], du[:i])
    return out / (h ** alpha * G(2 - alpha))


def stridge(X, y, lam, tol, iters=10):
    ys = np.linalg.norm(y) + 1e-12; y = y / ys
    nrm = np.linalg.norm(X, axis=0) + 1e-12; Xn = X / nrm
    w = np.linalg.lstsq(Xn.T @ Xn + lam * np.eye(X.shape[1]), Xn.T @ y, rcond=None)[0]
    big = np.abs(w) >= tol
    for _ in range(iters):
        if big.sum() == 0: break
        w = np.zeros_like(w)
        Xb = Xn[:, big]
        w[big] = np.linalg.lstsq(Xb.T @ Xb + lam * np.eye(big.sum()), Xb.T @ y, rcond=None)[0]
        nb = np.abs(w) >= tol
        if (nb == big).all(): break
        big = nb
    return w / nrm * ys


def fit_eq(Yg, k, alpha, tol, lam, deg, h):
    lhs = caputo_L1(Yg[:, k], alpha, h)
    Th, _ = library(Yg[1:], deg)
    xi = stridge(Th, lhs, lam, tol)
    r = lhs - Th @ xi
    return xi, float(np.mean(r ** 2) / (np.mean(lhs ** 2) + 1e-12))


def forecast(Yg, alphas, xis, n_ahead, deg, h, clip=3.0):
    """Product-integration rule for D^a y = f(y), memory from the observed history.
    y_n = y_last + [I^a f](t_n) - [I^a f](t_last), so the forecast starts exactly at the data."""
    nv = Yg.shape[1]
    n0 = len(Yg); F = [library(Yg[i:i + 1], deg)[0][0] for i in range(n0)]
    def I(n, k, al):
        j = np.arange(n); w = ((n - j) ** al - (n - j - 1) ** al) * h ** al / G(al + 1)
        return np.dot(w, np.array([F[i] @ xis[k] for i in range(n)]))
    base = [I(n0 - 1, k, alphas[k]) for k in range(nv)]
    Y = []
    for n in range(n0, n0 + n_ahead):
        y = np.array([Yg[-1, k] + I(n, k, alphas[k]) - base[k] for k in range(nv)])
        if clip is not None: y = np.clip(y, 0, clip)
        Y.append(y); F.append(library(y[None], deg)[0][0])
    return np.array(Y)


def val_err(Yg, al, tol, lam, deg, h, steps=4):
    """Rolling-origin validation: refit with the last o weeks removed (o = 2, 3, 4), forecast 2 weeks."""
    errs = []
    for o in (2, 3, 4):
        Ytr, Yva = Yg[:-steps * o], Yg[-steps * o:][:2 * steps]
        xs = [fit_eq(Ytr, k, al[k], tol, lam, deg, h)[0] for k in range(Yg.shape[1])]
        Yp = forecast(Ytr, al, xs, 4 * steps, deg, h)
        if (Yp >= 2.99).any() or (Yp <= 1e-6).all(axis=0).any(): return 10.0
        errs.append(np.mean(np.abs(Yp[:2 * steps] - Yva)))
    return float(np.mean(errs))


def select_val(Yg, lam, h, amin=0.3, alpha1=False, tols=(0.05, 0.1, 0.2, 0.4, 0.7), degs=(1, 2)):
    nv = Yg.shape[1]; best = None
    for deg, tol in itertools.product(degs, tols):
        if alpha1:
            al = [1.0] * nv; e = val_err(Yg, al, tol, lam, deg, h)
        else:
            r = differential_evolution(lambda v: val_err(Yg, list(v), tol, lam, deg, h), [(amin, 1.0)] * nv,
                                       seed=0, tol=1e-3, maxiter=25, popsize=8, polish=False)
            al, e = list(r.x), float(r.fun)
        if best is None or e < best[0]: best = (e, [float(x) for x in al], tol, deg)
    return best   # (val_err, alphas, tol, deg)


def select_fit(Yg, k, lam, tol, deg, h, amin=0.3, rounds=6):
    """Yu et al.: alternate STRidge (xi | alpha) and DE (alpha | xi) on the fit loss, one equation."""
    al = 1.0
    Th, _ = library(Yg[1:], deg)
    for _ in range(rounds):
        xi, _ = fit_eq(Yg, k, al, tol, lam, deg, h)
        obj = lambda v: float(np.sum((caputo_L1(Yg[:, k], v[0], h) - Th @ xi) ** 2) + lam * np.sum(xi ** 2))
        r = differential_evolution(obj, [(amin, 1.0)], seed=0, tol=1e-6, maxiter=50, popsize=10, polish=True)
        new = float(r.x[0])
        if abs(new - al) < 1e-3: al = new; break
        al = new
    xi, res = fit_eq(Yg, k, al, tol, lam, deg, h)
    return al, xi, res
