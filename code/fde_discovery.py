"""Experiment A (He Li, 21 Nov): learn a fractional ODE directly from the data, then forecast.

Follows the idea of Yu et al. (arXiv 2412.03970): sparse regression (STRidge) over a library of
candidate terms, alternated with differential evolution (DE) for the fractional orders.
Adapted to the weekly California series y = (cases, deaths, hospital admissions):

    D^{alpha_k} y_k(t) = Theta(y(t)) . xi_k ,  k = 1..3   (Caputo, t in weeks)

1. the training series (weeks TRAIN_START .. tr_end-1) is interpolated by a cubic spline on a
   grid with step h = 0.25 week and scaled by its maximum;
2. the Caputo derivative of each series is computed with the L1 scheme;
3. for each equation, xi_k is found by STRidge for a given alpha_k, and alpha_k by DE;
4. the discovered system is integrated forward (product-integration rule, keeping the whole
   observed history in the memory term) to forecast 1-4 weeks ahead.
--alpha1 fixes all orders to 1 (integer-order equation discovery = SINDy) for comparison.
Output JSON has the same format as fpinn_qian.py, so compare2.py can read it.
"""
import argparse, json, time, math, itertools
import numpy as np, scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution
from scipy.special import gamma as G

p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--out', required=True)
p.add_argument('--weeks', required=True)
p.add_argument('--alpha1', action='store_true')
p.add_argument('--tol', type=float, default=0.05)
p.add_argument('--lam', type=float, default=1e-3)
p.add_argument('--deg', type=int, default=2)
p.add_argument('--amin', type=float, default=0.3)
p.add_argument('--start', type=int, default=20)
p.add_argument('--val', type=int, default=2)
a = p.parse_args()

d = sio.loadmat(a.data)
OBS = np.vstack([np.nan_to_num(d[k].ravel().astype(float)).clip(0) for k in ('cases', 'deaths', 'hospitalized')]).T
NAMES = ['1'] + ['C', 'D', 'H']
H = 0.25

DEG = [2]
def library(Y):
    cols = [np.ones(len(Y))]; names = ['1']
    for i in range(3): cols.append(Y[:, i]); names.append('CDH'[i])
    if DEG[0] >= 2:
        for i, j in itertools.combinations_with_replacement(range(3), 2):
            cols.append(Y[:, i] * Y[:, j]); names.append('CDH'[i] + 'CDH'[j])
    return np.vstack(cols).T, names

def caputo_L1(u, alpha, h):
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

def fit_eq(Yg, k, alpha, tol=None):
    lhs = caputo_L1(Yg[:, k], alpha, H)
    Th, _ = library(Yg[1:])
    xi = stridge(Th, lhs, a.lam, a.tol if tol is None else tol)
    r = lhs - Th @ xi
    return xi, float(np.mean(r ** 2) / (np.mean(lhs ** 2) + 1e-12))

def forecast(Yg, alphas, xis, n_ahead):
    """Product-integration rule for D^a y = f(y), memory from the observed history.
    y_n = y_last + [I^a f](t_n) - [I^a f](t_last), so the forecast starts exactly at the data."""
    n0 = len(Yg); Y = list(Yg); F = [library(Yg[i:i + 1])[0][0] for i in range(n0)]
    def I(n, k, al):
        j = np.arange(n); w = ((n - j) ** al - (n - j - 1) ** al) * H ** al / G(al + 1)
        return np.dot(w, np.array([F[i] @ xis[k] for i in range(n)]))
    base = [I(n0 - 1, k, alphas[k]) for k in range(3)]
    for n in range(n0, n0 + n_ahead):
        y = np.array([Yg[-1, k] + I(n, k, alphas[k]) - base[k] for k in range(3)])
        y = np.clip(y, 0, 3.0)
        Y.append(y); F.append(library(y[None])[0][0])
    return np.array(Y[n0:])

def weeks(w):
    if ':' in w: lo, hi = map(int, w.split(':')); return list(range(lo, hi + 1))
    return [int(x) for x in w.split(',')]

out = {'args': vars(a), 'windows': []}
for tr_end in weeks(a.weeks):
    t1 = time.time()
    tw = np.arange(a.start, tr_end); yw = OBS[a.start:tr_end]
    sc = yw.max(0) + 1e-9
    tg = np.arange(a.start, tr_end - 1 + 1e-9, H)
    Yg = np.vstack([CubicSpline(tw, yw[:, k] / sc[k])(tg) for k in range(3)]).T.clip(0)
    # model selection by held-out forecast: fit on all but the last VAL weeks, forecast them
    # rolling-origin validation: refit with the last o weeks removed (o = 2, 3, 4) and forecast
    # the next 2 weeks; unstable forecasts (leaving [0, 3x training max]) are penalised
    def val_err(al, tol):
        errs = []
        for o in (2, 3, 4):
            Ytr, Yva = Yg[:-4 * o], Yg[-4 * o:][:8]
            xs = [fit_eq(Ytr, k, al[k], tol)[0] for k in range(3)]
            Yp = forecast(Ytr, al, xs, 16)
            if (Yp >= 2.99).any() or (Yp <= 1e-6).all(axis=0).any(): return 10.0
            errs.append(np.mean(np.abs(Yp[:8] - Yva)))
        return float(np.mean(errs))
    tols = [0.05, 0.1, 0.2, 0.4, 0.7]
    best = None
    for deg, tol in itertools.product([1, 2], tols):
        DEG[0] = deg
        if a.alpha1:
            al = [1.0, 1.0, 1.0]; e = val_err(al, tol)
        else:
            r = differential_evolution(lambda v: val_err(list(v), tol), [(a.amin, 1.0)] * 3,
                                       seed=0, tol=1e-3, maxiter=25, popsize=8, polish=False)
            al, e = list(r.x), float(r.fun)
        if best is None or e < best[0]: best = (e, al, tol, deg)
    verr, alphas, tol, DEG[0] = best
    alphas = [float(x) for x in alphas]
    xis, info = [], []
    for k in range(3):
        xi, loss = fit_eq(Yg, k, alphas[k], tol)
        xis.append(xi)
        _, nm = library(Yg[:1])
        info.append({'alpha': round(alphas[k], 3), 'rel_resid': round(loss, 4), 'tol': tol, 'deg': DEG[0], 'val_err': round(verr, 4),
                     'terms': {nm[i]: round(float(xi[i]), 4) for i in range(len(xi)) if xi[i] != 0}})
    Yf = forecast(Yg, alphas, xis, 16)          # 16 quarter-weeks = 4 weeks
    P = Yf[3::4] * sc                            # weeks tr_end .. tr_end+3
    preds = [[float(P[h, 0]), float(P[h, 1]), float(P[h, 2])] for h in range(4)]
    out['windows'].append({'train_end': tr_end, 'preds': preds, 'z': [alphas], 'eqs': info,
                           'secs': round(time.time() - t1, 1)})
    print(tr_end, [i['alpha'] for i in info], [len(i['terms']) for i in info],
          [round(x) for x in P[:, 0]], flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
