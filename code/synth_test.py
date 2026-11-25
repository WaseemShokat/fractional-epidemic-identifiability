"""Synthetic recovery test for Experiment A (He Li's point 1 and 3).

Data are generated from a fractional SIR model with a KNOWN order alpha (same order in all equations):
    D^a S = -b S I,   D^a I = b S I - g I,   D^a R = g I        (t in weeks, fractions of the population)
solved with the fractional Adams-Bashforth-Moulton predictor-corrector (Diethelm et al. 2002) on a fine grid,
then sampled weekly (like the real data), optionally with multiplicative noise.
The weekly data then go through exactly the same pipeline as the real data (spline to h = 0.25 week,
scaling by the maximum, L1 Caputo derivative, STRidge library, order selection) - functions in fde_core.py.

Scenarios
  states : S and I observed (R = 1 - S - I is left out, otherwise the library is collinear;
           the true equations lie in the degree-2 library)  -> can the code recover alpha and terms?
  counts : only the weekly flows C = b S I (new cases) and G = g I (recoveries) observed, as with real
           case/death counts. The G-equation is exactly in the library (D^a G = g C - g G); the C-equation is not.
Usage: python3 synth_test.py out.json
"""
import sys, json, time, numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma as G
import fde_core as fc

B, GM, I0, NTRAIN, NFC = 1.5, 0.7, 1e-3, 25, 4      # beta, gamma per week; weeks 0..24 train, 25..28 forecast
H, LAM = 0.25, 1e-3


def frac_abm(f, y0, alpha, T, dt):
    n = int(round(T / dt)); y = np.zeros((n + 1, len(y0))); y[0] = y0; F = np.zeros_like(y); F[0] = f(y0)
    c1 = dt ** alpha / G(alpha + 1); c2 = dt ** alpha / G(alpha + 2)
    for k in range(n):
        j = np.arange(k + 1)
        b = (k + 1 - j) ** alpha - (k - j) ** alpha
        yp = y0 + c1 * (b @ F[:k + 1])
        a = (k - j + 2.) ** (alpha + 1) + (k - j + 0.) ** (alpha + 1) - 2 * (k - j + 1.) ** (alpha + 1)
        a[0] = k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha
        y[k + 1] = y0 + c2 * (f(yp) + a @ F[:k + 1]); F[k + 1] = f(y[k + 1])
    return y


def sir(y): S, I, R = y; return np.array([-B * S * I, B * S * I - GM * I, GM * I])


def simulate(alpha, dt=0.01):
    T = NTRAIN + NFC
    y = frac_abm(sir, np.array([1 - I0, I0, 0.]), alpha, T, dt)
    idx = np.round(np.arange(T + 1) / dt).astype(int)
    return y[idx]                       # weekly states, weeks 0..T


def to_grid(W):
    tw = np.arange(len(W)); sc = W.max(0) + 1e-12
    tg = np.arange(0, len(W) - 1 + 1e-9, H)
    return np.vstack([CubicSpline(tw, W[:, k] / sc[k])(tg) for k in range(W.shape[1])]).T.clip(0), sc


def true_support(scn, deg):
    _, nm = fc.library(np.zeros((1, 2)), deg)
    if scn == 'states': want = [{'x0*x1'}, {'x0*x1', 'x1'}]
    else: want = [None, {'x0', 'x1'}]   # C-eq not in library; G-eq: g C - g G
    return nm, want


def run(scn, alpha, noise, mode, seed=0):
    rng = np.random.default_rng(seed)
    Y = simulate(alpha)
    W = Y[:, :2] if scn == 'states' else np.vstack([B * Y[:, 0] * Y[:, 1], GM * Y[:, 1]]).T
    Wn = W * (1 + noise * rng.standard_normal(W.shape)) if noise > 0 else W.copy()
    Wtr, Wte = Wn[:NTRAIN], W[NTRAIN:NTRAIN + NFC]
    Yg, sc = to_grid(Wtr); nv = Yg.shape[1]
    t1 = time.time()
    if mode == 'fit':
        deg, tol = 2, 0.1
        res = [fc.select_fit(Yg, k, LAM, tol, deg, H) for k in range(nv)]
        al = [r[0] for r in res]; xis = [r[1] for r in res]
    else:
        e, al, tol, deg = fc.select_val(Yg, LAM, H)
        xis = [fc.fit_eq(Yg, k, al[k], tol, LAM, deg, H)[0] for k in range(nv)]
    nm, want = true_support(scn, deg)
    terms = [sorted(nm[i] for i in range(len(x)) if x[i] != 0) for x in xis]
    ok = [None if w is None else set(t) == w for t, w in zip(terms, want)]
    Yf = fc.forecast(Yg, al, xis, 4 * NFC, deg, H, clip=None)[3::4] * sc
    fc_err = float(np.mean(np.abs(Yf - Wte)) / np.mean(np.abs(Wte)))
    return dict(scn=scn, alpha_true=alpha, noise=noise, mode=mode, deg=deg, tol=tol,
                alpha_hat=[round(x, 3) for x in al], terms=terms, support_ok=ok,
                coefs=[{nm[i]: round(float(x[i]), 3) for i in range(len(x)) if x[i] != 0} for x in xis],
                fc_rel_err=round(fc_err, 3), secs=round(time.time() - t1, 1))


if __name__ == '__main__':
    out = []
    for mode in ('fit', 'val'):
      for scn in ('states', 'counts'):
        for alpha in (0.7, 0.85, 1.0):
            for noise in (0.0, 0.05):
                    r = run(scn, alpha, noise, mode); out.append(r)
                    print(r['scn'], r['alpha_true'], r['noise'], r['mode'], r['alpha_hat'], r['support_ok'], r['fc_rel_err'], r['secs'], flush=True)
                    json.dump(out, open(sys.argv[1], 'w'), indent=1)
