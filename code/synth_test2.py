"""Synthetic recovery tests for Experiment A (answers He Li's points 1-3). Uses fde_core.py (same code as
the real-data runs) and the fractional SIR data generator of synth_test.py.

T1  code check: structure known (only the true terms S*I and I), order fitted by least squares on the L1
    derivative. 20 noise replicates per setting -> mean and sd of the estimated order.
T2  full equation learning with the fit criterion of Yu et al. (residual), made sparse by a BIC penalty,
    jointly over the order grid 0.40..1.00 and the STRidge threshold: estimated order + correct terms?
    Also with DAILY instead of weekly sampling (7x more data).
T2b the literal alternating scheme of Yu et al. (STRidge xi | alpha, DE alpha | xi) from two start values.
T3  our forecast-validation selection (as used on the real data): order, terms, 4-week forecast error.
T4  counts only (new cases C = bSI and recoveries G = gI): G-equation known structure (D^a G = gC - gG)
    and full learning for both equations.
"""
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, sys.path[0])
import synth_test as st, fde_core as fc
H, LAM = 0.25, 1e-3
TOLS = (0.05, 0.1, 0.2, 0.4, 0.7)
AGRID = np.round(np.arange(0.40, 1.0001, 0.025), 3)


def data(alpha, noise, seed, obs='states', daily=False):
    rng = np.random.default_rng(seed)
    if daily:
        T = st.NTRAIN + st.NFC
        y = st.frac_abm(st.sir, np.array([1 - st.I0, st.I0, 0.]), alpha, T, 0.01)
        idx = np.round(np.arange(0, st.NTRAIN - 1 + 1e-9, 1 / 7) / 0.01).astype(int); Y = y[idx]
    else:
        Y = st.simulate(alpha)[:st.NTRAIN]
    W = Y[:, :2] if obs == 'states' else np.vstack([st.B * Y[:, 0] * Y[:, 1], st.GM * Y[:, 1]]).T
    if noise > 0: W = W * (1 + noise * rng.standard_normal(W.shape))
    if daily:   # spline in weeks with 1/7-week knots
        tw = np.arange(len(W)) / 7; sc = W.max(0)
        from scipy.interpolate import make_smoothing_spline   # smoothing spline (GCV) for the noisy daily data
        tg = np.arange(0, tw[-1] + 1e-9, H)
        return np.vstack([make_smoothing_spline(tw, W[:, k] / sc[k])(tg) for k in range(2)]).T.clip(0), sc
    return st.to_grid(W)


def ls_order(Yg, k, cols):
    Th, _ = fc.library(Yg[1:], 2); best = None
    for al in np.arange(0.3, 1.0001, 0.005):
        lhs = fc.caputo_L1(Yg[:, k], al, H); X = Th[:, cols]
        c = np.linalg.lstsq(X, lhs, rcond=None)[0]; r = np.mean((lhs - X @ c) ** 2) / np.mean(lhs ** 2)
        if best is None or r < best[0]: best = (r, al)
    return best[1]


def bic_select(Yg, k):
    n = len(Yg) - 1; best = None; _, nm = fc.library(Yg[:1], 2)
    for al in AGRID:
        for tol in TOLS:
            xi, r = fc.fit_eq(Yg, k, al, tol, LAM, 2, H); kk = int((xi != 0).sum())
            b = n * np.log(r + 1e-12) + kk * np.log(n)
            if best is None or b < best[0]: best = (b, float(al), sorted(nm[i] for i, x in enumerate(xi) if x != 0))
    return best[1], best[2]


TRUE = {'states': [{'x0*x1'}, {'x0*x1', 'x1'}], 'counts': [None, {'x0', 'x1'}]}
R = {'T1': [], 'T2': [], 'T2b': [], 'T3': [], 'T4': []}
out = sys.argv[1]
for a in (0.7, 0.85, 1.0):
    for nz in (0.0, 0.02, 0.05):
        reps = range(20) if nz > 0 else range(1)
        e0 = [ls_order(data(a, nz, s)[0], 0, [4]) for s in reps]; e1 = [ls_order(data(a, nz, s)[0], 1, [2, 4]) for s in reps]
        R['T1'].append(dict(alpha=a, noise=nz, S_mean=float(np.mean(e0)), S_sd=float(np.std(e0)), I_mean=float(np.mean(e1)), I_sd=float(np.std(e1))))
        for daily in (False, True):
            for s in (range(5) if nz > 0 else range(1)):
                Yg, _ = data(a, nz, s, daily=daily)
                res = [bic_select(Yg, k) for k in (0, 1)]
                R['T2'].append(dict(alpha=a, noise=nz, daily=daily, seed=s, alpha_hat=[r[0] for r in res], terms=[r[1] for r in res],
                                    ok=[set(r[1]) == TRUE['states'][k] for k, r in enumerate(res)]))
        print('T1/T2', a, nz, R['T1'][-1], flush=True)
        json.dump(R, open(out, 'w'), indent=1)
# T2b literal alternating scheme
for a in (0.7, 1.0):
    for nz in (0.0, 0.05):
        Yg, _ = data(a, nz, 0)
        for start in (1.0, 0.5):
            row = []
            for k in (0, 1):
                al = start
                Th, _ = fc.library(Yg[1:], 2)
                for _ in range(6):
                    xi, _ = fc.fit_eq(Yg, k, al, 0.1, LAM, 2, H)
                    grid = np.arange(0.3, 1.0001, 0.005)
                    obj = [np.sum((fc.caputo_L1(Yg[:, k], g, H) - Th @ xi) ** 2) for g in grid]
                    new = float(grid[int(np.argmin(obj))])
                    if abs(new - al) < 1e-3: break
                    al = new
                row.append(round(al, 3))
            R['T2b'].append(dict(alpha=a, noise=nz, start=start, alpha_hat=row))
print('T2b', R['T2b'], flush=True); json.dump(R, open(out, 'w'), indent=1)
# T4 counts
for a in (0.7, 0.85, 1.0):
    for nz in (0.0, 0.05):
        e = [ls_order(data(a, nz, s, 'counts')[0], 1, [1, 2]) for s in (range(20) if nz > 0 else range(1))]
        Yg, _ = data(a, nz, 0, 'counts'); res = [bic_select(Yg, k) for k in (0, 1)]
        R['T4'].append(dict(alpha=a, noise=nz, G_known_mean=float(np.mean(e)), G_known_sd=float(np.std(e)),
                            full_alpha=[r[0] for r in res], full_terms=[r[1] for r in res]))
print('T4', R['T4'], flush=True); json.dump(R, open(out, 'w'), indent=1)
# T3 our validation selection (expensive)
for a in (0.7, 1.0):
    for nz in (0.0, 0.05):
        r = st.run('states', a, nz, 'val'); R['T3'].append(r)
        print('T3', r['alpha_true'], r['noise'], r['alpha_hat'], r['terms'], r['fc_rel_err'], r['secs'], flush=True)
        json.dump(R, open(out, 'w'), indent=1)
print('done')
