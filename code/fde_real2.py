"""Experiment A, revised after the synthetic tests (He Li's points 1-3).

Changes from fde_discovery.py:
  * smoothing spline (GCV) instead of an interpolating spline - the synthetic test showed that the
    interpolating spline biases the order downwards when the data are noisy (point 1);
  * order and terms chosen by the FIT criterion of Yu et al. (residual of the learned equation), made
    sparse with a BIC penalty, jointly over alpha in 0.40..1.00, the STRidge threshold and the library
    degree - no forecast-based selection and no clipping (point 2). Learned equations are saved;
  * a MECHANISTIC equation in the observed variables (point 3). In an SIR-type model with a common
    Caputo order, new cases C = beta S I and the outflow y = k I (deaths mu I in SIDR, admissions p gamma I
    in SIHDR) satisfy exactly
          D^alpha y = k beta S I - k (gamma + ...) I   =>   D^alpha y = a C - g y ,
    with g = rate of leaving the infectious class (about 1/infectious period). We fit a, g and alpha for
    y = deaths and y = admissions and record the profile of the residual over alpha.
Output JSON: compare2 format (preds) + 'eqs' + 'mech'.
"""
import argparse, json, time, itertools, numpy as np, scipy.io as sio, warnings
from scipy.interpolate import make_smoothing_spline
import fde_core as fc
warnings.filterwarnings('ignore')
p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
p.add_argument('--weeks', required=True); p.add_argument('--start', type=int, default=20)
a = p.parse_args()
d = sio.loadmat(a.data)
OBS = np.vstack([np.nan_to_num(d[k].ravel().astype(float)).clip(0) for k in ('cases', 'deaths', 'hospitalized')]).T
H, LAM, TOLS, AGRID = 0.25, 1e-3, (0.05, 0.1, 0.2, 0.4, 0.7), np.round(np.arange(0.40, 1.0001, 0.025), 3)
NAMES = ['C', 'D', 'H']


def weeks(w):
    if ':' in w: lo, hi = map(int, w.split(':')); return list(range(lo, hi + 1))
    return [int(x) for x in w.split(',')]


def bic_eq(Yg, k):
    n = len(Yg) - 1; best = None
    for deg in (1, 2):
        _, nm = fc.library(Yg[:1], deg, NAMES)
        for al in AGRID:
            for tol in TOLS:
                xi, r = fc.fit_eq(Yg, k, al, tol, LAM, deg, H); kk = int((xi != 0).sum())
                b = n * np.log(r + 1e-12) + kk * np.log(n)
                if best is None or b < best[0]:
                    best = (b, float(al), tol, deg, xi, r, {nm[i]: round(float(x), 4) for i, x in enumerate(xi) if x != 0})
    return best


def mech(Yg, k):
    """D^alpha y = a C - g y  (scaled variables; g is scale-free, per week). Profile over alpha."""
    prof = []
    for al in np.round(np.arange(0.30, 1.0001, 0.01), 2):
        lhs = fc.caputo_L1(Yg[:, k], al, H); X = np.vstack([Yg[1:, 0], -Yg[1:, k]]).T
        c = np.linalg.lstsq(X, lhs, rcond=None)[0]; r = float(np.mean((lhs - X @ c) ** 2) / np.mean(lhs ** 2))
        prof.append((float(al), r, float(c[0]), float(c[1])))
    i = int(np.argmin([q[1] for q in prof])); rmin = prof[i][1]
    flat = [q[0] for q in prof if q[1] <= 1.1 * rmin]
    return dict(alpha=prof[i][0], a=prof[i][2], g=prof[i][3], rel_resid=rmin, alpha_range_10pct=[min(flat), max(flat)],
                profile=[[q[0], round(q[1], 6)] for q in prof])


out = {'args': vars(a), 'windows': []}
for tr_end in weeks(a.weeks):
    t1 = time.time()
    tw = np.arange(a.start, tr_end); yw = OBS[a.start:tr_end]; sc = yw.max(0) + 1e-9
    tg = np.arange(a.start, tr_end - 1 + 1e-9, H)
    Yg = np.vstack([make_smoothing_spline(tw, yw[:, k] / sc[k])(tg) for k in range(3)]).T.clip(0)
    res = {}
    for variant in ('learn', 'a1'):
        eqs = []
        for k in range(3):
            if variant == 'learn': b = bic_eq(Yg, k)
            else:   # alpha = 1, same BIC selection of threshold and degree
                n = len(Yg) - 1; b = None
                for deg in (1, 2):
                    _, nm = fc.library(Yg[:1], deg, NAMES)
                    for tol in TOLS:
                        xi, r = fc.fit_eq(Yg, k, 1.0, tol, LAM, deg, H); kk = int((xi != 0).sum()); bb = n * np.log(r + 1e-12) + kk * np.log(n)
                        if b is None or bb < b[0]: b = (bb, 1.0, tol, deg, xi, r, {nm[i]: round(float(x), 4) for i, x in enumerate(xi) if x != 0})
            eqs.append(b)
        # common degree needed for forecasting: use the largest selected degree, re-express xi
        deg = max(e[3] for e in eqs); xis = []
        for e in eqs:
            if e[3] == deg: xis.append(e[4])
            else: x = np.zeros(fc.library(Yg[:1], deg)[0].shape[1]); x[:len(e[4])] = e[4]; xis.append(x)
        Yf = fc.forecast(Yg, [e[1] for e in eqs], xis, 16, deg, H, clip=None)
        P = Yf[3::4] * sc
        blow = bool((Yf > 3).any() or (Yf < 0).any())
        res[variant] = dict(preds=[[float(max(P[h, j], 0)) for j in range(3)] for h in range(4)], blowup=blow,
                            eqs=[dict(var=NAMES[k], alpha=e[1], tol=e[2], deg=e[3], rel_resid=round(e[5], 5), terms=e[6]) for k, e in enumerate(eqs)])
    m = {NAMES[k]: mech(Yg, k) for k in (1, 2)}
    out['windows'].append({'train_end': tr_end, 'preds': res['learn']['preds'], 'z': [[e['alpha'] for e in res['learn']['eqs']]],
                           'learn': res['learn'], 'a1': res['a1'], 'mech': m, 'secs': round(time.time() - t1, 1)})
    print(tr_end, 'alpha', [e['alpha'] for e in res['learn']['eqs']], 'blow', res['learn']['blowup'], res['a1']['blowup'],
          'mech D', m['D']['alpha'], m['D']['alpha_range_10pct'], round(m['D']['g'], 2), 'H', m['H']['alpha'], m['H']['alpha_range_10pct'], round(m['H']['g'], 2),
          round(time.time() - t1), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
