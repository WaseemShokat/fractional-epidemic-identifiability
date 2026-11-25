"""Experiment B (He Li, 21 Nov): established fractional models of Kharazmi et al. (Nat. Comput. Sci. 2021,
Fig. 1: F1 SIR, F2 SIDR, F3 SIHDR) fitted to the California data with the same rolling windows as before.

Time is in days (week w <-> t = 7w); states are population fractions; Caputo derivatives of variable
order kappa_i(t) per compartment are approximated by the L1 scheme on a uniform grid (default 3.5 days, since the data are weekly; the authors' code, PINN-COVID FDM1,
uses daily data). Observations (weekly averages of daily counts) are compared with daily flows:
  cases      = beta S I          (all models)
  deaths     = mu I (F2) or q phiD H (F3)
  admissions = p gamma I (F3 only)
Variants (--variant):
  paper : as in the authors' code – nominal constant beta, kappa_i(t) time-dependent (cubic polynomial
          in t per compartment), constant p, q (F3) / mu (F2) learned; kappa frozen after training end
  beta  : beta(t) by a neural network (extrapolated into the forecast, as in Qian et al.), constant
          kappa_i learned per compartment
  int   : like 'beta' but kappa_i = 1 (integer-order counterpart)
Output JSON matches fpinn_qian.py: preds[h] = [cases, deaths, admissions] for weeks tr_end..tr_end+3
(variables a model does not have are set to NaN).
"""
import argparse, json, math, time
import numpy as np, scipy.io as sio, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--out', required=True); p.add_argument('--weeks', required=True)
p.add_argument('--model', choices=['F1', 'F2', 'F3'], default='F3')
p.add_argument('--variant', choices=['paper', 'beta', 'int'], default='paper')
p.add_argument('--iter', type=int, default=30000); p.add_argument('--seed', type=int, default=0)
p.add_argument('--learnrates', action='store_true'); p.add_argument('--kfix', type=float, default=None)
p.add_argument('--wode', type=float, default=1.0); p.add_argument('--h', type=float, default=3.5); p.add_argument('--start', type=int, default=20)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

NPOP = 39.5e6
d = sio.loadmat(a.data)
OBS = np.vstack([np.nan_to_num(d[k].ravel().astype(float)).clip(0) for k in ('cases', 'deaths', 'hospitalized')]).T / NPOP
GAMMA, PHID, PHIR, BETA0 = 1 / 6, 1 / 15, 1 / 7.5, 0.25          # authors' nominal values (per day)
NC = {'F1': 3, 'F2': 4, 'F3': 5}[a.model]                         # S I R / S I D R / S I H D R

class MLP(nn.Module):
    def __init__(s, nin, nout, width=40, depth=3):
        super().__init__(); L = [nn.Linear(nin, width), nn.Tanh()]
        for _ in range(depth - 1): L += [nn.Linear(width, width), nn.Tanh()]
        s.f = nn.Sequential(*L, nn.Linear(width, nout))
    def forward(s, x): return s.f(x)

def var_order_L1(U, K, h):
    """U: (n, c) states on uniform grid, K: (n, c) orders at each grid point. Caputo derivative of
    variable order (order taken at the evaluation point) at points 1..n-1, L1 scheme."""
    n = U.shape[0]; dU = U[1:] - U[:-1]                                   # (n-1, c)
    i = torch.arange(1, n).view(-1, 1, 1).double(); j = torch.arange(0, n - 1).view(1, -1, 1).double()
    k = K[1:].unsqueeze(1)                                                # (n-1, 1, c)
    m = (i - 1 - j).clamp(min=0)
    b = (m + 1) ** (1 - k) - m ** (1 - k)
    b = torch.where(m == 0, torch.ones_like(b), b)
    W = torch.where(j <= i - 1, b, torch.zeros_like(b))                   # (n-1, n-1, c)
    out = torch.einsum('ijc,jc->ic', W, dU)
    return out / (h ** K[1:] * torch.exp(torch.lgamma(2 - K[1:])))

def run_window(tr_end):
    t0d, tend = 7 * a.start, 7 * (tr_end + 4)
    tg = torch.arange(t0d, tend + 1e-9, a.h).view(-1, 1)
    tn = (tg - t0d) / (tend - t0d)
    obs_w = np.arange(a.start, tr_end); obs_idx = torch.tensor(((7 * obs_w - t0d) / a.h).round().astype(int))
    Y = torch.tensor(OBS[a.start:tr_end]); ysc = Y.max(0).values + 1e-12
    t_tr_end = 7 * (tr_end - 1)
    state = MLP(1, NC)                                                   # softmax -> fractions summing to 1
    bnet = MLP(1, 1, 20, 2) if a.variant in ('beta', 'int') else None
    kcoef = torch.zeros(4, NC, requires_grad=True) if a.variant == 'paper' else None
    kconst = torch.zeros(NC, requires_grad=True) if (a.variant == 'beta' and a.kfix is None) else None
    with torch.no_grad():
        if kcoef is not None: kcoef[0] += 2.0                            # kappa ~ sigmoid(2) = 0.88 at start
        if kconst is not None: kconst += 2.0
    lp = torch.tensor([-2.0, -1.0], requires_grad=True)                  # p, q (F3) or mu (F2)
    lr_ = torch.zeros(3, requires_grad=True)                             # log-multipliers of gamma, phiD, phiR (--learnrates)
    params = list(state.parameters()) + [lp] + ([lr_] if a.learnrates else []) + ([*bnet.parameters()] if bnet else []) + \
             ([kcoef] if kcoef is not None else []) + ([kconst] if kconst is not None else [])
    opt = torch.optim.Adam(params, lr=2e-3)
    def fields():
        U = torch.softmax(state(tn) - 4 * torch.tensor([0.] + [1.] * (NC - 1)), -1)   # bias towards S at start
        if a.variant == 'paper':
            tt = torch.clamp(tn, max=(t_tr_end - t0d) / (tend - t0d))              # freeze kappa after training
            Pm = torch.cat([tt ** e for e in range(4)], -1)
            K = torch.sigmoid(Pm @ kcoef).clamp(0.05, 1.0)
            beta = torch.full_like(tn, BETA0)
        else:
            if a.kfix is not None: K = torch.full((len(tn), NC), a.kfix)
            else: K = torch.sigmoid(kconst).expand(len(tn), NC).clamp(0.05, 1.0) if a.variant == 'beta' else torch.ones(len(tn), NC)
            beta = 0.6 * torch.sigmoid(bnet(tn))
        return U, K, beta
    def rhs_and_obs(U, beta):
        m = torch.exp(0.7 * torch.tanh(lr_)) if a.learnrates else torch.ones(3)   # each rate within x0.5..x2 of nominal
        GAMMA_, PHID_, PHIR_ = GAMMA * m[0], PHID * m[1], PHIR * m[2]
        S, I = U[:, 0:1], U[:, 1:2]; inf = beta * S * I
        if a.model == 'F1':
            R = torch.cat([-inf, inf - GAMMA_ * I, GAMMA_ * I], -1); obs = [inf, None, None]
        elif a.model == 'F2':
            mu = 0.05 * torch.sigmoid(lp[0])
            R = torch.cat([-inf, inf - (GAMMA_ + mu) * I, mu * I, GAMMA_ * I], -1); obs = [inf, mu * I, None]
        else:
            pp, qq = torch.sigmoid(lp[0]), torch.sigmoid(lp[1]); Hh = U[:, 2:3]
            R = torch.cat([-inf, inf - GAMMA_ * I, pp * GAMMA_ * I - (qq * PHID_ + (1 - qq) * PHIR_) * Hh,
                           qq * PHID_ * Hh, (1 - pp) * GAMMA_ * I + (1 - qq) * PHIR_ * Hh], -1)
            obs = [inf, qq * PHID_ * Hh, pp * GAMMA_ * I]
        return R, obs
    best, best_sd = 1e18, None
    for it in range(a.iter):
        U, K, beta = fields()
        R, obs = rhs_and_obs(U, beta)
        lhs = var_order_L1(U, K, a.h)
        l_ode = (((lhs - R[1:]) / ysc[0]) ** 2).mean()
        l_data = sum((((o[obs_idx, 0] - Y[:, v]) / ysc[v]) ** 2).mean() for v, o in enumerate(obs) if o is not None)
        loss = l_data + a.wode * l_ode
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 999 and loss.item() < best:
            best = loss.item(); best_parts = (l_data.item(), l_ode.item())
            best_sd = [x.detach().clone() for x in params]
    with torch.no_grad():
        for x, y in zip(params, best_sd): x.copy_(y)
        U, K, beta = fields(); _, obs = rhs_and_obs(U, beta)
        idx = [int(round((7 * w - t0d) / a.h)) for w in range(tr_end, tr_end + 4)]
        preds = [[float(o[i, 0] * NPOP) if o is not None else float('nan') for o in obs] for i in idx]
        kap = K[int(round((7 * (tr_end - 1) - t0d) / a.h))].tolist()
        rates = (torch.exp(0.7 * torch.tanh(lr_)) * torch.tensor([GAMMA, PHID, PHIR])).tolist()
    return preds, kap, best, best_parts, rates

def weeks(w):
    if ':' in w: lo, hi = map(int, w.split(':')); return list(range(lo, hi + 1))
    return [int(x) for x in w.split(',')]

out = {'args': vars(a), 'windows': []}
for tr_end in weeks(a.weeks):
    t1 = time.time(); preds, kap, loss, parts, rates = run_window(tr_end)
    out['windows'].append({'train_end': tr_end, 'preds': preds, 'z': [kap], 'loss': loss, 'l_data': parts[0], 'l_ode': parts[1], 'rates': rates, 'secs': round(time.time() - t1, 1)})
    print(tr_end, 'kappa', [round(k, 3) for k in kap], 'cases', [round(p[0]) for p in preds], 'secs', round(time.time() - t1), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
