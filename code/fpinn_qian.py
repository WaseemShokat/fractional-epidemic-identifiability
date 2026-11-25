"""Fractional extension of the Qian et al. (2025) epidemic PINN.
Same 9-compartment O'Dea-Drake model, same networks, data, scaling, losses and
rolling window as the authors' public code (models_V5.py, continual_pred_with_past_data_V16).
--frac: replaces d/dt by a Caputo derivative of learned order z (L1 scheme on the ODE grid).
--recursive: 1-week-ahead forecasts chained: each predicted week is appended as a
             pseudo-observation and the model is updated before the next week (He Li's idea).
"""
import argparse, math, time, json
import numpy as np, scipy.io as sio, torch, torch.nn as nn
from torch.autograd import grad

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--weeks', type=str, required=True, help='train_end values, e.g. 36:45 or 40,41')
p.add_argument('--iter', type=int, default=50000)
p.add_argument('--iter_rec', type=int, default=10000, help='extra iterations per recursive update (warm start)')
p.add_argument('--frac', action='store_true')
p.add_argument('--recursive', action='store_true')
p.add_argument('--l2', type=float, default=1e-5)
p.add_argument('--wode', type=float, default=0.1)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--z0', type=float, default=0.9)
p.add_argument('--zfix', action='store_true')
p.add_argument('--last', action='store_true', help='use final iterate instead of best checkpoint (as original code)')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(2)

d = sio.loadmat(a.data)
WEEKMAX, TRAIN_START = 133, 20
s = [120000, 500, 2000, 20, 1e8]
def prep(x, sc):
    x = x.T.astype(float).copy(); x[np.isnan(x)] = 0; x[x <= 0] = 0; return x / sc
Zr, Dr, A, res, doses = [prep(d[k], sc) for k, sc in zip(['cases','deaths','hospitalized','res','doses'], s)]
t_all = torch.tensor(d['time'].T, dtype=torch.float32)
N, eta, gamma, gamma_dw, gamma_zw, gamma_h, rho = 1.0, .25, .25, .1, 1.0, .1, .5

class Net(nn.Module):
    def __init__(s_, n_out, out_tf=None):
        super().__init__()
        s_.l = nn.ModuleList([nn.Linear(1,50), nn.Linear(50,50), nn.Linear(50,50), nn.Linear(50,n_out)])
        s_.out_tf = out_tf
    def forward(s_, t):
        h = t / WEEKMAX
        for L in s_.l[:-1]: h = torch.tanh(L(h))
        h = s_.l[-1](h)
        return s_.out_tf(h) if s_.out_tf else h

def rhs(us, p_h, p_d, beta):
    X, L, Y, Z, Zr_, H, A_, D, Dr_ = us
    return torch.cat([-beta*X*Y/N, beta*X*Y/N - eta*L, eta*L - gamma*Y,
        (p_h + rho*(1-p_h))*gamma*Y - gamma_zw*Z, gamma_zw*Z/s[0],
        p_h*gamma*Y - gamma_h*H, p_h*gamma*Y/s[2], p_d*gamma_h*H - gamma_dw*D, gamma_dw*D/s[1]], -1)

def caputo_L1(u, z, h):
    """u: (n,9) values on a uniform grid. Returns L1 Caputo derivative at points 1..n-1."""
    n = u.shape[0]
    k = torch.arange(n, dtype=torch.float32)
    b = (k+1)**(1-z) - k**(1-z)                      # b_0..b_{n-1}
    du = u[1:] - u[:-1]                               # (n-1,9): du_j = u_{j+1}-u_j
    i = torch.arange(1, n).unsqueeze(1); j = torch.arange(0, n-1).unsqueeze(0)
    lag = (i - 1 - j).clamp(min=0)                    # for D at point i: sum_j b_{i-1-j} du_j, j<=i-1
    W = torch.where(j <= i-1, b[lag], torch.zeros(()))
    return (W @ du) / (h**z * math.gamma(2 - z.item()) if False else h**z * torch.exp(torch.lgamma(2 - z)))

def train(tr_end, Zr_tr, Dr_tr, A_tr, test_end, state=None, iters=50000):
    t_tr = t_all[TRAIN_START:tr_end]
    t_ode = torch.arange(TRAIN_START, test_end + 1e-6, 0.25, dtype=torch.float32).reshape(-1,1).requires_grad_(True)
    hstep = 0.25
    u = Net(9, torch.exp); u1 = Net(3)
    L0 = torch.tensor([[1.0]], requires_grad=True)
    lph = torch.tensor(0.5, requires_grad=True); lpd = torch.tensor(0.5, requires_grad=True)
    zr = torch.tensor(math.log((a.z0-0.5)/(1-a.z0)), requires_grad=True)   # z in (0.5,1)
    if state:
        u.load_state_dict(state['u']); u1.load_state_dict(state['u1'])
        with torch.no_grad(): L0.copy_(state['L0']); lph.copy_(state['lph']); lpd.copy_(state['lpd']); zr.copy_(state['zr'])
    params = list(u.parameters()) + list(u1.parameters()) + [L0, lph, lpd] + ([zr] if (a.frac and not a.zfix) else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=a.l2)
    t0 = torch.zeros(1,1); Zr_t, Dr_t, A_t = [torch.tensor(x, dtype=torch.float32) for x in (Zr_tr, Dr_tr, A_tr)]
    res_t = torch.tensor(res[TRAIN_START:tr_end], dtype=torch.float32); dos_t = torch.tensor(doses[TRAIN_START:tr_end], dtype=torch.float32)
    best, best_state = 1e18, None
    for it in range(iters):
        u1o = u1(t_tr); l_res = ((torch.exp(u1o[:,0:1]) - res_t)**2).mean(); l_dos = ((torch.exp(u1o[:,1:2]) - dos_t)**2).mean()
        p_h, p_d = torch.sigmoid(lph), torch.sigmoid(lpd)
        Y0 = eta*L0/gamma; Z0 = (p_h + rho*(1-p_h))*gamma*Y0/gamma_zw; H0 = p_h*gamma*Y0/gamma_h; D0 = p_d*gamma_h*H0/gamma_dw
        X0 = torch.clip(N - L0 - Y0 - H0 - D0, min=N/10)
        u0 = torch.cat([X0, L0, Y0, Z0, torch.zeros(1,1), H0, torch.zeros(1,1), D0, torch.zeros(1,1)], -1)
        U = u(t_ode); us = torch.split(U, 1, -1); beta = u1(t_ode)[:, 2:3]
        R = rhs(us, p_h, p_d, beta)
        if a.frac:
            z = 0.5 + 0.5*torch.sigmoid(zr)
            lhs = caputo_L1(U, z, hstep); R = R[1:]
        else:
            lhs = torch.cat([grad(ui, t_ode, torch.ones_like(ui), create_graph=True)[0] for ui in us], -1)
        e = (lhs - R)**2
        l_ode = e.mean(0).sum()
        l_init = ((u(t0) - u0)**2).mean()
        Up = u(t_tr)
        l_data = ((Up[:,4:5]-Zr_t)**2).mean() + ((Up[:,8:9]-Dr_t)**2).mean() + ((Up[:,6:7]-A_t)**2).mean()
        loss = a.wode*l_ode + l_init + l_data + l_res + l_dos
        opt.zero_grad(); loss.backward(); opt.step()
        if (it % 1000 == 0 and loss.item() < best) or (a.last and it == iters - 1):
            best = loss.item(); best_state = {'u': {k: v.clone() for k, v in u.state_dict().items()}, 'u1': {k: v.clone() for k, v in u1.state_dict().items()},
                         'L0': L0.detach().clone(), 'lph': lph.detach().clone(), 'lpd': lpd.detach().clone(), 'zr': zr.detach().clone()}
    u.load_state_dict(best_state['u'])
    zval = float(0.5 + 0.5*torch.sigmoid(best_state['zr'])) if a.frac else 1.0
    return u, best_state, zval

def parse_weeks(w):
    if ':' in w: lo, hi = map(int, w.split(':')); return list(range(lo, hi+1))
    return [int(x) for x in w.split(',')]

out = {'args': vars(a), 'windows': []}
for tr_end in parse_weeks(a.weeks):
    t1 = time.time(); test_end = tr_end + 4
    Zr_tr, Dr_tr, A_tr = Zr[TRAIN_START:tr_end].copy(), Dr[TRAIN_START:tr_end].copy(), A[TRAIN_START:tr_end].copy()
    if not a.recursive:
        u, st, z = train(tr_end, Zr_tr, Dr_tr, A_tr, test_end, iters=a.iter)
        with torch.no_grad(): P = u(t_all[tr_end:test_end])
        preds = [[P[h,4].item()*s[0], P[h,8].item()*s[1], P[h,6].item()*s[2]] for h in range(4)]; zs = [z]
    else:
        preds, zs, st = [], [], None
        for h in range(4):
            cur_end = tr_end + h
            u, st, z = train(cur_end, Zr_tr, Dr_tr, A_tr, cur_end + 1, state=st, iters=a.iter if h == 0 else a.iter_rec)
            with torch.no_grad(): P = u(t_all[cur_end:cur_end+1])
            preds.append([P[0,4].item()*s[0], P[0,8].item()*s[1], P[0,6].item()*s[2]]); zs.append(z)
            Zr_tr = np.vstack([Zr_tr, P[:,4:5].numpy()]); Dr_tr = np.vstack([Dr_tr, P[:,8:9].numpy()]); A_tr = np.vstack([A_tr, P[:,6:7].numpy()])
    out['windows'].append({'train_end': tr_end, 'preds': preds, 'z': zs, 'secs': round(time.time()-t1, 1)})
    print(tr_end, 'z=', [round(x,3) for x in zs], 'secs', round(time.time()-t1,1), flush=True)
    json.dump(out, open(a.out, 'w'))
