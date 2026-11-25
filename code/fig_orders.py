"""Point 5: learned fractional orders across training windows, for every model (with seeds where available)."""
import json, glob, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
waves = [('w', 'fde_learn_36_45', 'Winter wave'), ('omi', 'fde_learn_omi_all', 'Omicron wave')]
cols = {'C': '#3b6ea8', 'D': '#c0504d', 'H': '#6a9a3a'}
for r, (w, fA, title) in enumerate(waves):
    # (a) Experiment A, forecast-validation selection (report 2) and fit selection (revised)
    a = ax[r, 0]
    for f, mk, ls, lab in ((fA, 'o', '-', 'validation'), (f'fde2_{w}', 'x', ':', 'fit (Yu)')):
        d = J(f'code/{f}.json')
        if not d: continue
        W = [x['train_end'] for x in d['windows']]; Z = np.array([x['z'][0] for x in d['windows']])
        for k, v in enumerate('CDH'):
            a.plot(W, Z[:, k], marker=mk, ls=ls, color=cols[v], ms=4, label=f'{v}, {lab}')
    a.set_title(f'{title}\nA: learned equation (orders of C, D, H)', fontsize=10)
    # (b) mechanistic equation D^a y = aC - g y : best alpha and range within 10% of the minimum residual
    a = ax[r, 1]; d = J(f'code/fde2_{w}.json')
    if d:
        W = np.array([x['train_end'] for x in d['windows']])
        for j, v in enumerate('DH'):
            al = [x['mech'][v]['alpha'] for x in d['windows']]; lo = [x['mech'][v]['alpha_range_10pct'][0] for x in d['windows']]
            hi = [x['mech'][v]['alpha_range_10pct'][1] for x in d['windows']]
            a.errorbar(W + (j - 0.5) * 0.2, al, yerr=[np.array(al) - lo, np.array(hi) - al], fmt='o', color=cols[v], ms=4, capsize=2,
                       label=f'{"deaths" if v == "D" else "admissions"}')
    a.set_title(f'{title}\nA (mechanistic): D$^a$y = aC − gy, bars: residual within 10% of min', fontsize=10)
    # (c) Kharazmi models: order of the infectious compartment I, all seeds
    a = ax[r, 2]
    spec = [('F3_paper', '#7b3fa0', 'F3 SIHDR'), ('F3_beta', '#e08a1e', 'F3 + NN β'), ('F2_paper', '#2a9d8f', 'F2 SIDR'), ('F1_paper', '#888888', 'F1 SIR'), ('F3_int', None, None)][:4]
    for name, c, lab in spec:
        files = [f'code/kh_{name}_{w}.json', f'code/kh_{name}_{w}_all.json'] + sorted(glob.glob(f'code/rob/{name}_s*_{w}.json'))
        first = True
        for f in files:
            d = J(f)
            if not d: continue
            W = [x['train_end'] for x in d['windows']]; k = [x['z'][0][1] for x in d['windows']]
            a.plot(W, k, marker='o', ms=3, lw=0.8, color=c, alpha=0.9, label=lab if first else None); first = False
    a.set_title(f'{title}\nB: order κ of the infectious compartment I (each line = one seed)', fontsize=10)
for a in ax.ravel(): a.set_ylim(0, 1.05); a.grid(alpha=0.3); a.set_xlabel('training end week')
for a in ax[:, 0]: a.set_ylabel('fractional order')
ax[0, 0].legend(fontsize=7, ncol=2); ax[0, 1].legend(fontsize=8); ax[0, 2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/fig_orders.png', dpi=170)
