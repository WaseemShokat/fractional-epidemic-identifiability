import json, numpy as np, os, sys
sys.argv=['x']; exec(open('code/seedstats.py').read().split("WK={")[0])  # rel, ref, OB
WK={'w':list(range(36,46)),'omi':[92,94,96,98,100]}
base={'F3 SIHDR (fractional, paper)':('F3_paper',{'w':'code/kh_F3_paper_w.json','omi':'code/kh_F3_paper_omi.json'}),
      'F3 SIHDR, $\\kappa=1$':('F3_int',{'w':'code/kh_F3_int_w.json','omi':'code/kh_F3_int_omi_all.json'}),
      'F2 SIDR (fractional, paper)':('F2_paper',{'w':'code/kh_F2_paper_w.json','omi':'code/kh_F2_paper_omi.json'})}
L=[r'\begin{tabular}{l ccc ccc}\toprule',r'& \multicolumn{3}{c}{Winter (weeks 36--45)} & \multicolumn{3}{c}{Omicron (5 windows)}\\',r'Model & cases & deaths & admissions & cases & deaths & admissions\\\midrule']
row='Qian et al.\\ (saved)'
for wv in ('w','omi'):
    for v in range(3): row+=f' & {ref("Qian",WK[wv],v):.2f}'
L.append(row+r'\\')
row='Integer PINN (report 1)'
for wv,f in (('w','code/int_direct_36_45.json'),('omi','code/int_direct_omi_all.json')):
    for v in range(3): row+=f' & {rel([f],WK[wv],v)[0]:.2f}'
L.append(row+r'\\')
for lab,(m,b) in base.items():
    row=lab
    for wv in ('w','omi'):
        for v in range(3):
            x=rel([b[wv]]+[f'code/rob/{m}_s{s}_{wv}.json' for s in (1,2)],WK[wv],v)
            row+=' & --' if not x else f' & {np.mean(x):.2f} [{min(x):.2f}--{max(x):.2f}]'
    L.append(row+r'\\')
L.append(r'\midrule')
for lab,f in (('F3 paper, learned rates','code/rob/F3_paper_rates_{}.json'),('F3 paper, daily grid ($h=1$ day)','code/rob/F3_paper_h1_{}.json')):
    row=lab
    for wv in ('w','omi'):
        for v in range(3):
            p=f.format(wv)
            try: x=rel([p],WK[wv],v); row+=f' & {x[0]:.2f}'
            except Exception: row+=' & --'
    L.append(row+r'\\')
L.append(r'\bottomrule\end{tabular}')
open('figures/tseeds.tex','w').write('\n'.join(L)); print('\n'.join(L))
