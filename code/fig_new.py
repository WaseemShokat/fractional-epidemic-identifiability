import csv,math,collections,json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, scipy.io as sio
order=['Authors (saved)','Integer','Learned','FDE-a1','FDE-learn','F1-paper','F2-paper','F2-beta','F3-int','F3-paper','F3-beta']
lab={'Authors (saved)':'Qian et al. (saved)','Integer':'Integer PINN','Learned':'Fractional PINN','FDE-a1':'A: learned ODE','FDE-learn':'A: learned FDE',
 'F1-paper':'B: F1 SIR','F2-paper':'B: F2 SIDR','F2-beta':'B: F2 + NN β','F3-int':'B: F3 integer','F3-paper':'B: F3 SIHDR','F3-beta':'B: F3 + NN β'}
fig,ax=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
for a,(f,t) in zip(ax,[('code/cmp_winter.csv','Winter wave (train end 36–45)'),('code/cmp_omi.csv','Omicron wave (train end 90–101)')]):
    T=collections.defaultdict(dict)
    for r in csv.DictReader(open(f)):
        m=float(r['MAE']);
        if not math.isnan(m): T[r['variable']].setdefault(r['model'],[]).append(m)
    for i,v in enumerate(['cases','deaths','hospitalized']):
        nb=sum(T[v]['Naive'])
        for j,m in enumerate(order):
            if m in T[v]:
                x=sum(T[v][m])/nb; a.barh(j+0.27*(i-1),min(x,3),0.27,color=['#3b6ea8','#c0504d','#6a9a3a'][i],label=v if j==0 else None)
                if x>3: a.text(3.02,j+0.27*(i-1),f'{x:.1f}',va='center',fontsize=7)
    a.axvline(1,color='k',lw=1,ls='--'); a.set_title(t); a.set_xlabel('MAE relative to naive forecast (mean over 1–4 weeks)'); a.set_xlim(0,3.3)
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([lab[m] for m in order]); ax[0].invert_yaxis(); ax[0].legend(loc='lower right')
plt.tight_layout(); plt.savefig('figures/fig_relmae.png',dpi=180)
# time series, 2 weeks ahead cases
d=sio.loadmat('repo/unz/data/data_weekly_avg.mat'); y=np.nan_to_num(d['cases'].ravel().astype(float))
fig,ax=plt.subplots(1,2,figsize=(12,4))
for a,(fs,t) in zip(ax,[({'Integer PINN':'int_direct_36_45','A: learned FDE':'fde_learn_36_45','B: F3 SIHDR':'kh_F3_paper_w','B: F2 SIDR':'kh_F2_paper_w'},'Winter wave'),
                        ({'Integer PINN':'int_direct_omi_all','A: learned FDE':'fde_learn_omi_all','B: F3 SIHDR':'kh_F3_paper_omi','B: F2 SIDR':'kh_F2_paper_omi'},'Omicron wave')]):
    W=None
    for l,f in fs.items():
        o=json.load(open(f'code/{f}.json'))['windows']; W=[w['train_end']+1 for w in o]
        a.plot(W,[w['preds'][1][0]/1e3 for w in o],'o-',ms=3,label=l)
    ww=np.arange(min(W)-3,max(W)+1); a.plot(ww,y[ww]/1e3,'k-',lw=2,label='Observed')
    a.set_title(t+': 2-week-ahead case forecasts'); a.set_xlabel('week'); a.set_ylabel('daily cases (thousand)')
    if t=='Omicron wave': a.set_ylim(0,max(y[ww])/1e3*1.6)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig('figures/fig_cases_h2.png',dpi=180)
