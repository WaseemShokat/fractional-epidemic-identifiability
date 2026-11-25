"""Compare runs: MAE, RMSE, peak timing error (signed, weeks), peak magnitude error (%), per horizon.
Usage: python3 compare2.py label=file.json [label=file.json ...] [--csv out.csv]"""
import json, sys, csv, numpy as np, scipy.io as sio
R='repo/unz'; d=sio.loadmat(f'{R}/data/data_weekly_avg.mat'); r=sio.loadmat(f'{R}/result/res_V14_1.mat')
names=['cases','deaths','hospitalized']; sc=[120000,500,2000]
obs={k:np.clip(np.nan_to_num(d[k].ravel().astype(float)),0,None) for k in names}
args=sys.argv[1:]; out=None
if '--csv' in args: i=args.index('--csv'); out=args[i+1]; args=args[:i]+args[i+2:]
runs={}
for x in args:
    lab,f=x.split('=',1); runs[lab]=json.load(open(f))
rows=[]
for vi,k in enumerate(names):
    for h in range(4):
        series={}
        for lab,o in runs.items():
            series[lab]={w['train_end']+h:w['preds'][h][vi] for w in o['windows']}
        weeks=sorted(set.intersection(*[set(v) for v in series.values()]))
        W=np.array(weeks); y=obs[k][W]; tpk=W[np.argmax(y)]; ypk=y.max()
        allp={'Authors (saved)':r['all_'+k+'_pred'][h,W]*sc[vi], 'Naive':obs[k][W-(h+1)]}
        allp.update({lab:np.array([v[w] for w in weeks]) for lab,v in series.items()})
        for lab,p in allp.items():
            if np.isnan(p).any(): continue
            rows.append(dict(variable=k,horizon=h+1,model=lab,weeks=f'{weeks[0]}-{weeks[-1]}',
                MAE=round(float(np.mean(abs(p-y)))),RMSE=round(float(np.sqrt(np.mean((p-y)**2)))),
                peak_week_obs=int(tpk),peak_week_pred=int(W[np.argmax(p)]),peak_timing_err=int(W[np.argmax(p)]-tpk),
                peak_mag_err_pct=round(100*(p.max()-ypk)/ypk,1)))
for k in names:
    print(f'\n===== {k}')
    print(f"{'h':>2} {'model':<22}{'MAE':>8}{'RMSE':>8}{'pk obs':>7}{'pk pred':>8}{'timing':>7}{'mag%':>7}")
    for rw in rows:
        if rw['variable']==k:
            print(f"{rw['horizon']:>2} {rw['model']:<22}{rw['MAE']:>8}{rw['RMSE']:>8}{rw['peak_week_obs']:>7}{rw['peak_week_pred']:>8}{rw['peak_timing_err']:>+7}{rw['peak_mag_err_pct']:>7}")
if out:
    with open(out,'w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
