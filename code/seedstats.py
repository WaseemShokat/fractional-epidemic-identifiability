import json, numpy as np, scipy.io as sio, os
d=sio.loadmat('repo/unz/data/data_weekly_avg.mat')
OB=np.vstack([np.clip(np.nan_to_num(d[k].ravel().astype(float)),0,None) for k in ('cases','deaths','hospitalized')]).T
r=sio.loadmat('repo/unz/result/res_V14_1.mat'); sc=[120000,500,2000]
def rel(files, weeks, v):
    out=[]
    for f in files:
        W={w['train_end']:w for w in json.load(open(f))['windows']}
        e=[];n=[]
        for t in weeks:
            for h in range(4):
                p=W[t]['preds'][h][v]
                if p is None or (isinstance(p,float) and np.isnan(p)): return None
                e.append(abs(p-OB[t+h,v])); n.append(abs(OB[t-1,v]-OB[t+h,v]))
        out.append(sum(e)/sum(n))
    return out
def ref(name,weeks,v):
    if name=='Qian': p=[r['all_'+k+'_pred'] for k in ('cases','deaths','hospitalized')][v]
    e=[];n=[]
    for t in weeks:
        for h in range(4):
            e.append(abs(p[h,t+h]*sc[v]-OB[t+h,v])); n.append(abs(OB[t-1,v]-OB[t+h,v]))
    return sum(e)/sum(n)
WK={'w':list(range(36,46)),'omi':[92,94,96,98,100]}
base={'F3_paper':{'w':'code/kh_F3_paper_w.json','omi':'code/kh_F3_paper_omi.json'},
      'F3_int':{'w':'code/kh_F3_int_w.json','omi':'code/kh_F3_int_omi_all.json'},
      'F2_paper':{'w':'code/kh_F2_paper_w.json','omi':'code/kh_F2_paper_omi.json'}}
res={}
for wv in ('w','omi'):
    print('== wave',wv,'weeks',WK[wv])
    for v,vn in enumerate(('cases','deaths','admissions')):
        line=f'{vn:<11} Qian {ref("Qian",WK[wv],v):.2f} |'
        for m,b in base.items():
            fs=[b[wv]]+[f'code/rob/{m}_s{s}_{wv}.json' for s in (1,2)]
            x=rel(fs,WK[wv],v)
            if x: line+=f' {m} {np.mean(x):.2f} [{min(x):.2f}-{max(x):.2f}] |'; res[f'{m}_{wv}_{vn}']=x
        for extra,f in (('rates',f'code/rob/F3_paper_rates_{wv}.json'),('h1',f'code/rob/F3_paper_h1_{wv}.json')):
            if os.path.exists(f):
                try:
                    x=rel([f],WK[wv],v); line+=f' F3_{extra} {x[0]:.2f} |' if x else ''
                except KeyError: pass
        print(line)
json.dump(res,open('code/seedstats.json','w'))
