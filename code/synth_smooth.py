"""T1s/T2s: as T1/T2 of synth_test2.py but with the smoothing spline (GCV) used in fde_real2.py."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
import synth_test as st, fde_core as fc
from scipy.interpolate import make_smoothing_spline
H, LAM = 0.25, 1e-3; TOLS=(0.05,0.1,0.2,0.4,0.7); AGRID=np.round(np.arange(0.40,1.0001,0.025),3)
def grid(W):
    tw=np.arange(len(W)); sc=W.max(0); tg=np.arange(0,len(W)-1+1e-9,H)
    return np.vstack([make_smoothing_spline(tw,W[:,k]/sc[k])(tg) for k in range(W.shape[1])]).T.clip(0)
def dat(a,nz,s,obs='states'):
    rng=np.random.default_rng(s); Y=st.simulate(a)[:st.NTRAIN]
    W=Y[:,:2] if obs=='states' else np.vstack([st.B*Y[:,0]*Y[:,1],st.GM*Y[:,1]]).T
    return grid(W*(1+nz*rng.standard_normal(W.shape)) if nz else W)
def ls(Yg,k,cols):
    Th,_=fc.library(Yg[1:],2); best=None
    for al in np.arange(0.3,1.0001,0.005):
        lhs=fc.caputo_L1(Yg[:,k],al,H); X=Th[:,cols]; c=np.linalg.lstsq(X,lhs,rcond=None)[0]; r=np.mean((lhs-X@c)**2)/np.mean(lhs**2)
        if best is None or r<best[0]: best=(r,al)
    return best[1]
def bic(Yg,k):
    n=len(Yg)-1; best=None; _,nm=fc.library(Yg[:1],2)
    for al in AGRID:
        for tol in TOLS:
            xi,r=fc.fit_eq(Yg,k,al,tol,LAM,2,H); kk=int((xi!=0).sum()); b=n*np.log(r+1e-12)+kk*np.log(n)
            if best is None or b<best[0]: best=(b,float(al),sorted(nm[i] for i,x in enumerate(xi) if x!=0))
    return best[1],best[2]
TRUE=[{'x0*x1'},{'x0*x1','x1'}]
R={'T1s':[],'T2s':[],'T4s':[]}
for a in (0.7,0.85,1.0):
    for nz in (0.0,0.02,0.05):
        reps=range(20) if nz else range(1)
        e0=[ls(dat(a,nz,s),0,[4]) for s in reps]; e1=[ls(dat(a,nz,s),1,[2,4]) for s in reps]
        R['T1s'].append(dict(alpha=a,noise=nz,S_mean=float(np.mean(e0)),S_sd=float(np.std(e0)),I_mean=float(np.mean(e1)),I_sd=float(np.std(e1))))
        for s in (range(5) if nz else range(1)):
            res=[bic(dat(a,nz,s),k) for k in (0,1)]
            R['T2s'].append(dict(alpha=a,noise=nz,seed=s,alpha_hat=[r[0] for r in res],terms=[r[1] for r in res],ok=[set(r[1])==TRUE[k] for k,r in enumerate(res)]))
        g=[ls(dat(a,nz,s,'counts'),1,[1,2]) for s in reps]
        R['T4s'].append(dict(alpha=a,noise=nz,G_mean=float(np.mean(g)),G_sd=float(np.std(g))))
        print(a,nz,R['T1s'][-1],R['T4s'][-1],flush=True)
json.dump(R,open(sys.argv[1],'w'),indent=1)
