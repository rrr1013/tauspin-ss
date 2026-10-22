import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
P=np.load('/home/rbaba/taukin-gap-20261023/out/val_track_dl/posterior.npz')
w=P['weights'].astype(float)
ref=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz'); hx=ref['h'].astype(float); hl=ref['h_pred'].astype(float)
hc=P['ip3_sv_mass_Eh']
modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']
msk=av&(rm==0)&(modes==0)
W=np.broadcast_to(w[:,None],msk.shape)[msk]
phi=lambda h: np.arctan2(h[...,1],h[...,0])
dx=phi(hx)[msk]
for lab,h in (('classical',hc),('learner',hl)):
    d=np.angle(np.exp(1j*(phi(h)[msk]-dx)))
    print(lab,'|dphi| deg q25/50/75', np.degrees(np.quantile(np.abs(d),[.25,.5,.75])).round(1), 'E cos', np.average(np.cos(d),weights=W).round(3),
          'mean |h_perp| est/exact', np.average(np.hypot(h[...,0],h[...,1])[msk],weights=W).round(3), np.average(np.hypot(hx[...,0],hx[...,1])[msk],weights=W).round(3))
# reliability of classical: R = |E h_perp| / |h_perp exact| (a concentration proxy)
R=(np.hypot(hc[...,0],hc[...,1])/np.maximum(np.hypot(hx[...,0],hx[...,1]),0.05))[msk]
d=np.angle(np.exp(1j*(phi(hc)[msk]-dx))); dl=np.angle(np.exp(1j*(phi(hl)[msk]-dx)))
e=np.quantile(R,np.linspace(0,1,9))
print('classical concentration proxy bins: median R, E[cos dphi] classical, E[cos dphi] learner')
for a,b in zip(e[:-1],e[1:]):
    s=(R>=a)&(R<=b); print(f'  R~{np.median(R[s]):.2f}: classical {np.average(np.cos(d[s]),weights=W[s]):.3f}  learner {np.average(np.cos(dl[s]),weights=W[s]):.3f}')
L=np.broadcast_to(P['labels'][:,None],msk.shape)[msk]
print('by label within classical concentration bins (E cos dphi): classical H / Z, learner H / Z')
for a,b in zip(e[:-1],e[1:]):
    s=(R>=a)&(R<=b)
    f=lambda x,t: np.average(np.cos(x[s&(L==t)]),weights=W[s&(L==t)])
    print(f'  R~{np.median(R[s]):.2f}: classical {f(d,1):.3f} / {f(d,0):.3f}   learner {f(dl,1):.3f} / {f(dl,0):.3f}')
# learner with partner geometry shuffled
I=np.load('/home/rbaba/taukin-gap-20261023/out/learner_intervene.npz')
hs=np.stack([I['shuffle_other_0'][:,0],I['shuffle_other_1'][:,1]],1)
ds_=np.angle(np.exp(1j*(phi(hs)[msk]-dx)))
print('learner, partner geometry shuffled, by bin H / Z:')
for a,b in zip(e[:-1],e[1:]):
    s=(R>=a)&(R<=b)
    f=lambda x,t: np.average(np.cos(x[s&(L==t)]),weights=W[s&(L==t)])
    print(f'  R~{np.median(R[s]):.2f}: {f(ds_,1):.3f} / {f(ds_,0):.3f}')
