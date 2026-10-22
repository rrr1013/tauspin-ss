import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
from hquality import wcorr
P=np.load('/home/rbaba/taukin-gap-20261023/out/val_track_dl_joint/posterior.npz')
ids=P['global_indices']; w=P['weights'].astype(float); lab=P['labels']
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); hx=t['h'][v].astype(float)
hl=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz')['h_pred'].astype(float)
hc=P['ip3_sv_mass_Eh']; hc0=P['ip3_sv_Eh']
modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']
msk=av&(rm==0)&(modes==0); W=np.broadcast_to(w[:,None],msk.shape)
other=modes[:,::-1]
d=np.load('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')
m=d['truth_parent_mass_GeV']; on=np.broadcast_to((np.abs(m-91.1876)<0.05)[:,None],msk.shape)
L=np.broadcast_to(lab[:,None],msk.shape)
def q(h,s): return '%.3f/%.3f/%.3f'%tuple(wcorr(h[...,c][s],hx[...,c][s],W[s]) for c in range(3))
for name,s in (('other pi',msk&(other==0)),('other rho',msk&(other==1)),('other 3pi',msk&(other==3)),('mass on-shell',msk&on),('mass off',msk&~on),('H events',msk&(L==1)),('Z events',msk&(L==0))):
    print(f'{name:14s} n={s.sum():6d} learner {q(hl,s)}  classical(+mass) {q(hc,s)}  classical(no mass) {q(hc0,s)}')
