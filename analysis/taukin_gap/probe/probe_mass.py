import numpy as np, sys
sys.path.insert(0,'/home/rbaba/rho-pi0-marginal-20261022/code/analysis/atlas_classical_posterior')
from evaluate import GEN
d=np.load('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')
m=d['truth_parent_mass_GeV']; lab=d['labels']
for L in (0,1):
    x=m[lab==L]; print('label',L,'n',len(x),'q',np.quantile(x,[.01,.05,.1,.2,.5,.8,.9,.95,.99]).round(3), 'frac |m-91.1876|<0.01', np.mean(np.abs(x-91.1876)<0.01).round(3), '<1GeV',np.mean(np.abs(x-91.1876)<1).round(3))
tv=d['truth_visible_tau_lab4']+d['truth_neutrino_lab4']; tt=tv.sum(1); mm=np.sqrt(np.maximum(tt[:,3]**2-(tt[:,:3]**2).sum(1),0))
print('m(tau tau) truth q', np.quantile(mm,[.01,.1,.5,.9,.99]).round(3))
P=np.load('/home/rbaba/rho-pi0-marginal-20261022/out/val_marg_invE/posterior.npz')
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); hx=t['h'][v].astype(float)
ids=P['global_indices']; assert np.array_equal(d['global_indices'],ids)
w=P['weights'].astype(float); rm=P['reco_h_mode'].astype(int); av=P['reco_h_available_side']; modes=P['modes']
hl=np.load(GEN/'p11'/'base_s43'/'validation_predictions.npz')['h_pred']; hc=P['ip3_sv_mass_Eh']; hm=P['met_Eh']
on=np.abs(m-91.1876)<0.05
A=np.average
for lab_,ev in (('on-shell',on),('off-shell',~on)):
    msk=(rm==0)&(modes==0)&av&ev[:,None]
    ww=np.broadcast_to(w[:,None],msk.shape)[msk]
    def wc(a,b):
        a=a-A(a,weights=ww); b=b-A(b,weights=ww); return A(a*b,weights=ww)/np.sqrt(A(a*a,weights=ww)*A(b*b,weights=ww))
    print(lab_, msk.sum(), 'h_k learner base', round(wc(hl[...,2][msk],hx[...,2][msk]),3), 'classical ip3svmass', round(wc(hc[...,2][msk],hx[...,2][msk]),3), 'classical met', round(wc(hm[...,2][msk],hx[...,2][msk]),3))
