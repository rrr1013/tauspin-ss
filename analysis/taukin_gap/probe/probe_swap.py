import numpy as np, sys
sys.path.insert(0,'/home/rbaba/rho-pi0-marginal-20261022/code/analysis/atlas_classical_posterior')
from evaluate import GEN, wauc, pair_masks
P=np.load('/home/rbaba/rho-pi0-marginal-20261022/out/val_marg_invE/posterior.npz')
ids,y,w=P['global_indices'],P['labels'].astype(int),P['weights'].astype(float)
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); assert np.array_equal(t['global_indices'][v],ids)
hx=t['h'][v].astype(float)
hp=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz')
print(hp.files)
hl=hp['h_pred'].astype(float)
hc=P['ip3_sv_mass_Eh'].astype(float)
rm=P['reco_h_mode'].astype(int); av=P['reco_h_available_side']; modes=P['modes']
def lrp(h):
    a,b=h[:,0],h[:,1]
    T=a[:,0]*b[:,0]+a[:,1]*b[:,1]-a[:,2]*b[:,2]; L=a[:,2]*b[:,2]
    return np.log(np.clip(1+T,1e-3,None))-np.log(np.clip(1+L,1e-3,None))
sel=pair_masks(modes)
hc=np.where(np.isfinite(hc),hc,0)
def show(name,h):
    s=lrp(h); print(f"{name:40s}", ' '.join(f"{k}:{wauc(s[m],y[m],w[m]):.4f}" for k,m in sel.items()))
show('classical E[h] point',hc); show('learner s42 point',hl); show('exact',hx)
print('classical exact LR', wauc(P['ip3_sv_mass_lr'],y,w))
for T,name in ((0,'pi'),(1,'rho'),(3,'3pi')):
    m=(rm==T)[...,None]
    show(f'classical, learner on reco {name}', np.where(m,hl,hc))
    show(f'learner, classical on reco {name}', np.where(m,hc,hl))
# migrated sides
for lab,msk in (('reco pi truth pi',(rm==0)&(modes==0)&av),('reco pi truth rho',(rm==0)&(modes==1)&av),('reco rho truth pi',(rm==1)&(modes==0)&av)):
    ww=np.broadcast_to(w[:,None],msk.shape)[msk]
    from numpy import average as A
    def wc(a,b):
        a=a-A(a,weights=ww); b=b-A(b,weights=ww); return A(a*b,weights=ww)/np.sqrt(A(a*a,weights=ww)*A(b*b,weights=ww))
    print(lab, msk.sum(), 'classical', [round(wc(hc[...,i][msk],hx[...,i][msk]),3) for i in range(3)], 'learner',[round(wc(hl[...,i][msk],hx[...,i][msk]),3) for i in range(3)])
