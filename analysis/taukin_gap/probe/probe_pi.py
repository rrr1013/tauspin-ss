import numpy as np, sys, glob
sys.path.insert(0,'/home/rbaba/rho-pi0-marginal-20261022/code/analysis/atlas_classical_posterior')
from evaluate import GEN, wauc, pair_masks
P=np.load('/home/rbaba/rho-pi0-marginal-20261022/out/val_marg_invE/posterior.npz')
ids,w=P['global_indices'],P['weights'].astype(float)
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); hx=t['h'][v].astype(float)
rm=P['reco_h_mode'].astype(int); av=P['reco_h_available_side']; modes=P['modes']
rows={}
for m in ('met','ip3','sv','ip3_sv','ip3_sv_mass'):
    rows['classical '+m]=P[m+'_Eh']
for arm in ('base_s43','full22_s42'):
    rows['learner '+arm]=np.load(GEN/'p11'/arm/'validation_predictions.npz')['h_pred']
print(glob.glob('/home/rbaba/atlas-classical-posterior-20261017/*'))
for p in glob.glob('/home/rbaba/atlas-classical-posterior-20261017/*/posterior.npz'):
    Q=np.load(p)
    if np.array_equal(Q['global_indices'],ids): rows[p.split('/')[-2]+' ip3_sv_mass']=Q['ip3_sv_mass_Eh']; rows[p.split('/')[-2]+' met']=Q['met_Eh']
msk=(rm==0)&(modes==0)&av
ww=np.broadcast_to(w[:,None],msk.shape)[msk]
A=np.average
def wc(a,b):
    a=a-A(a,weights=ww); b=b-A(b,weights=ww); return A(a*b,weights=ww)/np.sqrt(A(a*a,weights=ww)*A(b*b,weights=ww))
for k,h in rows.items():
    h=np.where(np.isfinite(h),h,0)
    print(f"{k:45s}", [round(wc(h[...,i][msk],hx[...,i][msk]),3) for i in range(3)])
# visible consistency
d=np.load('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')
vis=d['reco_visible_tau_lab4']; pi=d['reco_h_pions_lab4'][:,:,0]; neu=d['reco_h_neutral_lab4']; tv=d['truth_visible_tau_lab4']
for lab,s in (('reco pi/truth pi',msk),('reco rho/truth rho',(rm==1)&(modes==1)&av),('reco 3pi',(rm==3)&av)):
    cnt=d['reco_h_counts']
    hv=(d['reco_h_pions_lab4']*(np.arange(3)[None,None,:,None]<cnt[...,None,None])).sum(2)+neu
    r1=vis[...,3][s]/tv[...,3][s]-1; r2=hv[...,3][s]/tv[...,3][s]-1
    print(lab,'E_vis(reco_visible)/truth-1 q16,50,84',np.quantile(r1,[.16,.5,.84]).round(4),' E(pions+neutral)/truth-1',np.quantile(r2,[.16,.5,.84]).round(4), 'max|diff| vis vs h-sum', np.quantile(np.abs(vis[s]-hv[s]).max(-1),[.5,.99]))
