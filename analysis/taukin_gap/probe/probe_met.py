import numpy as np
d=np.load('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/train.npz')
vis=d['reco_visible_tau_lab4']; cnt=d['reco_h_counts']; av=d['reco_h_available_side']
hv=(d['reco_h_pions_lab4']*(np.arange(3)[None,None,:,None]<cnt[...,None,None])).sum(2)+d['reco_h_neutral_lab4']
hv=np.where(av[...,None],hv,vis)
snu=d['truth_neutrino_lab4'][:,:,:2].sum(1); met=d['reco_met_xy']; tv=d['truth_visible_tau_lab4']
r0=met-snu
dvis=(vis[:,:,:2]-hv[:,:,:2]).sum(1)
r1=met+dvis-snu
for f in (0.5,1.0):
    r=met+f*dvis-snu; print('shift f',f,'rms',np.sqrt((r**2).mean(0)), 'q68 half', [np.diff(np.quantile(r[:,i],[.16,.84]))[0]/2 for i in (0,1)])
print('no shift rms',np.sqrt((r0**2).mean(0)), [np.diff(np.quantile(r0[:,i],[.16,.84]))[0]/2 for i in (0,1)])
# vis errors
for lab,v in (('tau',vis),('track',hv)):
    e=(v[:,:,:2]-tv[:,:,:2]).sum(1); print(lab,'vis_T error rms',np.sqrt((e**2).mean(0)))
# correlation of MET residual with vis error
e=(vis[:,:,:2]-tv[:,:,:2]).sum(1)
print('corr(MET res, tau vis err) x', np.corrcoef(r0[:,0],e[:,0])[0,1])
# regression coefficient
b=np.sum(r0*e)/np.sum(e*e); print('slope MET res on tau vis err', b)
e2=(hv[:,:,:2]-tv[:,:,:2]).sum(1); b2=np.sum(r0*e2)/np.sum(e2*e2); print('slope on track vis err',b2, 'corr',np.corrcoef(r0[:,0],e2[:,0])[0,1])
# MET residual vs ditau visible pT
vt=vis[:,:,:2].sum(1); pt=np.linalg.norm(vt,axis=1); u=vt/pt[:,None]
rpar=(r0*u).sum(1); rperp=r0[:,0]*u[:,1]-r0[:,1]*u[:,0]
qs=np.quantile(pt,np.linspace(0,1,6))
for a,b_ in zip(qs[:-1],qs[1:]):
    s=(pt>=a)&(pt<b_); print(f'vis pT {a:.0f}-{b_:.0f}: par median {np.median(rpar[s]):.1f} q68 {np.diff(np.quantile(rpar[s],[.16,.84]))[0]/2:.1f}  perp q68 {np.diff(np.quantile(rperp[s],[.16,.84]))[0]/2:.1f}  rms {np.sqrt(np.mean(rpar[s]**2)):.1f}/{np.sqrt(np.mean(rperp[s]**2)):.1f}')
