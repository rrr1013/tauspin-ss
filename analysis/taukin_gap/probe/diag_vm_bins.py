import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
from hquality import wcorr
import posterior as CP
base='/home/rbaba/taukin-gap-20261023/out/'
arms={k:np.load(base+f'sub_{k}/posterior.npz') for k in ('S16k','vm2','vm6','vm20','truthpsi')}
ref=arms['S16k']; ids=ref['global_indices']; w=ref['weights'].astype(float)
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); look={int(g):i for i,g in enumerate(t['global_indices'][v])}; pos=np.array([look[int(g)] for g in ids])
hx=t['h'][v][pos].astype(float)
hl=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz')['h_pred'][pos]
modes,rm,av=ref['modes'],ref['reco_h_mode'].astype(int),ref['reco_h_available_side']
tracks=dict(np.load(CP.TRACKS)); m=CP.P12.ip_measurements(tracks,ids,0)
msk=av&(rm==0)&(modes==0)&m['available']; W=np.broadcast_to(w[:,None],msk.shape)
x=np.log10(np.maximum(m['mag_um'],1e-3)); e=np.quantile(x[msk],np.linspace(0,1,5))
print('|IP| quartile edges um', np.round(10**e,1))
rows={'learner':hl, **{k:a['ip3_sv_mass_Eh'] for k,a in arms.items()}}
for k,h in rows.items():
    out=[]
    for i in range(4):
        s=msk&(x>=e[i])&(x<=e[i+1]); out.append('%.2f/%.2f'%(wcorr(h[...,0][s],hx[...,0][s],W[s]),wcorr(h[...,1][s],hx[...,1][s],W[s])))
    print(f'{k:9s}',' | '.join(out))
