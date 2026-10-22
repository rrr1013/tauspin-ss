import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
from hquality import wcorr
import posterior as CP
I=np.load('/home/rbaba/taukin-gap-20261023/out/learner_intervene2.npz')
ref=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz'); hx=ref['h'].astype(float)
P=np.load('/home/rbaba/taukin-gap-20261023/out/val_track_dl/posterior.npz')
w=P['weights'].astype(float); modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']
msk=av&(rm==0)&(modes==0); W=np.broadcast_to(w[:,None],msk.shape)
tracks=dict(np.load(CP.TRACKS)); m=CP.P12.ip_measurements(tracks,P['global_indices'],0)
x=np.log10(np.maximum(m['mag_um'],1e-3)); e=np.quantile(x[msk],np.linspace(0,1,7))
phi=lambda h: np.arctan2(h[...,1],h[...,0])
hc=P['ip3_sv_mass_Eh']
print('pi sides by |IP| sextile (um edges):',np.round(10**e,1))
for v in ('identity','rot90','flip','ipmag_const','offset_zero'):
    h=I[v]; row=[]
    for i in range(6):
        s=msk&(x>=e[i])&(x<=e[i+1])
        dphi=np.angle(np.exp(1j*(phi(h)-phi(I['identity']))))[s]
        row.append('%.2f/%.2f/%.2f shift%+.0f'%(wcorr(h[...,0][s],hx[...,0][s],W[s]),wcorr(h[...,1][s],hx[...,1][s],W[s]),wcorr(h[...,2][s],hx[...,2][s],W[s]),np.degrees(np.median(dphi))))
    print(f'{v:12s}',' | '.join(row))
row=[]
for i in range(6):
    s=msk&(x>=e[i])&(x<=e[i+1]); row.append('%.2f/%.2f/%.2f'%tuple(wcorr(hc[...,c][s],hx[...,c][s],W[s]) for c in range(3)))
print('classical   ',' | '.join(row))
