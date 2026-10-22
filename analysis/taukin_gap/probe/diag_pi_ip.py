import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
from hquality import wcorr
import posterior as CP
P=np.load('/home/rbaba/taukin-gap-20261023/out/val_track_dl/posterior.npz')
ids=P['global_indices']; w=P['weights'].astype(float)
t=np.load(GEN/'targets'/'validation.npz'); v=t['h_valid'].astype(bool); hx=t['h'][v].astype(float); assert np.array_equal(t['global_indices'][v],ids)
hl=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz')['h_pred'].astype(float)
hb=np.load(GEN/'p11'/'base_s43'/'validation_predictions.npz')['h_pred'].astype(float)
hc=P['ip3_sv_mass_Eh']
modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']
tracks=dict(np.load(CP.TRACKS)); m=CP.P12.ip_measurements(tracks, ids, 0); trk=tracks['trk'][ids]
msk=av&(rm==0)&(modes==0)&m['available']
W=np.broadcast_to(w[:,None],msk.shape)
def table(var,name,q=6):
    x=var[msk]; e=np.quantile(x,np.linspace(0,1,q+1))
    print(name, 'bin edges', np.round(e,2))
    for lab,h in (('classical',hc),('learner',hl),('base',hb)):
        out=[]
        for i in range(q):
            s=np.zeros_like(msk); s[msk]=(x>=e[i])&(x<=e[i+1])
            out.append('%.2f/%.2f/%.2f'%tuple(wcorr(h[...,c][s],hx[...,c][s],W[s]) for c in range(3)))
        print('  %-10s'%lab,' '.join(out))
table(np.log10(np.maximum(m['mag_um'],1e-3)),'log10|IP| um')
table(trk[...,0,0],'track pT')
eta=np.abs(-np.log(np.tan(trk[...,0,1]/2))); table(eta,'|eta|')
