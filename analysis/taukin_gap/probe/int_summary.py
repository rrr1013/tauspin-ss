import numpy as np, sys, json
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN
from hquality import wcorr
import posterior as CP
O='/home/rbaba/taukin-gap-20261023/out/'
I={**dict(np.load(O+'learner_intervene.npz')), **{k:v for k,v in np.load(O+'learner_intervene2.npz').items() if k not in ('identity','global_indices') and not k.startswith('shuffle')}, 'tok_d0z0_zero':np.load(O+'learner_intervene3.npz')['tok_d0z0_zero']}
ref=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz'); hx=ref['h'].astype(float)
P=np.load(O+'val_base/posterior.npz'); assert np.array_equal(P['global_indices'],I['global_indices'])
w=P['weights'].astype(float); modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']; W=np.broadcast_to(w[:,None],av.shape)
m=CP.P12.ip_measurements(dict(np.load(CP.TRACKS)),P['global_indices'],0); x=np.log10(np.maximum(m['mag_um'],1e-3))
out={'note':'learner full22 s42, validation; weighted corr with exact h (n,r,k); variants of learner_intervene.py','parity_identity_vs_stored_mean_abs':float(np.abs(I['identity']-ref['h_pred']).mean())}
def q(h,s): return [round(wcorr(h[...,c][s],hx[...,c][s],W[s]),4) for c in range(3)]
for g,(tm,r) in {'pi':(0,0),'rho':(1,1),'3pi':(3,3)}.items():
    s=av&(rm==r)&(modes==tm); o={'identity':q(I['identity'],s)}
    for v in ('shuffle_other','shuffle_own'):
        o[v]=q(np.stack([I[f'{v}_0'][:,0],I[f'{v}_1'][:,1]],1),s)
    out[g]=o
s=av&(rm==0)&(modes==0)&m['available']; e=np.quantile(x[s],np.linspace(0,1,7)); out['pi_ip_sextile_edges_um']=np.round(10**e,1).tolist()
for v in ('identity','rot90','flip','ipmag_const','offset_zero','tok_d0z0_zero'):
    out['pi_by_ip_sextile_'+v]=[q(I[v],s&(x>=e[i])&(x<=e[i+1])) for i in range(6)]
json.dump(out,open(O+'learner_intervene_summary.json','w'),indent=1); print(json.dumps(out)[:600])
