import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
import posterior as CP
d=np.load(CP.LADDER/'validation.npz')
ids=d['global_indices']; tracks=dict(np.load(CP.TRACKS)); m=CP.P12.ip_measurements(tracks,ids,0)
pi=d['reco_h_pions_lab4'][:,:,0,:3]; vis=d['reco_visible_tau_lab4'][...,:3]
tt=d['truth_visible_tau_lab4']+d['truth_neutrino_lab4']; tdir=CP.unit(tt[...,:3])
s=d['reco_h_available_side']&(d['reco_h_mode']==0)&(d['modes']==0)&m['available']
ang=lambda a,b: np.arccos(np.clip(np.sum(CP.unit(a)*CP.unit(b),-1),-1,1))
print('mrad q16/50/84 on truth-pi reco-pi sides')
print(' ladder pion dir vs track-file dir', np.quantile(1e3*ang(pi,m['tdir'])[s],[.16,.5,.84]).round(3))
print(' tau 4-vector dir vs track-file dir', np.quantile(1e3*ang(vis,m['tdir'])[s],[.16,.5,.84]).round(3))
print(' truth tau vs track (alpha)', np.quantile(1e3*ang(tdir,m['tdir'])[s],[.16,.5,.84]).round(3))
print(' truth tau vs ladder pion', np.quantile(1e3*ang(tdir,pi)[s],[.16,.5,.84]).round(3))
pt_l=np.hypot(pi[...,0],pi[...,1]); pt_t=tracks['trk'][ids][:,:,0,0]
print(' pT ladder/trackfile -1', np.quantile((pt_l/pt_t-1)[s],[.16,.5,.84]).round(4))
# component of offset: azimuth of ladder-pion offset around track relative to the IP direction
