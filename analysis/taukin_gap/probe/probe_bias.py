import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
import posterior as CP
P12=CP.P12
with np.load(CP.LADDER/'train.npz') as d:
    ids=d['global_indices'][:250000]; tt=(d['truth_visible_tau_lab4']+d['truth_neutrino_lab4'])[:250000]
tracks=dict(np.load(CP.TRACKS)); m=P12.ip_measurements(tracks,ids,0)
psi_t=P12.azimuth_in_frame(CP.unit(tt[...,:3]),m['e1'],m['e2'])
dp=np.angle(np.exp(1j*(psi_t-m['psi'])))
ok=m['available']&(m['n_core']<2)
for lo,hi in ((10,25),(25,60),(60,125)):
    s=ok&(m['mag_um']>=lo)&(m['mag_um']<hi)
    pb=np.floor((m['psi']+np.pi)/(2*np.pi)*8).astype(int)%8
    print(f'|IP| {lo}-{hi}: by psi_meas octant (centre deg): circular mean of dpsi [deg], median|dpsi|')
    print('   ', [(int(-180+22.5+45*k), round(float(np.degrees(np.angle(np.mean(np.exp(1j*dp[s&(pb==k)]))))),1), round(float(np.degrees(np.median(np.abs(dp[s&(pb==k)])))),1)) for k in range(8)])
# distribution of psi_true itself (is the tau azimuth around the track uniform?)
s=ok
h,_=np.histogram(psi_t[s],np.linspace(-np.pi,np.pi,9)); print('psi_true octant fractions', np.round(h/h.sum(),3))
h,_=np.histogram(m['psi'][s],np.linspace(-np.pi,np.pi,9)); print('psi_meas octant fractions', np.round(h/h.sum(),3))
