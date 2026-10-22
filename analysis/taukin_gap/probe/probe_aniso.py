import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
import posterior as CP
P12=CP.P12
with np.load(CP.LADDER/'train.npz') as d:
    ids=d['global_indices'][:200000]; tt=(d['truth_visible_tau_lab4']+d['truth_neutrino_lab4'])[:200000]
tracks=dict(np.load(CP.TRACKS)); m=P12.ip_measurements(tracks,ids,0)
psi_t=P12.azimuth_in_frame(CP.unit(tt[...,:3]),m['e1'],m['e2'])
dp=np.abs(np.angle(np.exp(1j*(psi_t-m['psi']))))
fold=np.abs(np.angle(np.exp(1j*np.where(np.abs(m['psi'])>np.pi/2,m['psi']-np.pi,m['psi']))))
ok=m['available']&(m['n_core']<2)
for lo,hi in ((10,20),(20,40),(40,80),(80,160)):
    s=ok&(m['mag_um']>=lo)&(m['mag_um']<hi)
    print(f'|IP| {lo}-{hi} um: median |dpsi| [deg] by orientation bin (0=d0 axis ... 3=z0 axis):',
          [round(float(np.degrees(np.median(dp[s&(fold>=a)&(fold<a+np.pi/8)]))),1) for a in np.arange(4)*np.pi/8], 'n',s.sum())
