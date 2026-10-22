"""TRAIN (event_number % 5 != 0), 1-prong leading track: the pull of the measured IP azimuth toward the z0 axis, by |eta|,
track pT and |IP|, for the PV-referenced IP (psi) and the beam-spot-referenced legacy IP (psi_legacy).
Statistic: circular mean of (truth - measured) azimuth for measured azimuth in [40, 80] deg modulo the reflections
(sign folded as in --ip-orient), and the measured-azimuth anisotropy A = P(|sin psi| > sin 67.5) / P(|sin psi| < sin 22.5)."""
import numpy as np, sys, json
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
import posterior as CP
P12=CP.P12
with np.load(CP.LADDER/'train.npz') as d:
    ids, ev = d['global_indices'], d['event_numbers']; tt=d['truth_visible_tau_lab4']+d['truth_neutrino_lab4']
keep=ev%5!=0
tracks=dict(np.load(CP.TRACKS)); m=P12.ip_measurements(tracks,ids,0); trk=tracks['trk'][ids][:,:,0]
td=CP.unit(tt[...,:3]); psi_t=P12.azimuth_in_frame(td,m['e1'],m['e2'])
eta=np.abs(-np.log(np.tan(trk[...,1]/2))); pt=trk[...,0]
ok=m['available']&(m['n_core']<2)&keep[:,None]
out={}
def stat(psim, s):
    u=np.mod(psim,np.pi); sg=np.where(u>np.pi/2,-1,1); uf=np.where(u>np.pi/2,np.pi-u,u)
    dp=np.angle(np.exp(1j*(psi_t-psim)))*sg
    w=s&(uf>np.radians(40))&(uf<np.radians(80))
    pull=float(np.degrees(np.angle(np.mean(np.exp(1j*dp[w])))))
    a=np.abs(np.sin(psim[s])); A=float(np.mean(a>np.sin(np.radians(67.5)))/np.mean(a<np.sin(np.radians(22.5))))
    return round(pull,1), round(A,2), int(s.sum())
for key,psim,mag in (('PV-referenced',m['psi'],m['mag_um']),('beam-spot legacy',m['psi_legacy'],m['mag_legacy_um'])):
    for lo,hi in ((25,60),(60,125)):
        sel=ok&(mag>=lo)&(mag<hi)
        r={}
        for nm,var,edges in (('|eta|',eta,[0,0.5,1.0,1.5,2.5]),('pT',pt,[0,30,60,120,1e4])):
            r[nm]={f'{a}-{b}':stat(psim, sel&(var>=a)&(var<b)) for a,b in zip(edges[:-1],edges[1:])}
        out[f'{key} |IP| {lo}-{hi}']=r
        print(key, lo, hi, json.dumps(r))
json.dump(out,open('/home/rbaba/taukin-gap-20261023/out/probe_mech.json','w'),indent=1)
