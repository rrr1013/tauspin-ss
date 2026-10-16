"""Per-side h quality on reco rho / pi sides split by truth mode and reco raw mode (1p1n / 1p2n). Run from the lxgpu02 run directory."""
import numpy as np
L=np.load("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz")
P=np.load("main/posterior.npz"); E=np.load("exactmet/posterior.npz"); w=P["weights"]
t=np.load("/home/rbaba/gen3pi-teacher-20260924/artifacts/targets/validation.npz"); v=t["h_valid"].astype(bool); hx=t["h"][v]
def wc(a,b,w):
    a=a-np.average(a,weights=w); b=b-np.average(b,weights=w); return np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w))
H={"truthnu":P["truthnu_h"],"classical ipsv_mass":P["ip3_sv_mass_Eh"],"classical exactMET ipsv_mass":E["ip3_sv_mass_Eh"]}
for arm,p in [("learner full22","/home/rbaba/gen3pi-teacher-20260924/artifacts/p11/full22_s42"),("learner P4 base","/home/rbaba/hz-beyond-ceiling-20260923/artifacts/p4/baseline")]:
    H[arm]=np.load(p+"/validation_predictions.npz")["h_pred"]
rm=L["reco_h_mode"]; raw=L["reco_mode_raw"]; tm=L["modes"]; av=L["reco_h_available_side"]
W=np.broadcast_to(w[:,None],rm.shape)
nrho=(av&(rm==1)).sum(); npi=(av&(rm==0)).sum()
cats=[("reco rho, truth rho, raw 1p1n",(rm==1)&(tm==1)&(raw==1),nrho),("reco rho, truth rho, raw 1p2n",(rm==1)&(tm==1)&(raw==2),nrho),("reco rho, truth pi",(rm==1)&(tm==0),nrho),("reco rho, truth 3pi",(rm==1)&(tm==3),nrho),("reco pi, truth pi",(rm==0)&(tm==0),npi),("reco pi, truth rho",(rm==0)&(tm==1),npi)]
for name,m,den in cats:
    m=m&av
    print(f"{name:32s} frac {m.sum()/den:.3f} n={m.sum()}")
    if m.sum()<200: continue
    for k,h in H.items(): print(f"   {k:30s}", ",".join(f"{wc(h[...,i][m],hx[...,i][m],W[m]):.2f}" for i in range(3)))
