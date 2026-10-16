"""Reco rho (truth rho, 1p1n) sides: h_k quality vs reco visible mass; exact-MET classical, reco visible + truth nu, P4 learner. Run from the lxgpu02 run directory."""
import numpy as np, json
L=np.load("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz")
P=np.load("main/posterior.npz"); E=np.load("exactmet/posterior.npz"); w=P["weights"]
t=np.load("/home/rbaba/gen3pi-teacher-20260924/artifacts/targets/validation.npz"); v=t["h_valid"].astype(bool); hx=t["h"][v]
hp4=np.load("/home/rbaba/hz-beyond-ceiling-20260923/artifacts/p4/baseline/validation_predictions.npz")["h_pred"]
def wc(a,b,w):
    f=np.isfinite(a)&np.isfinite(b); a,b,w=a[f],b[f],w[f]
    a=a-np.average(a,weights=w); b=b-np.average(b,weights=w); return np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w))
rm,raw,tm,av=L["reco_h_mode"],L["reco_mode_raw"],L["modes"],L["reco_h_available_side"]
W=np.broadcast_to(w[:,None],rm.shape)
m=av&(rm==1)&(tm==1)&(raw==1)
print("exactmet pair ess median", np.median(E["met_ess_pair"]), "q05", np.quantile(E["met_ess_pair"],0.05), "no-valid", (E["met_nvalid"]==0).mean())
vis=L["reco_visible_tau_lab4"]; mv=np.sqrt(np.maximum(vis[...,3]**2-(vis[...,:3]**2).sum(-1),0))
tv=L["truth_visible_tau_lab4"]; tn=L["truth_neutrino_lab4"]; xt=tv[...,3]/(tv[...,3]+tn[...,3])
print("rho k corr: classical exactMET met",wc(E["met_Eh"][...,2][m],hx[...,2][m],W[m]),"truthnu",wc(P["truthnu_h"][...,2][m],hx[...,2][m],W[m]),"P4",wc(hp4[...,2][m],hx[...,2][m],W[m]))
for lo,hi in [(0,0.6),(0.6,0.9),(0.9,1.2),(1.2,1.6),(1.6,9)]:
    s=m&(mv>=lo)&(mv<hi)
    print(f"m_vis {lo}-{hi} frac {s.sum()/m.sum():.3f}  k: cl-exactMET {wc(E['met_Eh'][...,2][s],hx[...,2][s],W[s]):.2f} truthnu {wc(P['truthnu_h'][...,2][s],hx[...,2][s],W[s]):.2f} P4 {wc(hp4[...,2][s],hx[...,2][s],W[s]):.2f}")
# is exact h_k for rho a function of x and y ? correlation of exact h_k with xt
print("corr(exact h_k, x_true) rho", wc(hx[...,2][m], xt[m], W[m]), "pi", wc(hx[...,2][av&(rm==0)&(tm==0)], xt[av&(rm==0)&(tm==0)], W[av&(rm==0)&(tm==0)]))
print("corr(P4 h_k, x_true) rho", wc(hp4[...,2][m], xt[m], W[m]), " classical exactMET", wc(E["met_Eh"][...,2][m], xt[m], W[m]))
