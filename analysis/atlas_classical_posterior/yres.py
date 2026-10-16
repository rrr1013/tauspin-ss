"""Resolution of the rho energy asymmetry y = (E_pi - E_pi0)/(E_pi + E_pi0): reco PFO vs truth, truth-rho reco-1p1n sides."""
import numpy as np
L=np.load("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz")
S=np.load("/home/rbaba/gen3pi-teacher-20260924/artifacts/targets/truth_surface_gen3pi.npz")
print(S.files)
ids=L["global_indices"]; sid=S["global_indices"]
pos={g:i for i,g in enumerate(sid)}
ix=np.array([pos.get(g,-1) for g in ids]); ok=ix>=0; print("matched",ok.mean())
w=L["weights"]
rm=L["reco_h_mode"]; raw=L["reco_mode_raw"]; tm=L["modes"]; av=L["reco_h_available_side"]
sel=av&(rm==1)&(tm==1)&(raw==1)&ok[:,None]
tp=S["pions"][ix]; t0=S["pi0"][ix]
Etp=tp[...,0,3]; Et0=t0[...,3]; yt=(Etp-Et0)/(Etp+Et0)
rp=L["reco_h_pions_lab4"][...,0,3]; r0=L["reco_h_neutral_lab4"][...,3]; ev=L["reco_visible_tau_lab4"][...,3]
y_pfo=(rp-r0)/(rp+r0); y_cal=(2*rp-ev)/ev
def wc(a,b,w):
    a=a-np.average(a,weights=w); b=b-np.average(b,weights=w); return np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w))
W=np.broadcast_to(w[:,None],sel.shape)[sel]
print("truth-rho reco-1p1n sides",sel.sum())
print("corr(y_true, y_pfo)", wc(yt[sel],y_pfo[sel],W), "corr(y_true, y_cal)", wc(yt[sel],y_cal[sel],W))
print("rms y_pfo-y_true", np.sqrt(np.average((y_pfo-yt)[sel]**2,weights=W)), "rms y_cal-y_true", np.sqrt(np.average((y_cal-yt)[sel]**2,weights=W)))
ev_t=(tp[...,0,3]+t0[...,3])
print("E_vis reco/true: pfo-sum median",np.median(((rp+r0)/ev_t)[sel]),"q16-84",np.quantile(((rp+r0)/ev_t)[sel],[.16,.84]),"calibrated",np.median((ev/ev_t)[sel]),np.quantile((ev/ev_t)[sel],[.16,.84]))
print("E_pi0 reco/true", np.quantile((r0/Et0)[sel],[.16,.5,.84]), "E_pi+ reco/true", np.quantile((rp/Etp)[sel],[.16,.5,.84]))
