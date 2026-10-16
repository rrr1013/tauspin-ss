"""Review follow-up: AUC on events with both reco polarimeters; consistency of the visible 4-vector used for the shell with the PFO constituents used in the polarimeter."""
import json, sys
import numpy as np
sys.path.insert(0, "code/analysis/atlas_classical_posterior")
from evaluate import wauc, pair_masks
L = np.load("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz")
P = np.load("main/posterior.npz"); y = P["labels"].astype(int); w = P["weights"]
av = P["reco_h_available_side"].all(1)
out = {"both_available_fraction": float(av.mean())}
lf = np.load("/home/rbaba/gen3pi-teacher-20260924/artifacts/p11_readout/readout_scores_ens_full22.npz")["scores"]
for k, s in {"classical ip3_sv": P["ip3_sv_lr"], "classical ip3_sv_mass": P["ip3_sv_mass_lr"], "learner ens_full22": lf}.items():
    out[f"auc_all:{k}"] = wauc(s, y, w); out[f"auc_both_available:{k}"] = wauc(s[av], y[av], w[av])
vis = L["reco_visible_tau_lab4"]; pf = (L["reco_h_pions_lab4"] * (np.arange(3)[None, None, :, None] < L["reco_h_counts"][..., None, None])).sum(2) + L["reco_h_neutral_lab4"]
rm = L["reco_h_mode"]; a = L["reco_h_available_side"]
for md, lab in ((0, "pi"), (1, "rho"), (3, "3pi")):
    m = a & (rm == md)
    d = np.abs(vis[..., 3] - pf[..., 3])[m] / vis[..., 3][m]
    out[f"rel |E_vis - E_PFO| {lab} median,q95"] = [float(np.median(d)), float(np.quantile(d, 0.95))]
print(json.dumps(out, indent=1))
