"""Slim copy of the regressor hold-out events (same order as *_holdout.npz):
kinematic matching variables, textbook spin observables and exact h."""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
ho = np.flatnonzero((d["uid"] // 7) % 5 == 4)
cols = ["kin_pt_tau1", "kin_pt_tau2", "kin_eta_tau1", "kin_eta_tau2", "kin_m_vis", "kin_met",
        "kin_dr_tautau", "kin_m_tautau", "kin_mmc_status"]
cols += [k for k in d.files if k.startswith("spin_") and any(t in k for t in ("upsilon", "_x_", "hhyb", "rmode"))]
np.savez_compressed(sys.argv[2], proc=d["proc"][ho], uid=d["uid"][ho], h_exact=d["h_exact"][ho],
                    **{c: d[c][ho] for c in cols})
print(len(ho))
