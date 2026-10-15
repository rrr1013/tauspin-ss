"""Ceiling of the per-side polarimeter on the measured visible system: true neutrinos,
reconstructed visible decay products, reco decay class (all events).  Pearson correlation
with the exact h per reco class and component, written as JSON."""
import json
import sys

import numpy as np

from posterior_h import RMODE_TO_CLS, load_reco, ordered
from polarimeter_pythia import polarimeter_nrk

d = load_reco("/home/rbaba/dihiggs-spin-20261006/runs/ent/reco")
o = ordered(d)
cls = RMODE_TO_CLS[o["rmode"]]
nu = o["tau_nu_p4"]
h = np.concatenate([polarimeter_nrk(o["ch"][a:a + 20000], o["ch_q"][a:a + 20000], o["pi0"][a:a + 20000],
                                    nu[a:a + 20000], (o["vis"] + nu)[a:a + 20000], cls[a:a + 20000])
                    for a in range(0, len(cls), 20000)])
hx = d["h_exact"]
out = {}
for c, nm in enumerate(["pi", "rho", "a1(1p)", "a1(3p)"]):
    m = (cls == c) & np.isfinite(h).all(-1) & np.isfinite(hx).all(-1)
    out[nm] = {k: float(np.corrcoef(h[..., j][m], hx[..., j][m])[0, 1]) for j, k in enumerate("nrk")}
    out[nm]["n_sides"] = int(m.sum())
T = lambda a: a[:, 0, 0] * a[:, 1, 0] + a[:, 0, 1] * a[:, 1, 1]
both = (cls >= 0).all(1) & np.isfinite(h).all((1, 2)) & np.isfinite(hx).all((1, 2))
out["pair_T_all"] = float(np.corrcoef(T(h)[both], T(hx)[both])[0, 1])
np.savez_compressed(sys.argv[1].replace(".json", ".npz"), h_oracle=h)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
