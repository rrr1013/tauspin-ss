"""Variant: background spin template from the real Z(125)+jet hold-out events
(Z kinematics and Pythia's Z spin state), signal templates from the spin-flat
H sample; common 2D binning defined on the flat sample.  Mass-binned HL-LHC
categories as in ent_hllhc.py; also with the response-scale nuisance."""
import json
import numpy as np
from ent_closure_syst import z_ent_shape
from ent_hllhc import CATS, LUMI


def binning(G, nq=8):
    e1 = np.quantile(G[:, 0], np.linspace(0, 1, nq + 1)[1:-1])
    e2 = []
    b1 = np.searchsorted(e1, G[:, 0])
    for i in range(nq):
        m = b1 == i
        e2.append(np.quantile(G[m, 1], np.linspace(0, 1, nq + 1)[1:-1]))
    def idx(X):
        a = np.searchsorted(e1, X[:, 0])
        return a * nq + np.array([np.searchsorted(e2[i], x) for i, x in zip(a, X[:, 1])])
    return idx, nq * nq


z = np.load("outputs/ent/closure_g.npz")
G, Gh, isH = z["G"], z["Gh"], z["isH"]
idx, nb = binning(G)
bf, bh = idx(G), idx(Gh)
T = {k: np.bincount(bf, weights=z[f"w{k}"], minlength=nb) / z[f"w{k}"].sum() for k in ("H", "W13", "U")}
T["Z"] = np.bincount(bh[~isH], minlength=nb) / (~isH).sum()
acc = json.load(open("outputs/ent/mass_acc.json"))
out = {}
for sig in (0.0, 0.1, 0.2):
    z2 = 0.0
    for k, (s_, zz, f, t, o) in CATS.items():
        for i in range(len(acc["S"])):
            S = s_ * LUMI * 1.13 * acc["S"][i]
            B = (zz * acc["Z"][i] + f * acc["F"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15
            z2 += z_ent_shape(T, S, B, 0.02, sig) ** 2
    out[str(sig)] = float(np.sqrt(z2))
    print("real-Z background, response sigma", sig, "Z_ent", round(np.sqrt(z2), 3), flush=True)
json.dump(out, open("outputs/ent/ent_realZ.json", "w"), indent=1)
