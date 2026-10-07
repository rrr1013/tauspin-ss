"""Entanglement significance versus integrated luminosity (ATLAS alone, mass-binned
categories, B +-2 %, response nuisance 10 %), for each reconstruction level."""
import json
import numpy as np
from ent_closure_syst import z_ent_shape
from ent_final import PT_RANGE, binning
from ent_hllhc import CATS, LUMI

z = np.load("outputs/ent/ent_final_G.npz")
acc = json.load(open("outputs/ent/mass_acc.json"))
ptt = z["ptt"]
W = {k: z[f"w_{k}"] for k in ("H", "W13", "Z", "U")}
out = {}
for name in ("exact", "tauspin", "tauspin_noIPSV", "textbook"):
    G = z[name]
    idx, nb = binning(G)
    b = idx(G)
    Ts = {}
    for c, (lo, hi) in PT_RANGE.items():
        m = (ptt >= lo) & (ptt < hi)
        Ts[c] = {k: np.bincount(b[m], weights=W[k][m], minlength=nb) / W[k][m].sum() for k in W}
    rows = []
    for L in (300, 450, 1000, 2000, 3000, 6000):
        sc = L / 3000
        z2 = 0.0
        for c, (s_, zz, f, t, o) in CATS.items():
            for i in range(len(acc["S"])):
                S = s_ * LUMI * 1.13 * acc["S"][i] * sc
                B = (zz * acc["Z"][i] + f * acc["F"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15 * sc
                z2 += z_ent_shape(Ts[c], S, B, 0.02, 0.1) ** 2
        rows.append((L, float(np.sqrt(z2))))
    out[name] = rows
    print(name, rows, flush=True)
json.dump(out, open("outputs/ent/ent_lumi.json", "w"), indent=1)
