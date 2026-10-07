"""Template MC-statistics check of the final projection.

The H and p = 1/3 templates are reweightings of the same finite spin-flat sample, so
sampling noise in T_H - T_W13 biases the Asimov q upwards.  Recompute the final
per-category significance with templates built from each half of the sample
(alternate events; the cross-fitted G is kept) and from bootstrap replicas.  A
half-sample result close to the full one means the bias is small; the spread of the
bootstrap replicas is the template-statistics uncertainty.
"""
import json

import numpy as np

from ent_closure_syst import z_ent_shape
from ent_final import PT_RANGE, binning
from ent_hllhc import CATS, LUMI


def zfinal(G, W, ptt, acc, sel, sig=0.1):
    idx, nb = binning(G[sel])
    b = idx(G)
    z2 = 0.0
    for c, (s_, zz, f, t, o) in CATS.items():
        lo, hi = PT_RANGE[c]
        m = sel & (ptt >= lo) & (ptt < hi)
        T = {k: np.bincount(b[m], weights=W[k][m], minlength=nb) / W[k][m].sum() for k in ("H", "W13", "Z", "U")}
        for i in range(len(acc["S"])):
            S = s_ * LUMI * 1.13 * acc["S"][i]
            B = (zz * acc["Z"][i] + f * acc["F"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15
            z2 += z_ent_shape(T, S, B, 0.02, sig) ** 2
    return float(np.sqrt(z2))


def main():
    z = np.load("outputs/ent/ent_final_G.npz")
    acc = json.load(open("outputs/ent/mass_acc.json"))
    ptt = z["ptt"]
    n = len(ptt)
    out = {}
    rng = np.random.default_rng(1)
    for name in ("tauspin", "textbook"):
        G = z[name]
        Wb = {k: z[f"w_{k}"] for k in ("H", "W13", "Z", "U")}
        full = zfinal(G, Wb, ptt, acc, np.ones(n, bool))
        halves = [zfinal(G, Wb, ptt, acc, (np.arange(n) % 2) == j) for j in (0, 1)]
        boots = []
        for r in range(6):
            c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            Wr = {k: v * c for k, v in Wb.items()}
            boots.append(zfinal(G, Wr, ptt, acc, c > 0))
        out[name] = {"full": full, "half_samples": halves, "bootstrap": boots,
                     "bootstrap_sd": float(np.std(boots, ddof=1))}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open("outputs/ent/ent_mcstat.json", "w"), indent=1)


if __name__ == "__main__":
    main()
