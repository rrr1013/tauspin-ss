"""Where do the learned and the classical statistics differ?

On the cached cross-fitted statistics (ent_robust_G.npz of this run), per reco decay-mode
pair: Pearson correlation of g_T with the exact target T = h-_n h+_n + h-_r h+_r and the
per-event Fisher information of a small transverse correlation,
    I_T = E_flat[g_T^2] (the information on c in rho = 1 + c T, at c = 0),
which is what the entanglement and CP fits use.  Also as a function of the impact-parameter
magnitude of the 1-prong leading-track impact vector projected on the plane transverse to the
visible tau (|b| in the local (n, r) basis, um), and the event-level correlation between
the learned and the classical statistics (complementarity).
"""
import argparse
import json

import numpy as np

CLS = ["pi", "rho", "a1(1p)", "a1(3p)", "3pXn"]
LEVELS = ["exact", "tauspin", "ipkin", "ipkin_m125", "tauspin+ipkin", "tauspin+ipkin_m125", "ipkin_noip", "phicp", "phicp_zmf",
          "textbook_pair"]


def wcorr(a, b, w):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data", default="/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.cache, allow_pickle=True)
    d = np.load(args.data, allow_pickle=True)
    T, w = z["T"][:, 0], z["w"]
    rm = np.stack([np.stack([d[f"spin_rmode{m}_{q}"] for m in range(5)], 1).argmax(1) for q in (0, 1)], 1)
    lo, hi = np.sort(rm, 1).T
    res = {"pairs": {}, "ipsig": {}, "levels": LEVELS}
    G = {l: z[f"G_{l}"][:, 0] for l in LEVELS}
    for a in range(5):
        for b in range(a, 5):
            m = (lo == a) & (hi == b)
            if m.sum() < 500:
                continue
            key = f"{CLS[a]}-{CLS[b]}"
            res["pairs"][key] = {"n": int(m.sum()), "wfrac": float(w[m].sum() / w.sum())}
            for l in LEVELS:
                g = G[l][m]
                res["pairs"][key][l] = {"corr": wcorr(g, T[m], w[m]) if g.std() > 0 else None,
                                        "info": float(np.average(g ** 2, weights=w[m]))}
            res["pairs"][key]["corr_tauspin_ipkin"] = wcorr(G["tauspin"][m], G["ipkin"][m], w[m])
    res["all"] = {l: {"corr": wcorr(G[l], T, w), "info": float(np.average(G[l] ** 2, weights=w))} for l in LEVELS}
    res["all"]["corr_tauspin_ipkin"] = wcorr(G["tauspin"], G["ipkin"], w)
    # 1-prong impact vector projected on the plane transverse to the visible tau (local n, r), um
    sig = np.stack([np.hypot(d[f"spin_ip_n_{q}"], d[f"spin_ip_r_{q}"]) for q in (0, 1)], 1)  # um
    one = rm <= 2
    smin = np.where(one.all(1), sig.min(1), np.where(one[:, 0], sig[:, 0], np.where(one[:, 1], sig[:, 1], np.nan)))
    edges = [0, 10, 20, 40, 80, 1e9]
    for i in range(len(edges) - 1):
        m = (smin >= edges[i]) & (smin < edges[i + 1])
        key = f"{edges[i]:g}-{edges[i + 1]:g}um"
        res["ipsig"][key] = {"n": int(m.sum())}
        for l in LEVELS:
            res["ipsig"][key][l] = wcorr(G[l][m], T[m], w[m]) if G[l][m].std() > 0 else None
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{'pair':<16}{'n':>7}" + "".join(f"{l[:13]:>14}" for l in LEVELS) + f"{'r(ts,ik)':>10}")
    for k, v in list(res["pairs"].items()) + [("all", {**res["all"], "n": len(T)})]:
        print(f"{k:<16}{v['n']:>7}" + "".join(f"{(v[l]['corr'] or 0):>14.3f}" for l in LEVELS)
              + f"{v['corr_tauspin_ipkin']:>10.3f}")
    print("min 1-prong |IP_T| bins (corr g_T)")
    for k, v in res["ipsig"].items():
        print(f"{k:<16}{v['n']:>7}" + "".join(f"{(v[l] or 0):>14.3f}" for l in LEVELS))


if __name__ == "__main__":
    main()
