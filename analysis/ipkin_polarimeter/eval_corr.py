"""Event-level closure of the posterior polarimeter against the exact h.

Compares, on the same events, the posterior means of posterior_h.py with the learned
tauspin regressor (hpred_HU.npz: h_pred, hh_pred) and the MMC-neutrino hybrid
polarimeter of build_dataset (spin_hhyb_*).  Reported per reco decay class of each
side: Pearson correlation of each h component with the exact one, and for the pair the
correlation of the transverse product T = h-_n h+_n + h-_r h+_r and of the CP-odd
A = h-_n h+_r - h-_r h+_n with their exact values.
"""
import argparse
import json

import numpy as np

CLS = {0: "pi", 1: "rho", 2: "a1(1p)", 3: "a1(3p)"}
RMODE_TO_CLS = np.array([0, 1, 2, 3, -1])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def pair_stats(hh):
    return hh[:, 0, 0] + hh[:, 1, 1], hh[:, 0, 1] - hh[:, 1, 0], hh[:, 2, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", nargs="+", required=True, help="tag=path of posterior_h outputs")
    ap.add_argument("--dataset", default="/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz")
    ap.add_argument("--hpred", default="/home/rbaba/dihiggs-spin-20261006/runs/ent/hpred_HU.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = np.load(args.dataset, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    uid = ds["uid"]
    est = {}
    for item in args.post:
        tag, path = item.split("=")
        z = np.load(path)
        idx = np.searchsorted(uid, z["uid"])
        assert np.array_equal(uid[idx], z["uid"])
        est[tag] = (idx, z["Eh"], z["Ehh"])
    idx = est[next(iter(est))][0]
    for tag, (i, _, _) in est.items():
        assert np.array_equal(i, idx), "posterior files cover different events"
    hx = ds["h_exact"][idx]
    hxx = hx[:, 0, :, None] * hx[:, 1, None, :]
    rm = np.stack([ds[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)[idx].reshape(-1, 2, 5)
    cls = RMODE_TO_CLS[rm.argmax(-1)]
    hyb = np.stack([np.stack([ds[f"spin_hhyb_{c}_{q}"] for c in "nrk"], 1) for q in (0, 1)], 1)[idx]
    hyb = np.where(cls[..., None] >= 0, hyb, np.nan)
    cands = {"tauspin": (hp["h_pred"][idx], hp["hh_pred"][idx]),
             "mmc_hybrid": (hyb, hyb[:, 0, :, None] * hyb[:, 1, None, :])}
    cands.update({t: (e[1], e[2]) for t, e in est.items()})
    res = {"n_events": int(len(idx)), "single": {}, "pair": {}}
    for name, (h, hh) in cands.items():
        r = {}
        for c, cn in CLS.items():
            m = cls == c
            r[cn] = {comp: corr(h[..., j][m], hx[..., j][m]) for j, comp in enumerate("nrk")}
            r[cn]["n_sides"] = int(m.sum())
        res["single"][name] = r
        T, A, L = pair_stats(hh)
        Tx, Ax, Lx = pair_stats(hxx)
        both = (cls >= 0).all(1)
        rp = {"all": {"T": corr(T[both], Tx[both]), "A": corr(A[both], Ax[both]), "L": corr(L[both], Lx[both]),
                      "n": int(both.sum())}}
        for a in range(4):
            for b in range(a, 4):
                m = both & (((cls[:, 0] == a) & (cls[:, 1] == b)) | ((cls[:, 0] == b) & (cls[:, 1] == a)))
                rp[f"{CLS[a]}-{CLS[b]}"] = {"T": corr(T[m], Tx[m]), "A": corr(A[m], Ax[m]), "n": int(m.sum())}
        res["pair"][name] = rp
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{'pair T corr':<22}" + "".join(f"{k:>14}" for k in cands))
    for key in res["pair"]["tauspin"]:
        print(f"{key:<22}" + "".join(f"{(res['pair'][k][key]['T'] or float('nan')):>14.3f}" for k in cands)
              + f"   n={res['pair']['tauspin'][key]['n']}")
    print(f"{'pair A corr (all)':<22}" + "".join(f"{(res['pair'][k]['all']['A'] or float('nan')):>14.3f}" for k in cands))
    print(f"{'pair L corr (all)':<22}" + "".join(f"{(res['pair'][k]['all']['L'] or float('nan')):>14.3f}" for k in cands))
    for cn in CLS.values():
        print(f"single {cn:<15}" + "".join(
            f"{'/'.join(f'{(res['single'][k][cn][c] or float('nan')):.2f}' for c in 'nrk'):>14}" for k in cands))


if __name__ == "__main__":
    main()
