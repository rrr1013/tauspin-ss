"""Split the single-side closure by whether the reco decay class equals the true one.

Per reco class: share of sides whose true class (h_cls of the generator record) differs, and
corr(h component, exact) for learned / classical / true-neutrino ceiling on the correctly
and wrongly classified sides.  Also the pair statistic T restricted to events with both
sides correctly classified."""
import json
import sys

import numpy as np

O = sys.argv[1]
d = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz", allow_pickle=True)
hp = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/hpred_HU.npz", allow_pickle=True)
post = {v: np.load(f"{O}/post_{v}.npz") for v in ("ip", "ip_m125")}
orc = np.load(f"{O}/oracle_single.npz")["h_oracle"]
# oracle_single follows reco-charge order; dataset h_exact is true-charge order: equal for true taus
rm = np.stack([np.stack([d[f"spin_rmode{m}_{q}"] for m in range(5)], 1).argmax(1) for q in (0, 1)], 1)
cls = np.array([0, 1, 2, 3, -1])[rm]
tcls = d["h_cls"]
hx = d["h_exact"]
H = {"tauspin": hp["h_pred"], "ipkin": post["ip"]["Eh"], "ipkin_m125": post["ip_m125"]["Eh"], "oracle": orc}
names = ["pi", "rho", "a1(1p)", "a1(3p)"]


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 50 else None


res = {}
for c, nm in enumerate(names):
    m = cls == c
    good, bad = m & (tcls == c), m & (tcls != c)
    r = {"n": int(m.sum()), "frac_migrated": float(bad.sum() / m.sum()),
         "true_of_migrated": {str(t): int(((tcls == t) & bad).sum()) for t in (-1, 0, 1, 2, 3)}}
    for k, h in H.items():
        r[k] = {sel: [corr(h[..., j][mm], hx[..., j][mm]) for j in range(3)] for sel, mm in (("all", m), ("correct", good), ("migrated", bad))}
    res[nm] = r
T = lambda a: a[:, 0, 0] * a[:, 1, 0] + a[:, 0, 1] * a[:, 1, 1]
both_ok = (cls >= 0).all(1) & (cls == tcls).all(1)
any_bad = (cls >= 0).all(1) & ~(cls == tcls).all(1)
res["pair_T"] = {}
for k, h in list(H.items()):
    hh = None
    if k == "tauspin":
        t = hp["hh_pred"][:, 0, 0] + hp["hh_pred"][:, 1, 1]
    elif k == "oracle":
        t = T(h)
    else:
        e = post["ip" if k == "ipkin" else "ip_m125"]["Ehh"]
        t = e[:, 0, 0] + e[:, 1, 1]
    res["pair_T"][k] = {"both_correct": corr(t[both_ok], T(hx)[both_ok]), "some_migrated": corr(t[any_bad], T(hx)[any_bad]),
                        "n_both_correct": int(both_ok.sum()), "n_some_migrated": int(any_bad.sum())}
json.dump(res, open(f"{O}/migration.json", "w"), indent=1)
for nm in names:
    r = res[nm]
    print(f"{nm:8s} n={r['n']:6d} migrated={r['frac_migrated']:.3f} {r['true_of_migrated']}")
    for k in H:
        print(f"   {k:11s}" + "  ".join(f"{sel}:" + "/".join("nan" if x is None else f"{x:.2f}" for x in r[k][sel]) for sel in ("correct", "migrated")))
print(json.dumps(res["pair_T"], indent=1))
