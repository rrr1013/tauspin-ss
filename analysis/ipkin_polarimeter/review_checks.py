"""Checks requested by the skeptical review.
(a) learned-polarimeter IP/SV dependence per reco mode pair: tauspin - tauspin_noIPSV (cached G of 10/07);
(b) pull of the 1-prong IP-plane likelihood at the TRUE tau direction, per true decay class,
    with the azimuth measured around the leading track as in ipmmc2.sample_tau;
(c) negative controls for the combination gain: tauspin + 15 Gaussian noise columns, tauspin + ipkin_noip."""
import json
import sys

import numpy as np

sys.path.insert(0, "../ip_tau_reco")
import ent_robust as er  # noqa: E402
from add_levels import post_feats  # noqa: E402
from breakdown import wcorr  # noqa: E402
from posterior_h import build_sides, load_reco, ordered  # noqa: E402

O = sys.argv[1]
out = {}
c = dict(np.load(f"{O}/ent_robust_G.npz", allow_pickle=True))
d = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz", allow_pickle=True)
T, w = c["T"][:, 0], c["w"]
CLS = ["pi", "rho", "a1(1p)", "a1(3p)", "3pXn"]
rm = np.stack([np.stack([d[f"spin_rmode{m}_{q}"] for m in range(5)], 1).argmax(1) for q in (0, 1)], 1)
lo, hi = np.sort(rm, 1).T
out["noipsv"] = {}
for a in range(5):
    for b in range(a, 5):
        m = (lo == a) & (hi == b)
        if m.sum() < 500:
            continue
        out["noipsv"][f"{CLS[a]}-{CLS[b]}"] = {l: wcorr(c[f"G_{l}"][m, 0], T[m], w[m])
                                               for l in ("tauspin", "tauspin_noIPSV", "ipkin", "ipkin_noip")}
# (b) pulls
r = load_reco("/home/rbaba/dihiggs-spin-20261006/runs/ent/reco")
o = ordered(r)
S = build_sides(o)
tcls = d["h_cls"]
out["pull"] = {}
for s in (0, 1):
    t = o["tau_p4"][:, s, :3]
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    p = S[s]["trk"]
    tperp = t - (t * p).sum(-1, keepdims=True) * p
    tperp /= np.linalg.norm(tperp, axis=-1, keepdims=True)
    dpsi = np.arctan2((tperp * S[s]["e2"]).sum(-1), (tperp * S[s]["e1"]).sum(-1))
    pull = dpsi / S[s]["sig_psi"]
    for k, nm in enumerate(["pi", "rho", "a1(1p)"]):
        m = (tcls[:, s] == k) & (S[s]["sig_psi"] <= 1.5)
        q = out["pull"].setdefault(nm, {"pulls": []})
        q["pulls"].append(pull[m])
for nm, q in out["pull"].items():
    x = np.concatenate(q.pop("pulls"))
    q.update(n=int(len(x)), robust_sd=float(0.7413 * np.subtract(*np.percentile(x, [75, 25]))),
             frac_abs_gt3=float((np.abs(x) > 3).mean()), median_abs=float(np.median(np.abs(x))))
# (c) negative controls
hp = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/hpred_HU.npz", allow_pickle=True)
tsp = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9)], 1)
noip = np.load(f"{O}/post_noip.npz")
rng = np.random.default_rng(7)
X = {"tauspin+noise15": np.concatenate([tsp, rng.standard_normal((len(T), 15)), er.rm(d)], 1),
     "tauspin+ipkin_noip": np.concatenate([tsp, post_feats(noip), er.rm(d)], 1)}
fold = np.arange(len(T)) % 2
for lev, F in X.items():
    c[f"G_{lev}"] = er.regress(F, c["T"], fold, None, w)[0]
np.savez_compressed(f"{O}/ent_robust_G_controls.npz", **c)
er.load_prep(f"{O}/ent_robust_G_controls.npz")
out["controls"] = {}
for lev in list(X) + ["tauspin"]:
    _, res = er.job((lev, "nominal", {}))
    out["controls"][lev] = {"Z": res["Z"], "corr_gT": float(np.corrcoef(er.PREP["P"]["levels"][lev]["G"][:, 0], T)[0, 1])}
json.dump(out, open(f"{O}/review_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
