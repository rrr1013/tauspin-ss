"""Cross-check anchored on the ATLAS composition: gain from the lepton impact parameter in the
most signal-like tau_lep tau_had bins of the Run-2 analysis (arXiv:2209.10910 Table 5, SLT and
LTT columns, as phase1_lephad.py).

The lepton-IP information is (to a good approximation) independent of the event kinematics
given the lepton pT and flavour, so it can be added to the ATLAS bins by factorisation:
  * an IP-only score (gradient boosting on the lepton IP features of build_lephad.py plus the
    lepton pT and flavour as conditioning variables, tau-origin vs prompt), cross-fitted;
  * per ATLAS process class, the score template is taken from our MC in a signal-like
    kinematic region (top `--kin-frac` of HH by a kinematics-only classifier), mixing
    tau-lepton and prompt-lepton events as they occur there: ttbar = ttll + ttlt, single top =
    tWll, fakes = ttlj + ttljp + tWlj, Z+HF = zbb, single H = zh + tth, HH = hhC; "other"
    (W/Z+jets, dibosons) is given the prompt template (conservative: 50/50 as variant);
  * the bin is split by the IP score (10 bins of equal HH fraction) and the Asimov q0 with the
    group normalisations profiled is compared with the unsplit bin (R = Z_split / Z_bin), at
    139 fb^-1 x 3000/139 x 14 TeV factors as phase1_lephad.py.
Scenarios: `as_published` treats the ATLAS yields as they are (templates from leptons passing
the standard |d0/sigma| < 3/5 requirement, the conservative reading); `no_ttva` assumes the
published yields include that requirement and removes it (tau-lepton parts of each class
scale by 1/eff).
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

TABLES = {
    "SLT": {"HH": (0.77 + 0.0075, 1.18, None), "ttbar": (0.44, 1.18, 0.10), "single top": (1.2, 1.18, 0.10),
            "Z+HF": (1.55, 1.18, 0.10), "fakes": (1.09, 1.18, 0.20), "single H": (1.10, 1.15, 0.15),
            "other": (0.48, 1.18, 0.20)},
    "LTT": {"HH": (0.25 + 0.00455, 1.18, None), "ttbar": (1.76, 1.18, 0.10), "single top": (0.61, 1.18, 0.10),
            "Z+HF": (1.7, 1.18, 0.10), "fakes": (0.8, 1.18, 0.20), "single H": (0.4, 1.15, 0.15),
            "other": (0.33, 1.18, 0.20)},
}
CLASS = {"HH": ("hhC",), "ttbar": ("ttll", "ttlt"), "single top": ("tWll",), "Z+HF": ("zbb",),
         "fakes": ("ttlj", "ttljp", "tWlj"), "single H": ("zh", "tth")}
LUMI = 3000 / 139


def q0(s, B, prior):
    n = s + B.sum(0)
    def nll(th):
        b = np.maximum((B * (1 + prior[:, None] * th[:, None])).sum(0), 1e-12)
        return float((b - n * np.log(b)).sum() + 0.5 * (th ** 2).sum())
    l0 = minimize(nll, np.zeros(len(prior)), method="L-BFGS-B").fun
    return max(2 * (l0 - float((n - n * np.log(n)).sum())), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kin-frac", type=float, default=0.2)
    ap.add_argument("--lip", default="lip_")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    proc, w, uid = d["proc"], d["w"], d["uid"]
    base = (d["kin_m_bb"] < 150) & (d["kin_m_tautau"] > 60)
    fold = uid % 3
    lip = args.lip
    ipcols = [lip + s for s in ("d0sig", "absd0", "z0sig", "n", "r", "k", "sd0", "is_e")] + ["kin_pt_lep"]
    Xip = np.stack([d[c] for c in ipcols], 1)
    kin = [k for k in d.files if k.startswith("kin_") and k != "kin_mmc_valid"]
    Xk = np.stack([d[c] for c in kin], 1)
    tau_origin = d["lep_from_tau"].astype(int)
    # IP-only score: tau-origin vs prompt (unweighted per class, cross-fitted)
    s_ip = np.zeros(len(w))
    for f in range(3):
        tr = base & (fold != f)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0)
        clf.fit(Xip[tr], tau_origin[tr])
        s_ip[fold == f] = clf.predict_proba(Xip[fold == f])[:, 1]
    # kinematics-only HH classifier to define the signal-like region
    y = (proc == "hhC").astype(int)
    s_k = np.zeros(len(w))
    for f in range(3):
        tr = base & (fold != f)
        wt = np.abs(w[tr]).copy()
        wt[y[tr] == 1] *= wt[y[tr] == 0].sum() / wt[y[tr] == 1].sum()
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=100, random_state=0)
        clf.fit(Xk[tr], y[tr], sample_weight=wt)
        s_k[fold == f] = clf.predict_proba(Xk[fold == f])[:, 1]
    res = {"config": vars(args), "categories": {}}
    edges_ip = None
    for cat, tv in (("SLT", 0), ("LTT", 1)):
        m0 = base & (d["trigger"] == tv)
        thr = np.quantile(s_k[m0 & (y == 1)], 1 - args.kin_frac)
        reg = m0 & (s_k >= thr)
        sig = reg & (y == 1)
        edges_ip = np.quantile(s_ip[sig], np.linspace(0, 1, 11))
        edges_ip[0], edges_ip[-1] = -np.inf, np.inf
        ttva = d[lip + "ttva"] > 0
        info = {}
        for c, procs in CLASS.items():
            mc = reg & np.isin(proc, procs)
            ww = np.abs(w[mc])
            tau_mc = mc & (tau_origin == 1)
            info[c] = {"mc_events": int(mc.sum()), "ess": float(ww.sum() ** 2 / max((ww ** 2).sum(), 1e-30)),
                       "prompt_frac": float(np.average(1 - tau_origin[mc], weights=ww)) if mc.any() else None,
                       "ttva_eff_tau": float(np.average(ttva[tau_mc], weights=np.abs(w[tau_mc]))) if tau_mc.any() else 1.0}
        out = {"region_info": info}
        for scen in ("as_published", "no_ttva"):
            keep = ttva if scen == "as_published" else np.ones(len(w), bool)
            tmpl = {}
            for c, procs in CLASS.items():
                mc = reg & keep & np.isin(proc, procs)
                h, _ = np.histogram(s_ip[mc], edges_ip, weights=np.abs(w[mc]))
                tmpl[c] = h / max(h.sum(), 1e-30)
            mp = reg & keep & (tau_origin == 0) & ~np.isin(proc, ("hhC",))
            h, _ = np.histogram(s_ip[mp], edges_ip, weights=np.abs(w[mp]))
            tmpl["other"] = h / h.sum()
            names = list(TABLES[cat])
            yields = {}
            for c in names:
                v, k14, _ = TABLES[cat][c]
                v = v * LUMI * k14
                if scen == "no_ttva" and c != "other":
                    # published yields include the TTVA requirement: N = v / (f_prompt + f_tau eff_tau),
                    # with the fractions of the no-TTVA sample
                    pf, eff = info[c]["prompt_frac"], info[c]["ttva_eff_tau"]
                    v = v / (pf + (1 - pf) * eff)
                yields[c] = v
            s_un = np.array([yields["HH"]])
            B_un = np.array([[yields[c]] for c in names if c != "HH"])
            prior = np.array([TABLES[cat][c][2] for c in names if c != "HH"])
            z_un = np.sqrt(q0(s_un, B_un, prior))
            s_sp = yields["HH"] * tmpl["HH"]
            B_sp = np.stack([yields[c] * tmpl[c] for c in names if c != "HH"])
            z_sp = np.sqrt(q0(s_sp, B_sp, prior))
            out[scen] = {"Z_bin": float(z_un), "Z_split": float(z_sp), "R": float(z_sp / z_un), "yields": yields}
            print(cat, scen, "Z_bin", round(z_un, 3), "Z_split", round(z_sp, 3), "R", round(z_sp / z_un, 4), flush=True)
        res["categories"][cat] = out
    # reference: as-published bin without IP vs no-ttva bin with IP
    for cat in res["categories"]:
        c = res["categories"][cat]
        c["R_total_vs_published"] = c["no_ttva"]["Z_split"] / c["as_published"]["Z_bin"]
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("region_info",)} for k, v in res["categories"].items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
