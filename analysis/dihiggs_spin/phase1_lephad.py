"""Phase 1, tau_lep tau_had: only the hadronic side carries a usable polarimeter,
so the spin information is the single-tau polarisation of that side.

Each validation event contributes both of its sides as single-tau samples.  The
spin hypotheses on the measured side are (other side unpolarised and
uncorrelated): H / fakes / single H: 1 (P = 0), Z: 1 + P_Z h_k, W: 1 - h_k.
The h_pred of a side and its reco mode are the classifier inputs.  The
composition is arXiv:2209.10910 Table 5 (SLT and LTT columns); "combined fakes"
are unpolarised (mostly ttbar with a fake tau_had).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from phase1_reweight import DATA, P_TAU_Z_GEN, P_TAU_Z_PHYS, weighted_auc
from phase1_significance import LUMI_SCALE, asimov_z, fisher_z

OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
HYPS = ("H", "Z", "W")
ARMS = {"exact": None, "reco_noIPSV": "gen3pi_base_s43.npz",
        "reco_IPSV": "gen3pi_full22_s42.npz", "reco_idealIP": "gen3pi_idealip22_s42.npz"}

TABLES = {
    "SLT": {"ggF HH": (0.77, "H", 1.18, None), "VBF HH": (0.0075, "H", 1.19, None),
            "ttbar": (0.44, "W", 1.18, 0.10), "single top": (1.2, "W", 1.18, 0.10),
            "Z+HF": (1.55, "Z", 1.18, 0.10), "combined fakes": (1.09, "H", 1.18, 0.20),
            "single H": (1.10, "H", 1.15, 0.15), "other": (0.48, "Z", 1.18, 0.20)},
    "LTT": {"ggF HH": (0.25, "H", 1.18, None), "VBF HH": (0.00455, "H", 1.19, None),
            "ttbar": (1.76, "W", 1.18, 0.10), "single top": (0.61, "W", 1.18, 0.10),
            "Z+HF": (1.7, "Z", 1.18, 0.10), "combined fakes": (0.8, "H", 1.18, 0.20),
            "single H": (0.4, "H", 1.15, 0.15), "other": (0.33, "Z", 1.18, 0.20)},
}
SIGNALS = ("ggF HH", "VBF HH")


def side_samples(h, labels):
    """Stack the two sides; weights relative to the pooled generator density."""
    hm, hp = h[:, 0], h[:, 1]
    kk = hm[:, 2] * hp[:, 2]
    rho_h = 1 + hm[:, 0] * hp[:, 0] + hm[:, 1] * hp[:, 1] - kk
    zgen = 1 + P_TAU_Z_GEN * (hm[:, 2] + hp[:, 2]) + kk
    base = 0.5 * (rho_h + zgen)
    w = {}
    for X in HYPS:
        cols = []
        for s in (0, 1):
            k = h[:, s, 2]
            rho = {"H": np.ones_like(k), "Z": 1 + P_TAU_Z_PHYS * k, "W": 1 - k}[X]
            cols.append(rho / base)
        w[X] = np.concatenate(cols)
        w[X] /= w[X].mean()
    return w


def main(nbins=20):
    lad = np.load(DATA / "ladder_validation.npz")
    ref = np.load(DATA / "gen3pi_full22_s42.npz")
    h = ref["h"].astype(np.float64)
    gidx = ref["global_indices"]
    fold = np.concatenate([gidx % 2, gidx % 2]).astype(int)
    modes = np.concatenate([lad["reco_h_mode"][:, 0], lad["reco_h_mode"][:, 1]]).astype(int)
    w = side_samples(h, ref["labels"])
    res = {"arms": {}}
    for arm, fname in ARMS.items():
        if fname is None:
            k = np.concatenate([h[:, 0, 2], h[:, 1, 2]])
            dens = np.stack([np.ones_like(k), 1 + P_TAU_Z_PHYS * k, 1 - k], 1)
            probs = dens / dens.sum(1, keepdims=True)
        else:
            hpred = np.load(DATA / fname)["h_pred"].astype(np.float64)
            hs = np.concatenate([hpred[:, 0], hpred[:, 1]])
            X = np.concatenate([hs] + [(modes[:, None] == m).astype(float) for m in (0, 1, 3)], 1)
            probs = np.zeros((len(X), 3))
            for f in (0, 1):
                tr, te = fold != f, fold == f
                Xs = np.concatenate([X[tr]] * 3)
                ys = np.repeat(np.arange(3), tr.sum())
                ws = np.concatenate([w[c][tr] / w[c][tr].sum() for c in HYPS]) * tr.sum()
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                     min_samples_leaf=200, early_stopping=True,
                                                     validation_fraction=0.15, random_state=0)
                clf.fit(Xs, ys, sample_weight=ws)
                probs[te] = clf.predict_proba(X[te])
        arm_res = {"auc_H_vs_W": weighted_auc(np.log(probs[:, 0] / probs[:, 2]), w["H"], w["W"]),
                   "auc_H_vs_Z": weighted_auc(np.log(probs[:, 0] / probs[:, 1]), w["H"], w["Z"])}
        for cat, table in TABLES.items():
            s = sum(v[0] * v[2] for kname, v in table.items() if kname in SIGNALS) * LUMI_SCALE
            bkg = {kname: (v[0] * v[2] * LUMI_SCALE, v[1], v[3]) for kname, v in table.items() if kname not in SIGNALS}
            byh = {X: sum(y for y, hh, _ in bkg.values() if hh == X) for X in HYPS}
            bt = sum(byh.values())
            p = probs
            d = np.log(p[:, 0] + 1e-12) - np.log(sum(byh[X] / bt * p[:, i] for i, X in enumerate(HYPS)) + 1e-12)
            wb = sum(byh[X] / bt * w[X] for X in HYPS)
            o = np.argsort(d)
            cb = np.cumsum(wb[o]) / wb.sum()
            cut = np.searchsorted(cb, np.linspace(0, 1, nbins + 1)[1:-1])
            bin_of = np.zeros(len(d), int)
            bin_of[o] = np.searchsorted(cut, np.arange(len(d)), side="right")
            T = {X: np.bincount(bin_of, weights=w[X], minlength=nbins) / w[X].sum() for X in HYPS}
            s_b = s * T["H"]
            b_proc = {kn: y * T[hh] for kn, (y, hh, _) in bkg.items()}
            uncs = {kn: u for kn, (_, _, u) in bkg.items()}
            one = {kn: np.array([y]) for kn, (y, _, _) in bkg.items()}
            r = {"Z_stat_1bin": asimov_z([s], [bt]), "Z_stat_spin": asimov_z(s_b, sum(b_proc.values())),
                 "Z_syst_1bin": fisher_z(np.array([s]), one, uncs), "Z_syst_spin": fisher_z(s_b, b_proc, uncs)}
            r["R_stat"] = r["Z_stat_spin"] / r["Z_stat_1bin"]
            r["R_syst"] = r["Z_syst_spin"] / r["Z_syst_1bin"]
            arm_res[cat] = r
        res["arms"][arm] = arm_res
        print(arm, json.dumps(arm_res, default=lambda x: round(x, 4))[:600], flush=True)
    (OUT / "significance_lephad.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
