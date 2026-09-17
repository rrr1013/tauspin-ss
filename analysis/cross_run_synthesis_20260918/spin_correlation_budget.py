"""Cross-run synthesis (2026-09-18): where the H/Z spin information lives, and
how much of it survives the reconstruction of the polarimetric vector.

Read-only. Consumes the frozen artefact written by the 2026-09-08 residual-shortcut
run; trains nothing and writes no model.

Three measurements:
  1. the truth-level tau-pair spin-correlation matrix C_ij ~ <h^-_i h^+_j>, per
     decay-mode pair, in the common (n, r, k) basis;
  2. a closure test: degrade each component of the truth h to the component-wise
     correlation actually achieved by the full-reco point-h regression
     (n 0.616 / r 0.577 / k 0.759) and check that the observed H/Z separation is
     recovered;
  3. the sensitivity of the transverse spin correlation to the transverse
     polarimeter quality rho_T, which is the quantity an impact-parameter or
     decay-vertex measurement would improve.

Cohort caveat: the frozen file is the pre-pi0-fix hybrid cohort, so it contains
pi and 3pi sides only (no rho).  That does not affect the structural conclusions,
which are the same in every mode pair present.
"""

import json
import pathlib

import numpy as np

SCORES = pathlib.Path(
    "/Users/ryunosuke/Projects/tauspin-residual-data/residual-shortcut-scores.npz"
)
AXES = ("n", "r", "k")
# component-wise correlation(prediction, truth) of the full-reco point-h regression,
# from tauspin-full-reco-information-ladder-20260917 (tau- side; tau+ agrees to 0.005)
FULL_RECO_RHO = (0.616, 0.577, 0.759)


def rankdata(a):
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(score, label):
    r = rankdata(score)
    n1 = int((label == 1).sum())
    n0 = int((label == 0).sum())
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def correlation_matrix(hm, hp):
    """Return the naive and the acceptance-corrected estimates of C_ij.

    For an unpolarised tau the polarimetric vector of tau -> pi nu is isotropic,
    so <h^-_i h^+_j> = C_ij / 9 and the naive estimator is 9 <h^-_i h^+_j>.

    The analysis requires both visible taus to pass pT > 18 GeV, which selects on
    the visible energy fraction and therefore on h_k alone.  Modelling that as a
    factorised acceptance a(h^-_k) b(h^+_k), the azimuthal cross terms integrate
    away and the surviving relation is

        <h^-_i h^+_i>_A = C_ii * <(h^-_i)^2>_a * <(h^+_i)^2>_b,

    so dividing by the measured second moments corrects it.  The naive estimator
    is biased in opposite directions for the two sectors: the pT cut depletes
    |h_k| and therefore dilutes C_kk, while pushing weight into the n-r plane and
    therefore inflating C_nn and C_rr.  For the multi-body modes h is not
    isotropic even before acceptance, so the corrected values there are still
    compressed; read the structure, not the last digit.
    """
    n = len(hm)
    raw = np.einsum("ni,nj->ij", hm, hp) / n
    var_m = (hm ** 2).mean(axis=0)
    var_p = (hp ** 2).mean(axis=0)
    return 9.0 * raw, raw / np.outer(var_m, var_p)


def degrade(h, rho, rng):
    """Replace each component by rho * truth + sqrt(1 - rho^2) * noise, keeping
    the per-component variance, i.e. a reconstruction of known quality and no bias."""
    out = np.empty_like(h)
    for side in range(2):
        for i in range(3):
            x = h[:, side, i]
            out[:, side, i] = rho[i] * x + np.sqrt(max(0.0, 1 - rho[i] ** 2)) * rng.normal(
                0.0, x.std(), len(x)
            )
    return out


def transverse(h):
    return h[:, 0, 0] * h[:, 1, 0] + h[:, 0, 1] * h[:, 1, 1]


def longitudinal(h):
    return h[:, 0, 2] * h[:, 1, 2]


def main():
    d = np.load(SCORES)
    h, y, modes = d["truth_h"], d["labels"], d["modes"]
    rng = np.random.default_rng(20260918)
    out = {"n_events": int(len(y)), "n_H": int((y == 1).sum()), "n_Z": int((y == 0).sum())}

    cohorts = {
        "all": np.ones(len(y), bool),
        "pi_pi": (modes[:, 0] == 0) & (modes[:, 1] == 0),
        "3pi_3pi": (modes[:, 0] == 3) & (modes[:, 1] == 3),
        "pi_3pi": ((modes[:, 0] == 0) & (modes[:, 1] == 3))
        | ((modes[:, 0] == 3) & (modes[:, 1] == 0)),
    }

    out["spin_correlation"] = {}
    for tag, mask in cohorts.items():
        entry = {"n": int(mask.sum())}
        for lab, name in ((1, "H"), (0, "Z")):
            m = mask & (y == lab)
            naive, C = correlation_matrix(h[m, 0, :], h[m, 1, :])
            entry[name] = {
                "diag_naive_9moment": {
                    AXES[i]: round(float(naive[i, i]), 4) for i in range(3)
                },
                "diag_acceptance_corrected": {
                    AXES[i]: round(float(C[i, i]), 4) for i in range(3)
                },
                "max_abs_offdiag_corrected": round(
                    float(np.abs(C - np.diag(np.diag(C))).max()), 4
                ),
            }
        stat = transverse(h[mask]) - longitudinal(h[mask])
        entry["auc"] = {
            "nn_plus_rr_minus_kk": round(auc(stat, y[mask]), 4),
            "nn_plus_rr": round(auc(transverse(h[mask]), y[mask]), 4),
            "kk": round(auc(longitudinal(h[mask]), y[mask]), 4),
        }
        out["spin_correlation"][tag] = entry

    # 2. closure test
    def repeated_auc(rho, trials=30):
        vals = [
            auc(transverse(hh := degrade(h, rho, rng)) - longitudinal(hh), y)
            for _ in range(trials)
        ]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    out["closure"] = {
        "truth_h": round(auc(transverse(h) - longitudinal(h), y), 4),
        "full_reco_measured": dict(
            zip(("mean", "std"), repeated_auc(FULL_RECO_RHO))
        ),
        "longitudinal_only": dict(zip(("mean", "std"), repeated_auc((0.0, 0.0, 1.0)))),
        "transverse_only": dict(zip(("mean", "std"), repeated_auc((1.0, 1.0, 0.0)))),
        "uniform_0p65": dict(zip(("mean", "std"), repeated_auc((0.65, 0.65, 0.65)))),
        "reference_note": (
            "measured full-reco point-h fixed-readout AUC is 0.617-0.619 on the "
            "larger rho-inclusive cohort of the 2026-09-17 ladder run"
        ),
    }

    # 3. sensitivity of the transverse spin correlation to rho_T
    rows = []
    for rho_t in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        sig = []
        for _ in range(30):
            T = transverse(degrade(h, (rho_t, rho_t, FULL_RECO_RHO[2]), rng))
            sig.append(abs(T[y == 1].mean() - T[y == 0].mean()) / T.std())
        T = transverse(degrade(h, (rho_t, rho_t, FULL_RECO_RHO[2]), rng))
        rows.append(
            {
                "rho_T": rho_t,
                "mean_T_H": round(float(T[y == 1].mean()), 5),
                "mean_T_Z": round(float(T[y == 0].mean()), 5),
                "significance_per_event": round(float(np.mean(sig)), 5),
            }
        )
    base = next(r for r in rows if r["rho_T"] == 0.60)["significance_per_event"]
    for r in rows:
        r["ratio_to_current"] = round(r["significance_per_event"] / base, 3)
        r["equivalent_luminosity"] = round((r["significance_per_event"] / base) ** 2, 2)
    out["transverse_sensitivity"] = {
        "rho_k_held_at": FULL_RECO_RHO[2],
        "rows": rows,
        "note": (
            "the observable is a product h^-_i h^+_i, so the H-Z difference is "
            "diluted by rho_T^2 while the spread is unchanged; the equivalent "
            "luminosity column is (significance ratio)^2"
        ),
    }

    dest = pathlib.Path(__file__).with_name("spin_correlation_budget.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
