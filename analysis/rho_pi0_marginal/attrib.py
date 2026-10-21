"""AUC attribution by spin-correlation component: transverse score E[h-_n h+_n + h-_r h+_r] vs longitudinal E[h-_k h+_k],
for the classical posteriors (own pair basis, posterior mean of the product) and the learner (product of h_pred), by truth pair."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas_classical_posterior"))
from evaluate import GEN, wauc, pair_masks
out = {}
P = {k: np.load(f"out/val_{k}/posterior.npz") for k in ("fixed_invE", "marg_invE", "truth_train")}
ref = P["marg_invE"]; y = ref["labels"]; w = ref["weights"].astype(float); sel = pair_masks(ref["modes"])
t = np.load(GEN / "targets" / "validation.npz"); v = t["h_valid"].astype(bool); hx = t["h"][v].astype(float)
rows = {k: (d["ip3_sv_mass_Ehh"][:, [0, 1, 2], [0, 1, 2]]) for k, d in P.items()}
for arm in ("full22_s42", "base_s43"):
    h = np.load(GEN / "p11" / arm / "validation_predictions.npz")["h_pred"].astype(float)
    rows[f"learner {arm}"] = h[:, 0] * h[:, 1]
rows["exact h"] = hx[:, 0] * hx[:, 1]
for k, d in rows.items():
    d = np.nan_to_num(d)
    out[k] = {p: {"transverse n+r": wauc(d[m, 0] + d[m, 1], y[m], w[m]), "longitudinal -k": wauc(-d[m, 2], y[m], w[m])}
              for p, m in sel.items() if p in ("overall", "rho x rho", "pi x rho", "pi x pi", "rho x 3pi")}
Path("out/attrib.json").write_text(json.dumps(out, indent=1))
for k, o in out.items():
    print("%-22s" % k, "  ".join("%s T %.4f L %.4f" % (p, x["transverse n+r"], x["longitudinal -k"]) for p, x in o.items()))
