"""Diagnostics: oracle neutrinos through the same polarimeter, and the posterior tau energy."""
import sys
import numpy as np
from posterior_h import load_reco, ordered, RMODE_TO_CLS
from polarimeter_pythia import polarimeter_nrk

d = load_reco("/home/rbaba/dihiggs-spin-20261006/runs/ent/reco")
o = ordered(d)
n = 4096
cls = RMODE_TO_CLS[o["rmode"][:n]]
nu = o["tau_nu_p4"][:n]
h_or = polarimeter_nrk(o["ch"][:n], o["ch_q"][:n], o["pi0"][:n], nu, o["vis"][:n] + nu, cls)
hx = d["h_exact"][:n]
for c, nm in enumerate(["pi", "rho", "a1(1p)", "a1(3p)"]):
    m = cls[:, 0] == c
    ok = m & np.isfinite(h_or[:, 0, 0]) & np.isfinite(hx[:, 0, 0])
    print("oracle-nu reco-vis", nm, [round(np.corrcoef(h_or[ok, 0, j], hx[ok, 0, j])[0, 1], 3) for j in range(3)])
for path in sys.argv[1:]:
    z = np.load(path)
    Et = o["tau_p4"][:n, :, 3]
    m = cls[:, 0] == 0
    print(path, "pi side0: corr(Etau post, true)", round(np.corrcoef(z["Etau"][m, 0], Et[m, 0])[0, 1], 3),
          "median rel err", round(np.median(np.abs(z["Etau"][m, 0] / Et[m, 0] - 1)), 3),
          "mpair median/16/84", np.percentile(z["mpair"], [50, 16, 84]).round(1).tolist())
    xt = o["vis"][:n, 0, 3] / Et[:, 0]
    xp = o["vis"][:n, 0, 3] / z["Etau"][:, 0]
    print("   corr x", round(np.corrcoef(xp[m], xt[m])[0, 1], 3))
