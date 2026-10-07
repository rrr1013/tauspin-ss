"""Training sample for the h regressor: Pythia 8 internal H(125) + jet and
Z + jet with the Z mass set to 125 GeV (Z couplings and spin structure kept),
both with tau tau decays.  Equal masses keep m_tautau from carrying the H/Z
spin prior into the regressed h (the design of the tauspin H/Z samples).
Output layout identical to pythia_shower.py (no fake candidates)."""
import argparse

import fastjet
import numpy as np
import pythia8

from pythia_shower import empty, process_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=["trainH", "trainZ", "trainHU"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    py = pythia8.Pythia("", False)
    cmds = ["Beams:eCM = 14000", "111:mayDecay = off", "TauDecays:mode = 4",
            "ParticleDecays:limitTau0 = off", "Next:numberCount = 0",
            "Random:setSeed = on", f"Random:seed = {args.seed}", "PhaseSpace:pTHatMin = 40."]
    if args.proc == "trainHU":
        # spin-flat reference for reweighting: unpolarised, uncorrelated tau decays
        cmds = [c for c in cmds if not c.startswith("TauDecays:mode")]
        cmds += ["TauDecays:mode = 3", "TauDecays:tauPolarization = 0."]
    if args.proc in ("trainH", "trainHU"):
        cmds += ["HiggsSM:gg2Hg(l:t) = on", "HiggsSM:qg2Hq(l:t) = on", "HiggsSM:qqbar2Hg(l:t) = on",
                 "25:onMode = off", "25:onIfAny = 15"]
    else:
        cmds += ["WeakBosonAndParton:qqbar2gmZg = on", "WeakBosonAndParton:qg2gmZq = on",
                 "WeakZ0:gmZmode = 2", "23:m0 = 125.0", "23:mMin = 115.0", "23:mMax = 135.0",
                 "23:onMode = off", "23:onIfAny = 15"]
    for c in cmds:
        py.readString(c)
    py.init()
    out = empty(args.n)
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    k = 0
    while k < args.n:
        if not py.next():
            continue
        out["weight"][k] = 1.0
        process_event(py.event, out, k, False, jd)
        k += 1
    out = {a: b for a, b in out.items() if not a.startswith("fk_")}
    out["sigma_pb"] = np.array(py.infoPython().sigmaGen() * 1e9)
    out["n_fail"] = np.array(0)
    np.savez_compressed(args.out, **out)
    print("stored", args.n)


if __name__ == "__main__":
    main()
