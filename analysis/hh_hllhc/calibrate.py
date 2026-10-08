"""Calibrate a simplified systematic model so that the HEPData-based likelihood reproduces
the ATLAS HL-LHC bbtautau degradation from 'no systematics' to the 'baseline' scenario
(ATL-PHYS-PUB-2024-016 Table 4, 3000 fb^-1: hadhad 4.0 -> 3.1, lephad 2.3 -> 1.8,
combined 4.6 -> 3.5).

The ATLAS uncertainty breakdown (Table 3, baseline) is dominated by the single-Higgs
modelling (+0.17/-0.15 on mu, i.e. H+heavy-flavour theory), then Z+jets (0.06), jets
(0.06), fakes (0.05).  The simplified model therefore has one correlated normalisation
uncertainty on single H (scanned), and fixed smaller ones on Z+HF acceptance (10%),
fakes (10%, Run-2 systematic halved), other (15%) and top (5%) on top of the free top and
Z+HF normalisation factors.  Signal-cross-section uncertainty does not enter a discovery
significance and is left out.
"""
import json
import os
from pathlib import Path

import numpy as np

from projection import DEFAULT, run

OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs")) / "calib"
TARGET = {"hadhad": 3.1 / 4.0, "lephad": 1.8 / 2.3, "combined": 3.5 / 4.6}
FIXED = {"zhf": 0.10, "fake": 0.10, "fake_mj": 0.10, "fake_tt": 0.10, "other": 0.15, "top": 0.05}


def lephad(z):
    return float(np.hypot(z["slt"], z["ltt"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stat, _ = run(dict(DEFAULT))
    rows = []
    for s_h in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        cfg = dict(DEFAULT)
        cfg["norm_sys"] = {**FIXED, "single_h": s_h}
        z, _ = run(cfg)
        r = {"single_h": s_h, **z,
             "ratio_hadhad": z["hadhad"] / stat["hadhad"],
             "ratio_lephad": lephad(z) / lephad(stat),
             "ratio_combined": z["combined"] / stat["combined"]}
        rows.append(r)
        print(json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
    json.dump({"stat_only": stat, "target_ratio": TARGET, "fixed": FIXED, "scan": rows},
              open(OUT / "calibration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
