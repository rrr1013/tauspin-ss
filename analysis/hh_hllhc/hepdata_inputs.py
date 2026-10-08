"""Public per-bin, per-process yields of the ATLAS Run-2 HH->bbtautau legacy analysis
(arXiv:2209.10910, HEPData ins2155171, Figure 8a/8b/8c: post-fit SM-HH BDT/NN
distributions, 139 fb^-1, 13 TeV) and the SM signal normalisation (Table 10).

Known HEPData inconsistency: in Figure 8c (lephad LTT) the "Z to tautau + HF" column
is identical, bin by bin, to the SLT column of Figure 8b, and the LTT components then
do not add up to the quoted "Total Background".  The LTT Z+HF yield is therefore
re-derived as Total Background minus the other components (sum 529 vs Table 10: 530).
The last LTT bin has Total Background == Top-quark (also corrupted); see load_run2.
"""
from pathlib import Path

import numpy as np
import yaml

LIT = Path(__file__).resolve().parent / "lit" / "mine"

# HEPData column -> analysis process key.  Spin hypotheses are attached in spin_model.py.
COLUMNS = {
    "hadhad": {
        "Top-quark": "top",
        "Jet to tau-had fakes (MJ)": "fake_mj",
        "Z to tautau + HF": "zhf",
        "Jet to tau-had fakes (ttbar)": "fake_tt",
        "Other": "other",
        "SM Higgs": "single_h",
    },
    "slt": {
        "Top-quark": "top",
        "Jet to tau-had fakes": "fake",
        "Z to tautau + HF": "zhf",
        "Other": "other",
        "SM Higgs": "single_h",
    },
    "ltt": {
        "Top-quark": "top",
        "Jet to tau-had fakes": "fake",
        "Z to tautau + HF": "zhf",
        "Other": "other",
        "SM Higgs": "single_h",
    },
}
FILES = {"hadhad": "hep2209_Figure8a.yaml", "slt": "hep2209_Figure8b.yaml", "ltt": "hep2209_Figure8c.yaml"}
# Table 10, SM ggF + VBF HH yields at 139 fb^-1 (pre-fit SM normalisation).
SM_SIGNAL = {"hadhad": 5.4 + 0.167, "slt": 5.9 + 0.2, "ltt": 1.42 + 0.055}


def _columns(path):
    d = yaml.safe_load(open(path))
    return {dv["header"]["name"]: np.array([v["value"] for v in dv["values"]], float)
            for dv in d["dependent_variables"]}


def load_run2():
    """Return {channel: {"signal": array, process: array, ..., "data": array}} at 139 fb^-1."""
    out = {}
    for ch, fname in FILES.items():
        c = _columns(LIT / fname)
        sig = c["SM HH at expected limit"]
        rec = {"signal": sig / sig.sum() * SM_SIGNAL[ch]}
        for col, key in COLUMNS[ch].items():
            rec[key] = c[col].copy()
        total = c["Total Background"]
        if ch == "ltt":
            others = sum(rec[k] for k in ("top", "fake", "other", "single_h"))
            rec["zhf"] = total - others
            # The last LTT bin is also corrupted (Total Background == Top-quark).  HEPData
            # Table 5 (most signal-like bin) gives its full breakdown: ttbar 1.76 +
            # single top 0.61 (= Top-quark 2.371), fakes 0.8, SM Higgs 0.4, other 0.33,
            # Z+HF 1.7, total 6.0; take Z+HF from there and rebuild that bin's total.
            rec["zhf"][-1] = 1.7
            total = others + rec["zhf"]
        bkg = sum(rec[k] for k in COLUMNS[ch].values())
        rec["_closure"] = float(np.max(np.abs(bkg - total) / total))
        rec["data"] = c["Data"]
        rec["postfit_unc"] = c["Post-fit Uncertainty"]
        out[ch] = rec
    return out


if __name__ == "__main__":
    for ch, rec in load_run2().items():
        procs = [k for k in rec if k not in ("data", "postfit_unc", "_closure")]
        print(ch, "max |sum-total|/total =", f"{rec['_closure']:.2e}",
              {k: round(float(rec[k].sum()), 2) for k in procs})


# ---------------------------------------------------------------------------------------
# Latest ATLAS analysis (arXiv:2607.26879): bins recovered from Figures 7/8 (extract_latest.py)
# Auxiliary tables give the true-tau ttbar / lepton-fake split that the figures merge.
LEPFAKE_FRAC = {  # lepton->tau_had fake share of the 'ttbar' stack (tabaux01/02)
    ("run2", "hadhad", "Hi"): 713 / (1343 + 713), ("run3", "hadhad", "Hi"): 329 / (563 + 329),
    ("run2", "hadhad", "Lo"): 497 / (722 + 497), ("run3", "hadhad", "Lo"): 564 / (666 + 564),
    ("run2", "hadhad", "VBF"): 93.9 / (206 + 93.9), ("run3", "hadhad", "VBF"): 69 / (114 + 69),
}


def load_latest(run):
    """{channel: rec} for the latest analysis; channel = '<hadhad|lephad>_<Lo|Hi|VBF>'.
    Signal = SM ggF+VBF HH (x20 overlays / 20).  'top' = ttbar (true tau, plus the
    lepton-fake share kept separately as 'top_lepfake' in hadhad) + tW."""
    import json
    import os
    out_dir = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs"))
    d = json.load(open(out_dir / "latest" / "latest_bins.json"))
    out = {}
    for key, r in d.items():
        if r["run"] != run:
            continue
        y = {k: np.array(v, float) for k, v in r["yields"].items()}
        ch = f"{r['channel']}_{r['sr']}"
        rec = {"signal": y["ggF_x20"] / 20 + y["VBF_x20"] / 20}
        if r["channel"] == "hadhad":
            f = LEPFAKE_FRAC[(run, "hadhad", r["sr"])]
            rec.update(top=y["ttbar"] * (1 - f) + y["tW"], top_lepfake=y["ttbar"] * f,
                       fake_mj=y["fake_mj"], fake_tt=y["fake_tt"])
        else:
            rec.update(top=y["ttbar"] + y["tW"], fake=y["fake"])
        rec.update(zhf=y["zhf"], other=y["other"], single_h=y["single_h"])
        rec["data"] = np.zeros_like(rec["signal"])
        rec["postfit_unc"] = np.zeros_like(rec["signal"])
        rec["_closure"] = 0.0
        out[ch] = rec
    return out
