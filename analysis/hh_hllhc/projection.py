"""Likelihood-level HL-LHC projection of the ATLAS HH->bbtautau analysis with and
without an additional TauSpin discriminant.

Baseline: per-bin, per-process post-fit yields of the final SM-HH BDT/NN distributions
of the Run-2 legacy analysis (hepdata_inputs.py), scaled to the HL-LHC luminosity and
14 TeV cross sections.  Spin: every baseline bin b is split into n_sub bins of a
composition-aware spin discriminant
    D_b = log p_H - log sum_p f_{p,b} p_{X(p)},
whose per-process templates come from out-of-fold spin-classifier probabilities on the
spin-flat HH cohort reweighted to each process's spin state (spin_response.py).  The
sub-bin edges are signal quantiles taken on fold 0; the templates are filled with folds
1 and 2 only.  Optionally (hadhad) the templates are conditioned on the range of the
kinematic score K that holds the same signal fraction as the ATLAS bin.

Statistics: binned Poisson likelihood (pyhf), discovery q0 on the Asimov dataset with
mu = 1.  Free normalisation factors for top and Z+HF shared by all channels (as in the
ATLAS fit); optional normalisation and spin-shape systematics.
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pyhf
from scipy.optimize import minimize

from hepdata_inputs import load_latest, load_run2

HERE = Path(__file__).resolve().parent
SPIN = Path(os.environ.get("HH_OUT", HERE / "outputs")) / "spin"

# process -> spin hypothesis, per channel type
DITAU_MAP = {"signal": "H", "single_h": "H", "zhf": "Z", "top": "W", "top_lepfake": "WU",
             "fake_mj": "U", "fake_tt": "WU", "other": "U"}
SINGLE_MAP = {"signal": "H", "single_h": "H", "zhf": "Z", "top": "W", "fake": "H", "other": "H"}
DI_HYPS = ("H", "Z", "W", "WU", "U")
SI_HYPS = ("H", "Z", "W")


SOURCES = {  # baseline analysis -> (loader, luminosity of the published distributions)
    "run2legacy": (load_run2, 139.0),
    "latest_run2": (lambda: load_latest("run2"), 140.0),
    "latest_run3": (lambda: load_latest("run3"), 56.0),
}
# 13.6 -> 14 TeV (LHC HWG / top cross sections): HH 36.69/34.43, ttbar 985/924, H ~1.04
XS_RUN3 = {"default": 1.066, "signal": 1.066, "single_h": 1.04}


def scale_factors(cfg, lumi0):
    """Luminosity x cross-section scaling from the published distributions to the target."""
    lum = cfg["lumi"] / lumi0
    xs = cfg["xs_ratio"]
    return {p: lum * xs.get(p, xs["default"]) for p in
            ("signal", "single_h", "zhf", "top", "top_lepfake", "fake_mj", "fake_tt", "fake", "other")}


def channel_yields(cfg):
    loader, lumi0 = SOURCES[cfg.get("source", "run2legacy")]
    data = loader()
    sf = scale_factors(cfg, lumi0)
    out = {}
    chans = cfg["channels"] if cfg.get("source", "run2legacy") == "run2legacy" else sorted(data)
    for ch in chans:
        rec = data[ch]
        procs = [k for k in rec if k not in ("data", "postfit_unc", "_closure")]
        out[ch] = {p: rec[p] * sf[p] for p in procs}
    return out


def mass_category(ch):
    """m_HH category of a latest-analysis channel (Hi: m_HH > 350 GeV), else None."""
    return {"Hi": 1, "Lo": 0}.get(ch.split("_")[-1]) if "_" in ch else None


class SpinTemplates:
    """Per-bin spin templates from out-of-fold probabilities."""

    def __init__(self, arm, tag, n_sub, cond_k):
        z = np.load(SPIN / f"spin_probs_{tag}.npz", allow_pickle=True)
        self.n_sub, self.cond_k, self.arm = n_sub, cond_k, arm
        self.di_p = z[f"di_{arm}"]
        self.di_w = z["rho"] * z["w"][:, None]
        self.di_fold = z["fold"]
        self.K = z["K"]
        self.si_p = z[f"si_{arm}"]
        self.si_w = z["si_rho"] * z["si_w"][:, None]
        self.si_fold = z["si_fold"]
        self.m_hh = z["m_hh"]
        self.si_m_hh = np.r_[z["m_hh"], z["m_hh"]]

    @staticmethod
    def _templates(p, W, fold, hyps, frac, n_sub, sel):
        """frac: background fractions per hypothesis (len hyps).  Signal is H (col 0)."""
        if n_sub == 1:
            return np.ones((len(hyps), 1)), np.zeros((len(hyps), 1))
        mix = np.maximum(p @ frac, 1e-300)
        D = np.log(np.maximum(p[:, 0], 1e-300)) - np.log(mix)
        e_sel = sel & (fold == 0)
        i = np.flatnonzero(e_sel)
        o = np.argsort(D[i])
        cw = np.cumsum(W[i][o, 0])
        cw /= cw[-1]
        edges = np.interp(np.arange(1, n_sub) / n_sub, cw, D[i][o])
        t_sel = sel & (fold != 0)
        b = np.searchsorted(edges, D[t_sel])
        T = np.stack([np.bincount(b, weights=W[t_sel, j], minlength=n_sub) for j in range(len(hyps))])
        V = np.stack([np.bincount(b, weights=W[t_sel, j] ** 2, minlength=n_sub) for j in range(len(hyps))])
        tot = T.sum(1, keepdims=True)
        return T / tot, np.sqrt(V) / tot

    def ditau(self, frac, sig_range=None, mcat=None):
        sel = np.ones(len(self.di_fold), bool)
        if mcat is not None:
            sel = (self.m_hh > 350) if mcat == 1 else (self.m_hh <= 350)
        if self.cond_k and sig_range is not None:
            # bins holding almost no signal (e.g. the lowest scores, where the published
            # signal overlay is clipped) get a K window of at least 2% of the signal
            lo, hi = sig_range
            if hi - lo < 0.02:
                c = 0.5 * (lo + hi)
                lo, hi = max(c - 0.01, 0.0), min(c + 0.01, 1.0)
            sig_range = (lo, hi)
            base_sel = sel.copy()
            # K interval holding the same H-signal fraction (counted from the top) as
            # the ATLAS bin; quantiles from fold 0 signal weights.
            m0 = sel & (self.di_fold == 0)
            o = np.argsort(-self.K[m0])
            cw = np.cumsum(self.di_w[m0][o, 0])
            cw /= cw[-1]
            k_hi = np.interp(sig_range[0], cw, self.K[m0][o]) if sig_range[0] > 0 else np.inf
            k_lo = np.interp(sig_range[1], cw, self.K[m0][o]) if sig_range[1] < 1 else -np.inf
            sel = sel & (self.K <= k_hi) & (self.K > k_lo)
            if (sel & (self.di_fold == 0)).sum() < 300:
                sel = base_sel
        return self._templates(self.di_p, self.di_w, self.di_fold, DI_HYPS, frac, self.n_sub, sel)

    def single(self, frac, mcat=None):
        sel = np.ones(len(self.si_fold), bool)
        if mcat is not None:
            sel = (self.si_m_hh > 350) if mcat == 1 else (self.si_m_hh <= 350)
        return self._templates(self.si_p, self.si_w, self.si_fold, SI_HYPS, frac, self.n_sub, sel)


def proc_mix(Y, ditau, override):
    """Spin-hypothesis mixture vector of every process in a channel."""
    cmap = DITAU_MAP if ditau else SINGLE_MAP
    hyps = DI_HYPS if ditau else SI_HYPS
    out = {}
    for p in Y:
        v = np.zeros(len(hyps))
        spec = override.get(p, {cmap[p]: 1.0})
        for h, w in spec.items():
            if h == "U" and not ditau:
                h = "H"  # a single unpolarised tau has the flat (H-marginal) density
            v[hyps.index(h)] += w
        out[p] = v / v.sum()
    return out


_LT_CACHE = {}
ORIGINS = ("tau", "prompt", "nonprompt")


def lifetime_templates(cfg, ch, b):
    """|d0|/sigma templates (rows: tau, prompt, nonprompt) for lep-had bin b (lifetime.py).
    The tau shape is the signal tau->l shape in the K range of that bin, taken from the
    nearest more signal-like bin when fewer than 200 signal leptons fall in the range;
    prompt and nonprompt shapes are pooled over the cohort."""
    lt = cfg["lifetime"]
    path = Path(os.environ.get("HH_OUT", HERE / "outputs")) / "lifetime" / "lifetime_templates.json"
    if path not in _LT_CACHE:
        _LT_CACHE[path] = json.load(open(path))
    v = _LT_CACHE[path]["variants"][lt.get("variant", "nominal")]
    key = lt.get("cut", "ttva" if lt.get("ttva", False) else "nocut")
    src = cfg.get("source", "run2legacy")
    bins = v["channels"][ch if src == "run2legacy" else f"{src}:{ch}"]
    k = b
    while bins[k]["templates"][key]["n_tau_signal"] < 200 and k + 1 < len(bins):
        k += 1
    tau = np.array(bins[k]["templates"][key]["tau_signal"])
    pool = v["pooled"][key]
    return np.stack([tau, np.array(pool["prompt"]), np.array(pool["nonprompt"])])


def origin_mix(cfg, p):
    """Lepton-origin mixture (tau, prompt, nonprompt) of a lep-had process."""
    lt = cfg.get("lifetime") or {}
    f_top = lt.get("top_tau_fraction", 0.08)
    f_h = lt.get("single_h_tau_fraction", 0.92)
    f_np = lt.get("fake_nonprompt", 0.0)
    # tt-bar l+jets fakes: the lepton comes from W -> tau -> l about as often as in the
    # true-tau top background (BR 0.039 vs 0.108 x lower pT acceptance) -> same share
    f_ft = lt.get("fake_tau_fraction", f_top)
    return {"signal": np.array([1.0, 0, 0]), "zhf": np.array([1.0, 0, 0]),
            "single_h": np.array([f_h, 1 - f_h, 0]), "top": np.array([f_top, 1 - f_top, 0]),
            "fake": np.array([f_ft * (1 - f_np), (1 - f_ft) * (1 - f_np), f_np]),
            "other": np.array([0, 1.0, 0])}[p]


def combine(t_spin, L, om):
    """Joint template: spin sub-bins x lepton-|d0|/sigma bins (independent given process)."""
    if L is None:
        return t_spin
    return np.outer(t_spin, om @ L).reshape(-1)


def build_spec(cfg, spin):
    """Return a pyhf workspace spec and bookkeeping."""
    yields = channel_yields(cfg)
    channels, info = [], {}
    sys_norm = cfg.get("norm_sys", {})
    shape_unc = cfg.get("spin_shape_unc", 0.0)
    for ch, Y in yields.items():
        ditau = ch.startswith("hadhad")
        mcat = mass_category(ch)
        hyps = DI_HYPS if ditau else SI_HYPS
        mix = proc_mix(Y, ditau, cfg.get("spin_mix", {}))
        bkg = [p for p in Y if p != "signal"]
        nb = len(Y["signal"])
        sig_cum = np.r_[0, np.cumsum(Y["signal"][::-1])][::-1] / Y["signal"].sum()  # fraction above bin edges
        rows = {p: [] for p in Y}
        templ_store = []
        for b in range(nb):
            btot = sum(Y[p][b] for p in bkg)
            frac = np.zeros(len(hyps))
            for p in bkg:
                frac += mix[p] * Y[p][b] / btot
            if spin is None:
                T = np.ones((len(hyps), 1))
            elif ditau:
                rng = (sig_cum[b + 1], sig_cum[b])  # (fraction above upper edge, above lower edge)
                T, _ = spin.ditau(frac, rng, mcat)
            else:
                T, _ = spin.single(frac, mcat)
            L = lifetime_templates(cfg, ch, b) if (not ditau and cfg.get("lifetime") is not None) else None
            templ_store.append((T, L))
            for p in Y:
                rows[p].append(Y[p][b] * combine(mix[p] @ T, L, None if L is None else origin_mix(cfg, p)))
        samples = []
        for p in Y:
            data = np.concatenate(rows[p])
            mods = []
            if p == "signal":
                mods.append({"name": "mu", "type": "normfactor", "data": None})
            if p == "top" and cfg.get("free_top", True):
                mods.append({"name": "k_top", "type": "normfactor", "data": None})
            if p == "zhf" and cfg.get("free_z", True):
                mods.append({"name": "k_z", "type": "normfactor", "data": None})
            if p in sys_norm:
                s = sys_norm[p]
                mods.append({"name": f"norm_{p}", "type": "normsys", "data": {"hi": 1 + s, "lo": 1 - s}})
            if cfg.get("bin_sys", 0) > 0 and p != "signal":
                # uncorrelated per-(baseline)-bin background shape uncertainty, shared by
                # the spin sub-bins of that baseline bin (does not favour the spin split)
                n_sub = len(data) // nb
                for b in range(nb):
                    hi = data.copy()
                    lo = data.copy()
                    sl = slice(b * n_sub, (b + 1) * n_sub)
                    hi[sl] *= 1 + cfg["bin_sys"]
                    lo[sl] *= 1 - cfg["bin_sys"]
                    mods.append({"name": f"binsys_{ch}_{b}", "type": "histosys",
                                 "data": {"hi_data": hi.tolist(), "lo_data": lo.tolist()}})
            if cfg.get("lephad_shape", 0) > 0 and not ditau and p in ("top", "fake"):
                # score-dependent modelling uncertainty of the dominant lep-had backgrounds,
                # linear in the baseline-bin index and common to its spin/d0 sub-bins
                a = cfg["lephad_shape"]
                n_sub = len(data) // nb
                tilt = np.repeat(1 + a * np.arange(nb) / max(nb - 1, 1), n_sub)
                mods.append({"name": f"lhshape_{p}", "type": "histosys",
                             "data": {"hi_data": (data * tilt).tolist(), "lo_data": (data * (2 - tilt)).tolist()}})
            if spin is not None and shape_unc > 0:
                # independent response-contrast nuisance per spin hypothesis: the template
                # of hypothesis h moves by +-u (T_h - mean_h' T_h') (correlated over bins
                # and channels), shrinking or enlarging its distance to the others
                for j, h in enumerate(hyps):
                    if mix[p][j] == 0:
                        continue
                    hi_rows, lo_rows = [], []
                    for b in range(nb):
                        T, L = templ_store[b]
                        base = mix[p] @ T
                        d = mix[p][j] * shape_unc * (T[j] - T.mean(0))
                        om = None if L is None else origin_mix(cfg, p)
                        hi_rows.append(Y[p][b] * combine(base + d, L, om))
                        lo_rows.append(Y[p][b] * combine(base - d, L, om))
                    mods.append({"name": f"spinshape_{h}", "type": "histosys",
                                 "data": {"hi_data": np.concatenate(hi_rows).tolist(),
                                          "lo_data": np.concatenate(lo_rows).tolist()}})
            samples.append({"name": p, "data": data.tolist(), "modifiers": mods})
        channels.append({"name": ch, "samples": samples})
        info[ch] = {"n_bins": int(len(rows["signal"]) and sum(len(r) for r in rows["signal"]))}
    spec = {"channels": channels,
            "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
            "observations": [], "version": "1.0.0"}
    return spec, info


def significance(spec, return_fit=False):
    """Discovery significance on the mu=1 Asimov dataset with all nuisances at nominal.

    On Asimov data the unconditional maximum is the generating point, so its -2lnL is
    evaluated there directly.  The mu=0 conditional minimum uses L-BFGS-B on the pyhf
    log-pdf from two starting points (nominal, and background norms raised to absorb
    the signal); the lower minimum is kept and both are reported."""
    pyhf.set_backend("numpy")
    model = pyhf.Workspace({**spec, "observations": [
        {"name": c["name"], "data": [0.0] * len(c["samples"][0]["data"])} for c in spec["channels"]]}
    ).model(modifier_settings={"normsys": {"interpcode": "code4"}, "histosys": {"interpcode": "code4p"}})
    init = np.asarray(model.config.suggested_init(), float)
    asimov = np.asarray(model.expected_data(init))
    bounds = list(model.config.suggested_bounds())
    for name in ("k_top", "k_z"):
        if name in model.config.parameters:
            bounds[model.config.par_slice(name).start] = (0.2, 5.0)
    poi = model.config.poi_index
    free = [i for i in range(len(init)) if i != poi]

    def nll(x):
        p = init.copy()
        p[free] = x
        p[poi] = 0.0
        return float(-2 * model.logpdf(p, asimov)[0])

    nll_free = float(-2 * model.logpdf(init, asimov)[0])
    starts = [init[free]]
    alt = init.copy()
    for name in ("k_top", "k_z"):
        if name in model.config.parameters:
            alt[model.config.par_slice(name).start] = 1.05
    starts.append(alt[free])
    fits = [minimize(nll, x0, method="L-BFGS-B", bounds=[bounds[i] for i in free],
                     options={"maxiter": 20000, "maxfun": 200000, "ftol": 1e-13, "gtol": 1e-9})
            for x0 in starts]
    from iminuit import Minuit
    for x0 in starts:
        m = Minuit(nll, x0)
        m.errordef = 1.0
        m.strategy = 2
        m.limits = [bounds[i] for i in free]
        m.migrad(ncall=200000)
        fits.append(type("F", (), {"fun": float(m.fval), "success": bool(m.valid)})())
    best = min(fits, key=lambda f: f.fun)
    q0 = max(best.fun - nll_free, 0.0)
    z = float(np.sqrt(q0))
    if return_fit:
        return z, {"q0_starts": [float(f.fun - nll_free) for f in fits],
                   "success": [bool(f.success) for f in fits]}
    return z


def run(cfg, arm=None, tag="s0_ct0.5", n_sub=4, cond_k=False):
    spin = None if arm is None else SpinTemplates(arm, tag, n_sub, cond_k)
    spec, info = build_spec(cfg, spin)
    z = {}
    for ch in [c["name"] for c in spec["channels"]]:
        sub = {**spec, "channels": [c for c in spec["channels"] if c["name"] == ch]}
        z[ch] = significance(sub)
    z["combined"] = significance(spec)
    return z, info


DEFAULT = {
    "channels": ["hadhad", "slt", "ltt"],
    "lumi": 3000.0,
    # ATL-PHYS-PUB-2024-016 Table 1: HH 1.18; ggF/VBF H 1.13, ZH 1.12, ttH 1.21 (single H
    # taken as 1.15); all other processes 1.18.  The HEPData yields are post-fit, so the
    # Z+HF normalisation (x1.3 applied by ATLAS to the pre-fit yield) is already included.
    "xs_ratio": {"default": 1.18, "signal": 1.18, "single_h": 1.15},
    "free_top": True, "free_z": True,
    "norm_sys": {},
    "bin_sys": 0.0,
    "spin_shape_unc": 0.0,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="s0_ct0.5")
    ap.add_argument("--n_sub", type=int, default=4)
    args = ap.parse_args()
    cfg = dict(DEFAULT)
    if args.config:
        cfg.update(json.load(open(args.config)))
    base, _ = run(cfg)
    print("baseline", {k: round(v, 3) for k, v in base.items()})
    for arm in ("textbook", "tauspin", "exact"):
        z, _ = run(cfg, arm, args.tag, args.n_sub)
        print(arm, {k: round(v, 3) for k, v in z.items()},
              "gain", {k: round(z[k] / base[k] - 1, 4) for k in z})
