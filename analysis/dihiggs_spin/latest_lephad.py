"""Fixed Table-2-informed Transformer *proxy* for the latest lep-had analysis.

This is not an ATLAS reproduction: legacy toy trigger/b-tagging, no GN2 scores,
VBF jets, MMC object token, topness, or auxiliary top head.  Had inputs are
one-sided traditional hadronic observables, not a TauSpin polarimeter.
The uid%3=2 evaluation fold reuses the existing MC and is exploratory.
Run `prepare`, then `pilot`; production `train` requires --commit SHA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import socket

import numpy as np
from scipy.optimize import minimize

from classify_lephad import GROUPS, PRIOR
from reco import delta_r, dphi, eta, mass, phi, pt
from reco_lephad import lepton_resolution

ARMS = ("B", "B+d0", "B+Had", "B+d0+Had")
VARIANTS = ("nominal", "null", "real", "nonprompt")
CATEGORIES = (("Hi", 1), ("Lo", 0))
TOKEN_NAMES = ("b1", "b2", "lep", "tau_vis", "met")
TOKEN_COLS = ("log1p_pt_GeV", "eta", "sin_phi", "cos_phi", "log1p_mass_GeV",
              "is_b1", "is_b2", "is_lep", "is_tau", "is_met")


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "source_snapshot.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["commit"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return "uncommitted-pilot"


def verify_snapshot(commit):
    root = Path(__file__).resolve().parents[2]
    if git_commit() != commit:
        raise ValueError("Production requires exact --commit matching snapshot")
    manifest = root / "source_snapshot.json"
    if manifest.exists():
        checks = json.loads(manifest.read_text())["files"]
        for path, expected in checks.items():
            if sha256(root / path) != expected:
                raise ValueError(f"Frozen archive changed: {path}")
        pyfiles = {str(p.relative_to(root)) for p in (root / "analysis/dihiggs_spin").glob("*.py")}
        if pyfiles != {p for p in checks if p.startswith("analysis/dihiggs_spin/") and p.endswith(".py")}:
            raise ValueError("Unexpected/missing Python files in frozen source archive")
    else:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
        if status.strip():
            raise ValueError("Production requires a clean tracked tree")


def joined_raw(D, recodir):
    """Bijective join in original dataset row order; no reliance on row ordering."""
    n = len(D["uid"])
    raw = {k: np.empty((n, 2 if k == "met" else 4)) for k in TOKEN_NAMES}
    report = {}
    for proc in np.unique(D["proc"]):
        idx = np.flatnonzero(D["proc"] == proc)
        ids = D["uid"][idx]
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate dataset identity for {proc}")
        parts, uids = [], []
        files = sorted(Path(recodir).glob(f"{proc}_run_*.npz"))
        if not files:
            raise ValueError(f"No raw chunks for {proc}")
        for chunk, path in enumerate(files):
            with np.load(path) as z:
                ev = z["event"].astype(np.int64)
                if len(ev) and (ev.min() < 0 or ev.max() >= 1_000_000):
                    raise ValueError(f"uid chunk stride collision in {path}")
                uids.append(chunk * 1_000_000 + ev)
                parts.append({k: z[k] for k in TOKEN_NAMES})
        ids_raw = np.concatenate(uids)
        if len(np.unique(ids_raw)) != len(ids_raw) or not np.array_equal(np.sort(ids), np.sort(ids_raw)):
            raise ValueError(f"Raw/dataset identities not a bijection for {proc}")
        order = np.argsort(ids_raw)
        match = order[np.searchsorted(ids_raw[order], ids)]
        for k in raw:
            raw[k][idx] = np.concatenate([p[k] for p in parts])[match]
        if not np.allclose(raw["lep"][idx], D["lep_p4"][idx], atol=1e-9, rtol=1e-9):
            raise ValueError(f"Lepton-p4 closure fails for {proc}")
        report[str(proc)] = {"rows": len(idx), "chunks": len(files), "bijection": True,
                             "lep_p4_max_abs_residual": float(np.abs(raw["lep"][idx] - D["lep_p4"][idx]).max())}
    return raw, report


def geometry(ip, lep):
    """Straight-track transverse closest approach from 3D closest approach [um]."""
    px, py, pz = lep[:, :3].T
    pT = pt(lep)
    d0 = (ip[:, 0] * py - ip[:, 1] * px) / pT
    z0 = ip[:, 2] - (ip[:, 0] * px + ip[:, 1] * py) * pz / pT**2
    sth = pT / np.linalg.norm(lep[:, :3], axis=1)
    return d0, z0 * sth, sth


def corrected_ip(D, variant, noise):
    ip = D["lep_ip_true"].copy()
    if variant == "nonprompt":
        # Named stress only: 50% of simulated prompt fake candidates get an
        # isotropic perpendicular displacement, exponential mean 100 um.
        pick = np.isin(D["proc"], ("ttljp", "tWlj")) & ~D["lep_from_tau"].astype(bool) & (noise["inject_u"] < 0.5)
        k = D["lep_p4"][:, :3]
        k = k / np.linalg.norm(k, axis=1)[:, None]
        a = np.cross(k, np.array([0., 0., 1.]))
        a /= np.linalg.norm(a, axis=1)[:, None]
        b = np.cross(k, a)
        f = noise["inject_phi"]
        inj = (a * np.cos(f)[:, None] + b * np.sin(f)[:, None]) * noise["inject_r"][:, None]
        ip[pick] = inj[pick]
    s_d0, s_z0 = lepton_resolution(D["lep_p4"], D["lep_id"])
    real = variant in ("real", "nonprompt")
    if real:
        slope = 1 + 0.2 * np.abs(eta(D["lep_p4"]))
        s_d0 = np.hypot(s_d0 * slope, 10.0)
        s_z0 = s_z0 * slope
    tail_p = 0.02 + (0.07 * (np.abs(D["lep_id"]) == 11) if real else 0.0)
    tail = np.where(noise["tail_u"] < tail_p, 4.0, 1.0)
    d0true, z0strue, sth = geometry(ip, D["lep_p4"])
    if variant == "null":
        # Zero-true-d0 control: remove only the transverse displacement.
        # Retain original longitudinal z0, the common noise and the cohort.
        d0true = np.zeros_like(d0true)
    # s_z0 is longitudinal z0 resolution, not z0*sin(theta) resolution.
    # Scalar d0 smearing fixes the old radial-vector/Rayleigh construction.
    d0 = d0true + s_d0 * tail * noise["d0_normal"]
    z0s = z0strue + s_z0 * sth * tail * noise["z0_normal"]
    return dict(d0=d0, d0sig=np.abs(d0) / s_d0, sd0=s_d0, z0s=z0s,
                d0true=d0true, z0strue=z0strue, tail=tail, sz0=s_z0, sth=sth)


def prepare(args):
    start = time.monotonic()
    D = dict(np.load(args.data, allow_pickle=True))
    raw, join = joined_raw(D, args.recodir)
    n = len(D["w"])
    if not np.all(D["w"] > 0):
        raise ValueError("Positive-weight protocol requires every MC weight > 0")
    closures = {"kin_pt_b1": pt(raw["b1"]), "kin_pt_b2": pt(raw["b2"]),
                "kin_pt_tau": pt(raw["tau_vis"]), "kin_m_bb": mass(raw["b1"]+raw["b2"]),
                "kin_dr_bb": delta_r(raw["b1"],raw["b2"]), "kin_met": np.linalg.norm(raw["met"],axis=1)}
    for k,v in closures.items():
        if not np.allclose(v,D[k],atol=1e-8,rtol=1e-8):
            raise ValueError(f"Raw-v2 closure failed: {k}")
    obj = []
    for i, name in enumerate(TOKEN_NAMES):
        p = raw[name]
        if name == "met":
            p = np.column_stack([p, np.zeros(n), np.linalg.norm(p, axis=1)])
        t = np.column_stack([np.log1p(pt(p)), eta(p), np.sin(phi(p)), np.cos(phi(p)),
                             np.log1p(mass(p)) if i < 2 else np.zeros(n),
                             np.tile(np.eye(5)[i], (n, 1))])
        obj.append(t)
    tokens = np.stack(obj, axis=1).astype(np.float32)
    kin = sorted(k for k in D if k.startswith("kin_") and k != "kin_mmc_valid")
    extra = {}
    b1, b2, lep, tau = (raw[k] for k in TOKEN_NAMES[:4])
    for name, x in (("lep", lep), ("tau", tau)):
        mbx = np.stack([mass(b1 + x), mass(b2 + x)], 1)
        extra[f"m_b{name}_min"] = mbx.min(1)
        extra[f"m_b{name}_max"] = mbx.max(1)
        for j, b in enumerate((b1, b2), 1):
            extra[f"dr_b{j}_{name}"] = delta_r(b, x)
    metphi = np.arctan2(raw["met"][:, 1], raw["met"][:, 0])
    for name in TOKEN_NAMES[:4]:
        extra[f"abs_dphi_{name}_met"] = np.abs(dphi(phi(raw[name]), metphi))
    extra["pt_visible_bb_lep_tau"] = pt(b1 + b2 + lep + tau)
    gn = kin + ["lep_is_e", "mmc_lep_x"] + list(extra)
    globals_ = np.column_stack([D[c] for c in kin] + [(np.abs(D["lep_id"]) == 11), D["lip_x"]] + list(extra.values())).astype(np.float32)
    hadcols = sorted(k for k in D if k.startswith("had_") and not k.startswith(("had_tip_", "had_tsv_")))
    had = np.column_stack([D[k] for k in hadcols]).astype(np.float32)
    rng = np.random.default_rng(20261008)
    noise = {"tail_u": rng.random(n), "d0_normal": rng.standard_normal(n), "z0_normal": rng.standard_normal(n),
             "inject_u": rng.random(n), "inject_phi": rng.uniform(0, 2*np.pi, n), "inject_r": rng.exponential(100, n)}
    out = dict(tokens=tokens, globals=globals_, had=had, proc=D["proc"], uid=D["uid"],
               w=D["w"], y=(D["proc"] == "hhC").astype(np.int64), trigger=D["trigger"],
               m_hh=D["kin_m_hh"], mass_category=(D["kin_m_hh"] > 350.).astype(np.int64),
               lep_from_tau=D["lep_from_tau"], lep_is_e=np.abs(D["lep_id"]) == 11,
               lep_pt=pt(lep), lep_eta=eta(lep), true_ip=D["lep_ip_true"])
    base = ((D["trigger"] == 0) & (D["kin_m_bb"] > 40.) & (D["kin_m_bb"] < 150.)
            & (D["kin_m_vis"] > 40.) & (D["kin_m_tautau"] > 60.))
    report = {"N": n, "join": join, "raw_kinematic_closure_max_abs": {k:float(np.abs(v-D[k]).max()) for k,v in closures.items()},
              "variants": {}, "columns": {"tokens": TOKEN_COLS, "globals": gn, "had": hadcols}}
    d0true, z0strue, sth = geometry(D["lep_ip_true"], lep)
    out["d0_true_corrected"] = d0true
    out["z0sin_true_corrected"] = z0strue
    out["d0_true_legacy_radius"] = np.linalg.norm(D["lep_ip_true"][:, :2], axis=1)
    out["z0sin_true_legacy"] = D["lep_ip_true"][:, 2] * sth
    # Independently obtain closest transverse point on the track, verifying
    # both scalar definitions and 3D closest-point perpendicularity.
    a = -(D["lep_ip_true"][:, :2] * lep[:, :2]).sum(1) / pt(lep)**2
    transverse_point = D["lep_ip_true"] + a[:, None] * lep[:, :3]
    report["geometry"] = {"d0_abs_vs_transverse_distance_max_um": float(np.max(np.abs(np.abs(d0true) - np.linalg.norm(transverse_point[:, :2], axis=1)))),
                          "z0sin_vs_transverse_point_max_um": float(np.max(np.abs(z0strue - transverse_point[:, 2] * sth))),
                          "ip_dot_unit_momentum_max_um": float(np.max(np.abs((D["lep_ip_true"] * lep[:, :3]).sum(1) / np.linalg.norm(lep[:, :3], axis=1))))}
    for variant in VARIANTS:
        f = corrected_ip(D, variant, noise)
        out[f"ip_{variant}"] = np.column_stack([f["d0sig"], f["sd0"]]).astype(np.float32)
        out[f"z0s_{variant}"] = f["z0s"].astype(np.float32)
        out[f"sel_{variant}"] = base & (np.abs(f["z0s"]) < 500.)
        out[f"d0_{variant}"] = f["d0"].astype(np.float32)
        sel = out[f"sel_{variant}"]
        core_prompt = ~D["lep_from_tau"].astype(bool) & (f["tail"] == 1)
        pull = (f["d0"] - f["d0true"]) / f["sd0"]
        counts = {}
        for proc in np.unique(D["proc"]):
            m = sel & (D["proc"] == proc)
            counts[str(proc)] = {"rows": int(m.sum()), "yield_3ab": float(D["w"][m].sum()),
                                 "split_rows": np.bincount(D["uid"][m] % 3, minlength=3).tolist()}
        report["variants"][variant] = {"selected": int(sel.sum()), "selection_z0sin_max_um": 500.,
                                      "prompt_core_pull_mean": float(pull[core_prompt].mean()),
                                      "prompt_core_pull_sd": float(pull[core_prompt].std()), "processes": counts}
    for k in ("tokens", "globals", "had"):
        if not np.isfinite(out[k]).all():
            raise ValueError(f"Nonfinite {k}")
    if not np.array_equal(out["sel_null"], out["sel_nominal"]):
        raise ValueError("Zero-true-d0 null changed nominal selection")
    if not np.array_equal(out["z0s_null"], out["z0s_nominal"]):
        raise ValueError("Zero-true-d0 null changed longitudinal z0/noise")
    report["null_control"] = {"definition": "Set only true scalar transverse d0 to zero; original longitudinal z0 and all RNG draws retained",
                              "same_selection_as_nominal": True, "same_z0sin_as_nominal": True}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out) / "prepared.npz", **out)
    report["source_dataset"] = {"path": str(Path(args.data).resolve()), "sha256": sha256(args.data)}
    report["timing_seconds"] = time.monotonic() - start
    report["definition"] = "Latest-paper-informed proxy SR: SLT only, 40<m_bb<150 GeV, m_vis(lepton,tau)>40 GeV, MMC>60 GeV, corrected |z0*sin(theta)|<500um; NO d0 selection. Hi m_HH>350 GeV vs Lo m_HH<=350 GeV. Legacy toy trigger/b-tag response retained. Scalar d0 Gaussian width sigma with common 2% 4x tail; longitudinal z0 width converted by sin(theta)."
    dump(Path(args.out) / "prepared.json", report)
    print(json.dumps({"prepared": report["N"], "geometry": report["geometry"], "timing_seconds": report["timing_seconds"]}), flush=True)


def standardize(x, train):
    mean = x[train].mean(0, dtype=np.float64).astype(np.float32)
    sd = x[train].std(0, dtype=np.float64).astype(np.float32)
    sd = np.where(sd > 1e-6, sd, 1.)
    return ((x - mean) / sd).astype(np.float32), mean, sd


def model_factory(nglobal, ntoken=10):
    import torch
    from torch import nn

    class Proxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(ntoken, 64)
            self.cls = nn.Parameter(torch.zeros(1, 1, 64))
            layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, dropout=0.1,
                                              activation="gelu", batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
            self.global_mlp = nn.Sequential(nn.Linear(nglobal, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        def forward(self, t, g):
            t = self.embedding(t)
            t = self.encoder(torch.cat([self.cls.expand(len(t), -1, -1), t], 1))[:, 0]
            return self.head(torch.cat([t, self.global_mlp(g)], 1)).squeeze(1)
    return Proxy()


def inputs(D, variant, arm):
    g = [D["globals"]]
    if "d0" in arm:
        g.append(D[f"ip_{variant}"])
    if "Had" in arm:
        g.append(D["had"])
    return D["tokens"], np.concatenate(g, 1)


def train_one(args, D, variant, arm, seed, pilot=False):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    sel = D[f"sel_{variant}"].astype(bool)
    idx = np.flatnonzero(sel)
    if pilot:
        # Deterministic process/fold-stratified small pilot; no pilot score can
        # be used as an evaluation result or tune the architecture.
        picked = []
        for proc in np.unique(D["proc"]):
            for fold in range(3):
                m = idx[(D["proc"][idx] == proc) & (D["uid"][idx] % 3 == fold)]
                picked.extend(m[:args.pilot_per_group])
        idx = np.array(sorted(picked))
    folds = D["uid"][idx] % 3
    tr, va = folds == 0, folds == 1
    t, g = inputs(D, variant, arm)
    t = t[idx]
    _, tmean, tsd = standardize(t, tr)
    # Object-type one-hots are constant at each token slot and must not be
    # centered away: they are the only type/slot information in the encoder.
    tmean[:, 5:] = 0.; tsd[:, 5:] = 1.
    t = ((t-tmean)/tsd).astype(np.float32)
    g, gmean, gsd = standardize(g[idx], tr)
    y, w = D["y"][idx], D["w"][idx].copy()
    balance = w[tr & (y == 0)].sum() / w[tr & (y == 1)].sum()
    w *= np.where(y == 1, balance, 1.)
    w /= w[tr].mean()
    tt, gg, yy, ww = [torch.as_tensor(v, device=device) for v in (t, g, y.astype(np.float32), w.astype(np.float32))]
    model = model_factory(g.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, min_lr=1e-6)
    best, best_epoch, stale, history = np.inf, 0, 0, []
    out = Path(args.out) / ("pilot" if pilot else "models") / variant / arm.replace("+", "_") / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_idx, val_idx = np.flatnonzero(tr), np.flatnonzero(va)
    epochs = args.pilot_epochs if pilot else args.max_epochs
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        tic = time.monotonic()
        model.train()
        numerator = denominator = 0.
        permutation = rng.permutation(train_idx)
        for begin in range(0, len(permutation), args.batch_size):
            b = permutation[begin:begin+args.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(tt[b], gg[b])
            vals = F.binary_cross_entropy_with_logits(logits, yy[b], reduction="none")
            # Fixed training-fold normalization makes minibatch gradients an
            # unbiased estimate of the class-balanced weighted objective.
            loss = (vals * ww[b]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            numerator += float((vals.detach() * ww[b]).sum())
            denominator += float(ww[b].sum())
        model.eval()
        val_num = val_den = 0.
        with torch.no_grad():
            for begin in range(0, len(val_idx), args.batch_size):
                b = val_idx[begin:begin+args.batch_size]
                vals = F.binary_cross_entropy_with_logits(model(tt[b], gg[b]), yy[b], reduction="none")
                val_num += float((vals * ww[b]).sum())
                val_den += float(ww[b].sum())
        vl = val_num / val_den
        scheduler.step(vl)
        improved = vl < best - args.min_delta
        if improved:
            best, best_epoch, stale = vl, epoch, 0
            torch.save({"state_dict": model.state_dict(), "token_mean": tmean, "token_sd": tsd,
                        "global_mean": gmean, "global_sd": gsd, "balance": float(balance),
                        "variant": variant, "arm": arm, "seed": seed, "epoch": epoch,
                        "git_commit": git_commit(), "nglobal": g.shape[1]}, out / "best.pt")
        else:
            stale += 1
        h = dict(epoch=epoch, train_loss=numerator / denominator, val_loss=vl,
                 lr=opt.param_groups[0]["lr"], seconds=time.monotonic()-tic)
        history.append(h)
        dump(out / "curve.json", history)
        print(variant, arm, seed, json.dumps(h), flush=True)
        if not pilot and stale >= args.patience:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scores = np.full(len(D["uid"]), np.nan, dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(idx), args.batch_size):
            b = np.arange(begin, min(begin+args.batch_size, len(idx)))
            scores[idx[b]] = torch.sigmoid(model(tt[b], gg[b])).cpu().numpy()
    np.savez_compressed(out / "scores.npz", score=scores, uid=D["uid"], proc=D["proc"], selected=sel)
    status = dict(best_epoch=best_epoch, stopped_epoch=len(history), best_val_loss=float(best),
                  seconds=time.monotonic()-started, n_train=int(tr.sum()), n_val=int(va.sum()),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                  optimization_complete=(not pilot and stale >= args.patience),
                  technical_ceiling_reached=(not pilot and len(history) >= args.max_epochs and stale < args.patience),
                  git_commit=git_commit(), architecture="4 layers, 4 heads, d_model64, FF256, CLS pooling, globals MLP64; dropout0.1")
    dump(out / "status.json", status)
    return scores, status


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    return np.interp(q, (np.cumsum(w[order]) - .5*w[order]) / w.sum(), x[order])


def bin_edges(score, y, w, min_ess=25., min_b=3., nquant=8):
    sig = y == 1
    edges = np.unique(weighted_quantile(score[sig], w[sig], np.linspace(0, 1, nquant+1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    keep = [np.inf]
    hi = len(edges)-1
    while hi > 0:
        for lo in range(hi-1, -1, -1):
            m = (score >= edges[lo]) & (score < keep[-1]) & ~sig
            B, V = w[m].sum(), (w[m]**2).sum()
            ess = B**2 / V if V > 0 else 0.
            if (ess >= min_ess and B >= min_b) or lo == 0:
                keep.append(edges[lo])
                hi = lo
                break
    return np.array(keep[::-1])


def templates(score, D, mask, edges, multiplicity=None):
    ids = np.flatnonzero(mask)
    bins = np.clip(np.searchsorted(edges, score[ids], side="right")-1, 0, len(edges)-2)
    w = 3. * D["w"][ids]
    mult = np.ones(len(ids)) if multiplicity is None else multiplicity[ids]
    nbin = len(edges)-1
    def hist(m, values):
        return np.bincount(bins[m], weights=values[m], minlength=nbin)
    procs = {}
    prompt = ~D["lep_from_tau"][ids].astype(bool)
    for proc in np.unique(D["proc"]):
        m = D["proc"][ids] == proc
        procs[str(proc)] = {"count": hist(m, mult), "yield": hist(m, w*mult), "variance": hist(m, w*w*mult),
                            "prompt_count": hist(m & prompt, mult), "prompt_yield": hist(m & prompt, w*mult),
                            "prompt_variance": hist(m & prompt, w*w*mult)}
    s = procs["hhC"]["yield"]
    B = np.stack([sum((procs[p]["yield"] for p in ps), np.zeros(nbin)) for ps in GROUPS.values()])
    V = sum((v["variance"] for p, v in procs.items() if p != "hhC"), np.zeros(nbin))
    b = B.sum(0)
    ess = np.divide(b*b, V, out=np.zeros_like(b), where=V > 0)
    return s, B, V, ess, procs


def profile(s, B, V, norm=True, mc=True):
    """Shared positive normalization nuisances; effective-count Barlow-lite.

    Each bin's positive beta has Poisson auxiliary count tau=B^2/V.
    beta is analytically profiled under mu=0.  The signal+background Asimov
    alternative has theta=0, beta=1 and exactly zero relative NLL.
    No background floors or invented samples are used.
    """
    b = B.sum(0)
    if np.any((b <= 0) & (s > 0)):
        return {"Z": None, "success": False, "reason": "signal with zero evaluation background; sensitivity unresolved"}
    ok = b > 0
    s, B, V, b = s[ok], B[:, ok], V[ok], b[ok]
    n = s+b
    tau = np.divide(b*b, V, out=np.full_like(b, np.inf), where=V > 0)
    # Products/summation in extended precision keep Newton refinements
    # readable even when the fitted NLL is tiny relative to the event yields.
    B, n, tau = (np.asarray(a,dtype=np.longdouble) for a in (B,n,tau))
    prior = np.log1p(np.array(list(PRIOR.values()))) if norm else np.zeros(len(GROUPS))
    def poisson_relative_deviance(delta):
        # delta-log(1+delta), stable when the signal is tiny compared to B.
        out = delta - np.log1p(delta)
        small = np.abs(delta) < 1e-3
        x = delta[small]
        out[small] = x*x*(.5+x*(-1./3+x*(.25+x*(-.2+x/6))))
        return out
    def objective(theta):
        components = B * np.exp(prior[:, None]*theta[:, None])
        bs = components.sum(0)
        finite = np.isfinite(tau) if mc else np.zeros(len(b), dtype=bool)
        factor = np.ones_like(b)
        factor[finite] = tau[finite]/(bs[finite]+tau[finite])
        # lambda-n = factor*(bs-n); evaluate this before forming lambda.
        delta = factor*(bs-n)/n
        dev = np.sum(n*poisson_relative_deviance(delta))
        beta_delta = (n[finite]-bs[finite])/(bs[finite]+tau[finite])
        aux = np.sum(tau[finite]*poisson_relative_deviance(beta_delta))
        value = float(dev+aux+.5*np.dot(theta,theta))
        # Envelope derivative after the analytic per-bin beta profile.
        gradient = np.asarray(prior * (components/bs * (factor*(bs-n))[None,:]).sum(1) + theta,dtype=float)
        return value, gradient
    result = minimize(objective, np.zeros(len(GROUPS)), jac=True, method="L-BFGS-B",
                      options={"ftol":1e-16, "gtol":1e-8, "maxiter":1000, "maxls":100})
    theta = result.x.copy()
    value, gradient = objective(theta)
    # Polish the nearly stationary fit: large curvature can satisfy an NLL
    # stopping test before an absolute gradient test, even with stable NLL.
    polish_steps = 0
    for _ in range(4):
        if np.max(np.abs(gradient)) < 1e-8:
            break
        components = B*np.exp(prior[:,None]*theta[:,None])
        bs = components.sum(0)
        finite = np.isfinite(tau) if mc else np.zeros(len(b),dtype=bool)
        factor = np.ones_like(b)
        factor[finite] = tau[finite]/(bs[finite]+tau[finite])
        r = factor*(bs-n)
        curvature = n.copy()
        curvature[finite] = tau[finite]*(2*n[finite]*bs[finite]+n[finite]*tau[finite]-bs[finite]**2)/(bs[finite]+tau[finite])**2
        A = prior[:,None]*components/bs
        hessian = np.asarray(np.eye(len(prior))+(A*curvature)@A.T+np.diag((prior[:,None]*A*r).sum(1)),dtype=float)
        if np.linalg.eigvalsh(hessian).min() <= 0:
            break
        candidate = theta-np.linalg.solve(hessian,gradient)
        new_value,new_gradient = objective(candidate)
        rounding = 64*np.finfo(float).eps*max(1.,abs(value))
        if new_value > value+rounding or np.max(np.abs(new_gradient)) >= np.max(np.abs(gradient)):
            break
        theta,value,gradient = candidate,new_value,new_gradient
        polish_steps += 1
    gradient_max = float(np.max(np.abs(gradient)))
    converged = bool(result.success and gradient_max < 1e-6)
    return {"Z": float(np.sqrt(max(2*value, 0.))), "success": converged,
            "normalization_pull": theta.tolist(), "message": str(result.message),
            "gradient_max_abs": gradient_max,
            "newton_polish_steps": polish_steps,
            "numerics": "Stable Poisson relative deviance and analytic envelope gradient; unchanged likelihood"}


def serial_templates(s, B, V, ess, procs):
    def population(values, index=None):
        get = (lambda a: float(a.sum())) if index is None else (lambda a: float(a[index]))
        total, prompt = get(values["yield"]), get(values["prompt_yield"])
        return {k: get(a) for k,a in values.items()} | {"prompt_fraction": prompt/total if total > 0 else None}
    process_report = {}
    for p,values in procs.items():
        process_report[p] = {k:a.tolist() for k,a in values.items()}
        process_report[p]["prompt_fraction"] = [float(a/b) if b > 0 else None for a,b in zip(values["prompt_yield"],values["yield"])]
        process_report[p]["overall"] = population(values)
        process_report[p]["highest_score_bin"] = population(values,-1)
    bg_values = {k:sum((v[k] for p,v in procs.items() if p != "hhC"),np.zeros_like(s)) for k in ("count","yield","variance","prompt_count","prompt_yield","prompt_variance")}
    return dict(S=s.tolist(), background_groups={g:B[i].tolist() for i,g in enumerate(GROUPS)},
                total_background=B.sum(0).tolist(), mc_variance=V.tolist(), background_ESS=ess.tolist(),
                processes=process_report, background_overall=population(bg_values),
                background_highest_score_bin=population(bg_values,-1),
                poor_bins=np.flatnonzero((ess < 25.) | (B.sum(0) < 3.)).tolist())


def evaluate(score, D, variant):
    sel = D[f"sel_{variant}"].astype(bool) & np.isfinite(score)
    split = D["uid"] % 3
    all_s, all_B, all_V = [], [], []
    cats = {}
    for name, category in CATEGORIES:
        va = sel & (split == 1) & (D["mass_category"] == category)
        ev = sel & (split == 2) & (D["mass_category"] == category)
        edges = bin_edges(score[va], D["y"][va], 3*D["w"][va])
        s, B, V, ess, procs = templates(score, D, ev, edges)
        vs, vB, vV, vess, vp = templates(score, D, va, edges)
        cats[name] = dict(edges=[float(x) if np.isfinite(x) else ("-inf" if x < 0 else "inf") for x in edges],
                          validation=serial_templates(vs,vB,vV,vess,vp), evaluation=serial_templates(s,B,V,ess,procs))
        all_s.append(s); all_B.append(B); all_V.append(V)
    s, B, V = np.concatenate(all_s), np.concatenate(all_B,axis=1), np.concatenate(all_V)
    q = profile(s,B,V)
    poor = any(c["evaluation"]["poor_bins"] for c in cats.values())
    return dict(categories=cats, profile=q, normalization_only=profile(s,B,V,mc=False),
                stat_only=profile(s,B,V,norm=False,mc=False), sensitivity_resolved=(not poor and q["success"]),
                yield_fold_scale=3., norm_prior=PRIOR,
                boundary="Exploratory evaluation on reused MC; fixed validation bins; proxy sensitivity, not an ATLAS forecast.")


def bootstrap(args, D, results):
    # Multiplicities are paired across arms and preserve seed/arm score binning.
    rng = np.random.default_rng(991026)
    scores = {k:np.load(v["score_path"])["score"] for k,v in results.items() if k.startswith("nominal/")}
    ratios = {k:[] for k in scores if not k.split("/")[1] == "B"}
    for rep in range(args.bootstrap):
        mult = rng.poisson(1., len(D["uid"]))
        z = {}
        for key, score in scores.items():
            s, B, V = [], [], []
            for name,category in CATEGORIES:
                edge = np.array([float(x) for x in results[key]["evaluation"]["categories"][name]["edges"]])
                m = D["sel_nominal"] & (D["uid"]%3 == 2) & (D["mass_category"] == category)
                st, bt, vt, _, _ = templates(score,D,m,edge,mult)
                s.append(st);B.append(bt);V.append(vt)
            p = profile(np.concatenate(s),np.concatenate(B,1),np.concatenate(V))
            z[key] = p["Z"] if p["success"] else None
        for key in ratios:
            base = f"nominal/B/{key.split('/')[-1]}"
            if z[key] is not None and z[base] is not None and z[base] > 0:
                ratios[key].append(z[key]/z[base])
    return {k:{"replicates":v, "n_valid":len(v), "requested":args.bootstrap,
               "interval_16_84":np.quantile(v,[.16,.84]).tolist() if v else None,
               "sd":float(np.std(v,ddof=1)) if len(v)>1 else None} for k,v in ratios.items()}


def aggregate(results):
    out = {}
    for variant in VARIANTS:
        for arm in ARMS:
            pairs = []
            values = []
            resolved = []
            for key, result in results.items():
                if not key.startswith(f"{variant}/{arm}/") or "evaluation" not in result:
                    continue
                ev = result["evaluation"]
                z = ev["profile"]["Z"]
                if z is None or not ev["profile"]["success"]:
                    continue
                values.append(z); resolved.append(ev["sensitivity_resolved"])
                base = results.get(f"{variant}/B/{key.split('/')[-1]}", {}).get("evaluation")
                if base and base["profile"]["Z"] and base["profile"]["success"]:
                    pairs.append(z/base["profile"]["Z"])
            if values:
                out[f"{variant}/{arm}"] = dict(Z_runs=values, Z_mean=float(np.mean(values)),
                    Z_seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                    paired_R_runs=pairs, paired_R_mean=float(np.mean(pairs)) if pairs else None,
                    paired_R_seed_sd=float(np.std(pairs,ddof=1)) if len(pairs)>1 else None,
                    all_sensitivity_resolved=all(resolved),
                    interpretation="Named proxy/stress only; seed spread is distinct from paired MC bootstrap uncertainty")
    return out


def gbdt(args, D, variant, arm, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    import pickle
    t,g = inputs(D,variant,arm)
    X = np.concatenate([t.reshape(len(t),-1),g],1)
    sel=D[f"sel_{variant}"].astype(bool)
    tr=sel&(D["uid"]%3==0); va=sel&(D["uid"]%3==1)
    w=D["w"].copy(); y=D["y"]
    w *= np.where(y==1,w[tr&(y==0)].sum()/w[tr&(y==1)].sum(),1.)
    w /= w[tr].mean()
    best=np.inf; stale=0; hist=[]; best_model=None
    model=HistGradientBoostingClassifier(max_iter=1, learning_rate=.05,max_leaf_nodes=31,
                                       min_samples_leaf=100,l2_regularization=1.,early_stopping=False,
                                       warm_start=True,random_state=seed)
    start=time.monotonic()
    for it in range(1,args.gbdt_max_iter+1):
        model.set_params(max_iter=it).fit(X[tr],y[tr],sample_weight=w[tr])
        vl=log_loss(y[va],model.predict_proba(X[va])[:,1],sample_weight=w[va])
        hist.append(dict(iteration=it,val_loss=float(vl)))
        if vl < best-args.min_delta:
            best=vl; stale=0; best_model=pickle.dumps(model)
        else: stale+=1
        if stale>=args.patience: break
    model=pickle.loads(best_model)
    score=np.full(len(y),np.nan,dtype=np.float32); score[sel]=model.predict_proba(X[sel])[:,1]
    out=Path(args.out)/"models"/variant/("GBDT_"+arm.replace("+","_"))/f"seed{seed}"
    out.mkdir(parents=True,exist_ok=True)
    (out/"best.pkl").write_bytes(best_model)
    np.savez_compressed(out/"scores.npz",score=score,uid=D["uid"],proc=D["proc"],selected=sel)
    dump(out/"curve.json",hist)
    status=dict(optimization_complete=stale>=args.patience,technical_ceiling_reached=stale<args.patience,
                best_val_loss=float(best),iterations=len(hist),seconds=time.monotonic()-start,git_commit=git_commit())
    dump(out/"status.json",status)
    return score,status,out


def run(args, pilot=False):
    import torch
    if not pilot:
        if not args.commit:
            raise ValueError("Production requires --commit")
        verify_snapshot(args.commit)
    Path(args.out).mkdir(parents=True,exist_ok=True)
    D=dict(np.load(args.prepared,allow_pickle=True))
    dump(Path(args.out)/("pilot_config.json" if pilot else "config.json"),dict(vars(args),git_commit=git_commit(),
         host=socket.gethostname(), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
         torch_version=torch.__version__, torch_file=torch.__file__, cuda_version=torch.version.cuda,
         gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
         prepared_sha256=sha256(args.prepared),protocol="Fixed proxy; uid%3 train/val/exploratory-eval; train-only normalization; no eval-directed choice",
         nuisance="Shared lognormal-positive group norms, per-bin effective-count Poisson Barlow-lite MC nuisance"))
    seeds=[int(s) for s in args.seeds.split(",")]
    variants=args.variants.split(",")
    arms=args.arms.split(",")
    result_path=Path(args.out)/("pilot_summary.json" if pilot else "summary.json")
    results=json.loads(result_path.read_text()) if args.resume and result_path.exists() else {}
    for variant in variants:
        for seed in seeds if variant=="nominal" else seeds[:1]:
            for arm in (arms if variant != "null" else [a for a in arms if a in ("B", "B+d0")]):
                key=f"{variant}/{arm}/{seed}"
                if args.resume and key in results and results[key]["status"]["optimization_complete"]: continue
                sc,status=train_one(args,D,variant,arm,seed,pilot)
                directory=Path(args.out)/("pilot" if pilot else "models")/variant/arm.replace("+","_")/f"seed{seed}"
                entry=dict(status=status,score_path=str(directory/"scores.npz"))
                # Never evaluate an optimization-incomplete production model.
                if not pilot and status["optimization_complete"]:
                    entry["evaluation"]=evaluate(sc,D,variant)
                results[key]=entry; dump(result_path,results)
    if not pilot and args.gbdt:
        for arm in ("B","B+d0"):
            key=f"nominal/GBDT_{arm}/0"
            if args.resume and key in results and results[key]["status"]["optimization_complete"]: continue
            sc,status,directory=gbdt(args,D,"nominal",arm,0)
            entry=dict(status=status,score_path=str(directory/"scores.npz"))
            if status["optimization_complete"]: entry["evaluation"]=evaluate(sc,D,"nominal")
            results[key]=entry;dump(result_path,results)
    if not pilot and args.bootstrap and all("evaluation" in v for k,v in results.items() if k.startswith("nominal/") and "GBDT" not in k):
        nominal={k:v for k,v in results.items() if "GBDT" not in k}
        dump(Path(args.out)/"bootstrap.json",bootstrap(args,D,nominal))
    if not pilot:
        dump(Path(args.out)/"aggregate.json",aggregate(results))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("prepare","pilot","train","evaluate"))
    ap.add_argument("--data");ap.add_argument("--recodir");ap.add_argument("--prepared")
    ap.add_argument("--out",required=True);ap.add_argument("--commit")
    ap.add_argument("--seeds",default="0,1,2");ap.add_argument("--variants",default="nominal,null,real,nonprompt")
    ap.add_argument("--arms",default=",".join(ARMS));ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--batch-size",type=int,default=2048);ap.add_argument("--max-epochs",type=int,default=250)
    ap.add_argument("--patience",type=int,default=25);ap.add_argument("--min-delta",type=float,default=1e-5)
    ap.add_argument("--lr",type=float,default=3e-4);ap.add_argument("--pilot-per-group",type=int,default=400)
    ap.add_argument("--pilot-epochs",type=int,default=3);ap.add_argument("--bootstrap",type=int,default=100)
    ap.add_argument("--gbdt",action="store_true");ap.add_argument("--gbdt-max-iter",type=int,default=1000)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    if args.command=="prepare":prepare(args)
    elif args.command in ("train","pilot"):run(args,args.command=="pilot")
    else:
        D=dict(np.load(args.prepared,allow_pickle=True)); p=Path(args.out)/"summary.json"
        results=json.loads(p.read_text())
        for key,r in results.items():
            if r["status"]["optimization_complete"]:
                r["evaluation"]=evaluate(np.load(r["score_path"])["score"],D,key.split("/")[0])
        dump(p,results)
        dump(Path(args.out)/"aggregate.json",aggregate(results))
        if args.bootstrap: dump(Path(args.out)/"bootstrap.json",bootstrap(args,D,{k:v for k,v in results.items() if "GBDT" not in k}))


if __name__=="__main__":main()
