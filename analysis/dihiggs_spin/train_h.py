"""tauspin-style point-h regressor for the di-Higgs samples.

Inputs: per-tau reconstructed substructure (charged and pi0 energy fractions
and angular offsets in the local (n, r, k) basis of the visible tau), the
leading-track PV-referenced impact vector and the 3-prong SV direction in the
same local basis, MET projected on the local axes, the hybrid polarimeter
from reco visibles + MMC neutrinos, reco decay mode, plus event-level m_vis,
MET, MMC m_tautau and status.
Targets: exact h- and h+ (6) and the 9 products h-_i h+_j (the second moments
that enter the spin-correlation likelihood ratio), MSE.
Training: split 0 (event % 3 == 0), true tau pairs with a defined exact h on
both sides from H (hh, zh, tth) and Z (zbb) events, H and Z reweighted to equal
total weight (the analogue of the H/Z training population of tauspin); 10 %
of it is held out for early stopping.  Predictions are written for all events.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

KIN_IN = ["kin_m_vis", "kin_met", "kin_m_tautau", "kin_mmc_status", "kin_dr_tautau"]


def spin_columns(d):
    return sorted(k for k in d.files if k.startswith("spin_"))


def make_xy(d):
    cols = spin_columns(d) + KIN_IN
    X = np.stack([d[c] for c in cols], 1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    h = d["h_exact"]
    Y = np.concatenate([h.reshape(len(h), 6), np.einsum("ni,nj->nij", h[:, 0], h[:, 1]).reshape(len(h), 9)], 1)
    return X, Y.astype(np.float32), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="", help="comma-separated input prefixes to drop (ablation)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data, allow_pickle=True)
    X, Y, cols = make_xy(d)
    if args.drop:
        keep = [i for i, c in enumerate(cols) if not any(c.startswith("spin_" + p) for p in args.drop.split(","))]
        X, cols = X[:, keep], [cols[i] for i in keep]
    proc = d["proc"]
    split = d["uid"] % 3
    is_h = np.isin(proc, ["hh", "zh", "tth"])
    is_z = proc == "zbb"
    valid = np.isfinite(Y).all(1) & d["is_true"].all(1)
    tr_all = np.flatnonzero((split == 0) & valid & (is_h | is_z))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tr_all)
    nval = len(tr_all) // 10
    va, tr = tr_all[:nval], tr_all[nval:]
    w = np.where(is_h, 0.5 / max(is_h[tr].sum(), 1), 0.5 / max(is_z[tr].sum(), 1)).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    model = nn.Sequential(nn.Linear(X.shape[1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                          nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                          nn.Linear(256, 15)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    t = lambda a: torch.as_tensor(a, device=dev)
    Xtr, Ytr, Wtr = t(Xn[tr]), t(Y[tr]), t(w[tr] * len(tr))
    Xva, Yva, Wva = t(Xn[va]), t(Y[va]), t(w[va] * len(va))
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for a in range(0, len(tr), 1024):
            i = perm[a:a + 1024]
            loss = (((model(Xtr[i]) - Ytr[i]) ** 2).mean(1) * Wtr[i]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float((((model(Xva) - Yva) ** 2).mean(1) * Wva).mean())
            tl = float((((model(Xtr[:20000]) - Ytr[:20000]) ** 2).mean(1) * Wtr[:20000]).mean())
        sched.step(vl)
        hist.append((ep, tl, vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"epoch {ep} train {tl:.5f} val {vl:.5f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        P = np.concatenate([model(t(Xn[a:a + 65536])).cpu().numpy() for a in range(0, len(Xn), 65536)])
    np.savez_compressed(args.out, h_pred=P[:, :6].reshape(-1, 2, 3), hh_pred=P[:, 6:].reshape(-1, 3, 3),
                        uid=d["uid"], proc=proc, history=np.array(hist), columns=np.array(cols))
    # quick quality summary on split 2 true H/Z events
    ev = (split == 2) & valid & (is_h | is_z)
    corr = [float(np.corrcoef(Y[ev, j], P[ev, j])[0, 1]) for j in range(6)]
    summary = {"n_train": int(len(tr)), "n_val": int(len(va)), "epochs": len(hist), "best_val": best,
               "corr_eval_h": corr, "device": dev, "dropped": args.drop}
    print(json.dumps(summary))
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
