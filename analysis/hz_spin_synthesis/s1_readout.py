"""Fixed-recipe H/Z readout with a choice of training target and extra inputs.

Same MLP (64-64 GELU), AdamW 1e-3, 50 epochs, batch 512, seed and batch order as
``fixed_readout_parallel.py``.  What can change:

  --target hard   BCE against the H/Z label (the existing fixed readout)
  --target soft   BCE against the analytic spin weight s = w_H / (w_H + w_Z)
                  of the TRAIN event, w_X = 1 + h-^T C_X h+ with the exact truth
                  h, C_H = diag(1,1,-1), C_Z = diag(0,0,1).  Under a 50/50
                  mixture E[s | x] is p(H | x) whenever H and Z differ only in
                  the spin factor, so the soft target removes Bernoulli label
                  noise without changing the optimum.
  --pred-dirs     one or more point-h runs; h_pred is averaged (seed ensemble).
  --no-h          drop h_pred (control for --extra).
  --extra NPZ:KEY per-event feature arrays keyed by global index, for both
                  splits (``train_global_indices``/``train_<KEY>`` and
                  ``validation_...``); standardised with train mean/std.

Validation scores are written for the h-valid validation rows; test is never
loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TARGETS = Path("/home/rbaba/reco-h-supervision-20260906-targets")
C_H = np.array([1.0, 1.0, -1.0])
C_Z = np.array([0.0, 0.0, 1.0])


def load_pred(dirs, split, ids):
    arrs = []
    for d in dirs:
        with np.load(Path(d) / f"{split}_predictions.npz") as z:
            if not np.array_equal(np.asarray(z["global_indices"]), ids):
                raise RuntimeError(f"identity mismatch {d} {split}")
            arrs.append(np.asarray(z["h_pred"], np.float64).reshape(len(ids), 6))
    return np.mean(arrs, axis=0)


def spin_weight(h):
    prod = h[:, 0] * h[:, 1]
    w_h = 1.0 + prod @ C_H
    w_z = 1.0 + prod @ C_Z
    return w_h / (w_h + w_z)


def extra_features(spec, split, ids):
    path, key = spec.split(":")
    with np.load(path) as z:
        src = np.asarray(z[f"{split}_global_indices"])
        pos = {int(v): i for i, v in enumerate(src)}
        missing = [int(v) for v in ids if int(v) not in pos]
        if missing:
            raise RuntimeError(f"{len(missing)} {split} rows missing in {spec}")
        value = np.asarray(z[f"{split}_{key}"], np.float64)[[pos[int(v)] for v in ids]]
    return value.reshape(len(ids), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dirs", nargs="*", default=[])
    p.add_argument("--no-h", action="store_true")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--target", choices=("hard", "soft"), default="hard")
    p.add_argument("--rows", choices=("all", "extra"), default="all",
                   help="'extra': keep only rows present in every --extra file (train+val)")
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260916)
    a = p.parse_args()

    xs, ys, ids_all = {}, {}, {}
    for split in ("train", "validation"):
        with np.load(TARGETS / f"{split}.npz") as t:
            v = np.asarray(t["h_valid"], bool)
            ids = np.asarray(t["global_indices"])[v]
            labels = np.asarray(t["labels"])[v].astype(np.float64)
            h = np.asarray(t["h"], np.float64)[v]
        if a.rows == "extra":
            keep = np.ones(len(ids), bool)
            for spec in a.extra:
                path, _ = spec.split(":")
                with np.load(path) as z:
                    keep &= np.isin(ids, np.asarray(z[f"{split}_global_indices"]))
            ids, labels, h = ids[keep], labels[keep], h[keep]
        parts = []
        if not a.no_h:
            parts.append(load_pred(a.pred_dirs, split, ids))
        for spec in a.extra:
            parts.append(extra_features(spec, split, ids))
        xs[split] = np.concatenate(parts, 1)
        ys[split] = labels if (split == "validation" or a.target == "hard") else spin_weight(h)
        ids_all[split] = ids
    n_h = 0 if a.no_h else 6
    if xs["train"].shape[1] > n_h:  # standardise the extra block with train statistics
        mu = xs["train"][:, n_h:].mean(0)
        sd = xs["train"][:, n_h:].std(0) + 1e-9
        for split in xs:
            xs[split][:, n_h:] = (xs[split][:, n_h:] - mu) / sd
    for split in xs:
        if not np.isfinite(xs[split]).all():
            raise RuntimeError(f"non-finite input in {split}")
        xs[split] = xs[split].astype(np.float32)
    dim = xs["train"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                          nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,
                            fused=device.type == "cuda")
    ds = TensorDataset(torch.from_numpy(xs["train"]),
                       torch.from_numpy(ys["train"].astype(np.float32)))
    for epoch in range(50):
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(a.seed + epoch))
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        s = torch.sigmoid(model(torch.from_numpy(xs["validation"]).to(device)).squeeze(-1)).cpu().numpy()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output_dir / f"readout_scores_{a.name}.npz",
                        global_indices=ids_all["validation"], scores=s.astype(np.float64))
    print(a.name, a.target, dim, len(ids_all["train"]), len(s), float(s.mean()), flush=True)


if __name__ == "__main__":
    main()
