"""Fixed-recipe H/Z readout on variants of the point-h output.

Same MLP (64-64), optimiser, 50 epochs, seed and batch order as
``fixed_readout_parallel.py``; only the input vector changes:
  --pred-dirs A [B C]  average h_pred over several trained models (seed ensemble)
  --extra none|quality|quality_only
      quality = per tau: log(|IP_lead|/50um), SV available, core-track count/3,
      |IP| of tracks 2,3 (from the 16-channel geometry file) -- 10 numbers.
      quality_only drops h and is a shortcut control.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TARGETS = Path("/home/rbaba/reco-h-supervision-20260906-targets")


def load_pred(dirs, split, ids):
    arrs = []
    for d in dirs:
        with np.load(Path(d) / f"{split}_predictions.npz") as z:
            if not np.array_equal(np.asarray(z["global_indices"]), ids):
                raise RuntimeError(f"identity mismatch {d} {split}")
            arrs.append(np.asarray(z["h_pred"], np.float32).reshape(len(ids), 6))
    return np.mean(arrs, axis=0)


def quality(geo, split, ids):
    src = geo[f"{split}_global_indices"]
    pos = {int(v): i for i, v in enumerate(src)}
    f = np.asarray(geo[f"{split}_full"])[[pos[int(v)] for v in ids]]  # N,2,16
    q = np.stack([f[..., 2], f[..., 15], (f[..., 3] + f[..., 7] + f[..., 11]) / 3,
                  f[..., 6], f[..., 10]], -1)
    return q.reshape(len(ids), 10).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dirs", nargs="+", required=True)
    p.add_argument("--extra", choices=("none", "quality", "quality_only"), default="none")
    p.add_argument("--geometry", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    xs, ys, ids_all = {}, {}, {}
    geo = np.load(a.geometry) if a.extra != "none" else None
    for split in ("train", "validation"):
        with np.load(TARGETS / f"{split}.npz") as t:
            v = np.asarray(t["h_valid"], bool)
            ids = np.asarray(t["global_indices"])[v]
            ys[split] = np.asarray(t["labels"])[v].astype(np.float32)
        parts = []
        if a.extra != "quality_only":
            parts.append(load_pred(a.pred_dirs, split, ids))
        if a.extra != "none":
            parts.append(quality(geo, split, ids))
        xs[split] = np.concatenate(parts, 1)
        ids_all[split] = ids
    dim = xs["train"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    model = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                          nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=device.type == "cuda")
    ds = TensorDataset(torch.from_numpy(xs["train"]), torch.from_numpy(ys["train"]))
    for epoch in range(50):
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(20260916 + epoch))
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
    print(a.name, dim, float(s.mean()))


if __name__ == "__main__":
    main()
