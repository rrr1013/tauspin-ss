"""How much of the H/Z readout is single-tau (marginal) information?

Additive readout logit = f(h-) + g(h+): it can use each tau's own h (polarisation,
acceptance-shaped marginals, production kinematics that leak into one h) but no
product of the two, i.e. no spin correlation.  Same optimiser, epochs, batch and
seeds as the fixed readout.  Input: exact truth h (--source exact) or the
point-h prediction of one or more runs (--pred-dirs).  Test is never loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from s1_readout import TARGETS, load_pred


class Additive(nn.Module):
    def __init__(self):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.f, self.g = mk(), mk()

    def forward(self, x):
        return (self.f(x[:, :3]) + self.g(x[:, 3:])).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("exact", "pred"), required=True)
    p.add_argument("--pred-dirs", nargs="*", default=[])
    p.add_argument("--model", choices=("additive", "full"), default="additive")
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    xs, ys, ids_all = {}, {}, {}
    for split in ("train", "validation"):
        with np.load(TARGETS / f"{split}.npz") as t:
            v = np.asarray(t["h_valid"], bool)
            ids = np.asarray(t["global_indices"])[v]
            ys[split] = np.asarray(t["labels"])[v].astype(np.float32)
            h = np.asarray(t["h"], np.float32)[v].reshape(len(ids), 6)
        xs[split] = h if a.source == "exact" else load_pred(a.pred_dirs, split, ids).astype(np.float32)
        ids_all[split] = ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260916)
    model = Additive() if a.model == "additive" else nn.Sequential(
        nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1), nn.Flatten(0))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=device.type == "cuda")
    ds = TensorDataset(torch.from_numpy(xs["train"]), torch.from_numpy(ys["train"]))
    for epoch in range(50):
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(20260916 + epoch))
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        s = torch.sigmoid(model(torch.from_numpy(xs["validation"]).to(device))).cpu().numpy()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output_dir / f"readout_scores_{a.name}.npz",
                        global_indices=ids_all["validation"], scores=s.astype(np.float64))
    print(a.name, float(s.mean()))


if __name__ == "__main__":
    main()
