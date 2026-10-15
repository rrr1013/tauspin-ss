"""Misspecification stress of the SV response: degrade the evaluation sample only.

The stored SV of every tau is smeared once more so that the total resolution is twice the
reco.py value (10 um transverse, 250 um along the flight direction -> 20 / 500 um), with
the true tau direction as the flight axis (as reco.smear_sv).  Writes
  <out>/reco_sv2/trainHU_run_*.npz   reco chunks with the degraded SV (everything else identical)
  <out>/dataset_HU_sv2.npz           dataset_HU with spin_sv_{n,r,L}_* recomputed as in
                                     build_dataset.features (side 0 = tau- by reco charge)
The classical posterior is then rerun with the likelihood unchanged (10 um, misspecified) or
updated (20 um); the learned regressor of 10/07 is applied unchanged to the degraded features."""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dihiggs_spin"))
from build_dataset import local_basis  # noqa: E402

E = "/home/rbaba/dihiggs-spin-20261006/runs/ent"
out = Path(sys.argv[1])
(out / "reco_sv2").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261016)
k = np.sqrt(2 ** 2 - 1)
svs = []
for f in sorted(glob.glob(f"{E}/reco/trainHU_run_*.npz")):
    d = dict(np.load(f, allow_pickle=True))
    u = d["tau_p4"][..., :3] / np.linalg.norm(d["tau_p4"][..., :3], axis=-1, keepdims=True)
    g = rng.standard_normal(d["sv"].shape) * 0.010 * k
    g = g - (g * u).sum(-1, keepdims=True) * u
    d["sv"] = d["sv"] + g + u * (0.250 * k * rng.standard_normal(d["sv"].shape[:-1]))[..., None]
    np.savez_compressed(out / "reco_sv2" / Path(f).name, **d)
    svs.append(d)
cat = lambda key: np.concatenate([p[key] for p in svs])
sv, q, vis, nch = cat("sv"), cat("q"), cat("vis"), cat("nch")
swap = q[:, 0] > 0
S = lambda a: np.where(swap.reshape((-1,) + (1,) * (a.ndim - 1)), a[:, ::-1], a)
sv_s, vis_s, nch_s = S(sv), S(vis), S(nch)
ds = dict(np.load(f"{E}/dataset_HU.npz", allow_pickle=True))
for s in (0, 1):
    n_, r_, k_ = local_basis(vis_s[:, s])
    L = np.linalg.norm(sv_s[:, s], axis=-1)
    three = nch_s[:, s] >= 3
    new_n = np.where(three, (sv_s[:, s] * n_).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0.0)
    print("side", s, "max |old-new| on 1-prong (must be 0):", float(np.abs(ds[f"spin_sv_n_{s}"][~three] - new_n[~three]).max()))
    ds[f"spin_sv_n_{s}"] = new_n
    ds[f"spin_sv_r_{s}"] = np.where(three, (sv_s[:, s] * r_).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0.0)
    ds[f"spin_sv_L_{s}"] = np.where(three, L, 0.0)
np.savez_compressed(out / "dataset_HU_sv2.npz", **ds)
print("wrote", out)
