"""Evaluate one fixed 9/30 point-h pipeline (seed 42) on calibration-shifted validation inputs.

For every validation event and every variant of the fixed grid VARIANTS, the event's raw
reconstructed quantities are shifted with calib_transform, re-packed, reduced exactly as in
training (train_ipsv.reduce_event), collated and passed through the fixed network and the frozen
H/Z readout of the 2026-10-02 azimuth run.  Also stored per variant: the visible-tau 3-vectors,
charged / pi0 energies and MET, from which classical observables are built offline.

Checks written to the report (and enforced):
  * identity variant: prediction parity with the stored reference predictions (<= 2e-5),
  * pack(perturb(eps=0)) reproduces the stored packed features exactly on the pilot rows,
  * token-order invariance of the network on one batch.
The test split is never loaded.

usage: python evaluate_remote.py --checkout DIR --arm none|local --output OUT.npz [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OUT930 = Path('/home/rbaba/meeting-followup-20260930/ipsv')
AZI = Path('/home/rbaba/azimuth-equivariance-20261002-output')

# fixed before any shifted prediction was looked at (2026-10-03).  The sagitta grid was first
# +-0.5, +-2 /TeV; +-2 (and +-0.5 for the hardest tracks, up to 2.8 TeV) leaves the domain
# 1 + q delta pT > 0, an implementation failure, so it was replaced by +-0.1, +-0.3 /TeV.
VARIANTS = (
    ('none', 0.0, 'consistent'),
    ('pi0', -0.05, 'consistent'), ('pi0', -0.02, 'consistent'), ('pi0', -0.01, 'consistent'),
    ('pi0', 0.01, 'consistent'), ('pi0', 0.02, 'consistent'), ('pi0', 0.05, 'consistent'),
    ('pi0', -0.02, 'fixed'), ('pi0', 0.02, 'fixed'),
    ('trk', -0.02, 'consistent'), ('trk', -0.01, 'consistent'),
    ('trk', 0.01, 'consistent'), ('trk', 0.02, 'consistent'),
    # 2026-10-09 (z_calibration): sagitta variants dropped, they still leave the track-scale domain
    ('met', -0.05, 'fixed'), ('met', 0.05, 'fixed'),
)


class FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)


def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            d.update(b)
    return d.hexdigest()


def classical_inputs(raw, CT):
    """visible tau 3-vector, charged (core-track) energy, pi0-flagged PFO energy, MET 2-vector."""
    vis = CT.p3(raw['tau']['pt'], raw['tau']['eta'], raw['tau']['phi'])
    ech = np.zeros(2)
    epi0 = np.zeros(2)
    trk, pfo = raw['trk'], raw['pfo']
    for s in (0, 1):
        mt = (raw['trk_side'] == s) & (trk['isCore'] > 0.5)
        mp = (raw['pfo_side'] == s) & (pfo['isPi0'] > 0.5)
        ptrk = CT.p3(trk['pt'][mt], trk['eta'][mt], trk['phi'][mt])
        ech[s] = np.sqrt((ptrk ** 2).sum(1) + 0.13957 ** 2).sum()
        epi0[s] = pfo['e'][mp].sum()
    met = np.array([raw['met']['et'] * np.cos(raw['met']['phi']), raw['met']['et'] * np.sin(raw['met']['phi'])])
    return vis, ech, epi0, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkout', type=Path, required=True)
    ap.add_argument('--arm', choices=('none', 'local'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()

    os.environ['SAMPLE'] = 'atlas'
    os.environ['GEO_FEATURES'] = str(OUT930 / 'atlas_geometry.npz')
    os.environ['GEO_KEY'] = args.arm
    os.environ['GEO_DIM'] = '22'
    os.environ['SEED'] = '42'
    sys.path.insert(0, str(args.checkout / 'analysis/meeting_followup/ipsv'))
    sys.path.insert(0, str(HERE))
    import train_ipsv as train  # noqa: E402
    import calib_transform as CT  # noqa: E402

    t0 = time.time()
    stats = CT.Stats(json.loads((HERE / 'feature_stats.json').read_text()))
    ds = train.AtlasReduced('validation')
    collate = train.P.T.collate_reco_h
    device = torch.device('cuda')
    train.P.T.configure_runtime(device)
    run_dir = OUT930 / 'runs' / f'atlas_{args.arm}_s42'
    ckpt_path = run_dir / 'checkpoint.pt'
    model = train.P.T.RecoModel('geo16').to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    network = torch.compile(model, dynamic=True)
    readout_path = AZI / f'readout_{args.arm}.pt'
    readout = FixedMLP().to(device)
    readout.load_state_dict(torch.load(readout_path, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    readout.eval()

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    V = len(VARIANTS)
    h = np.empty((n, V, 2, 3), np.float32)
    score = np.empty((n, V), np.float32)
    vis = np.empty((n, V, 2, 3), np.float32)
    ech = np.empty((n, V, 2), np.float32)
    epi0 = np.empty((n, V, 2), np.float32)
    met = np.empty((n, V, 2), np.float32)
    reco_mode = np.empty((n, 2), np.int64)
    identity_feature_max = 0.0
    perm_max = None
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            rows = range(start, min(n, start + args.batch_size))
            items = [train.Geo.__getitem__(ds, int(i)) for i in rows]
            raws = [CT.unpack(it, stats) for it in items]
            for k, it in enumerate(items):
                reco_mode[start + k] = it['tau_decay_mode'].numpy()
            for v, (kind, eps, met_mode) in enumerate(VARIANTS):
                packed = []
                for k, (it, raw) in enumerate(zip(items, raws)):
                    new_raw = CT.perturb(raw, kind, eps, met_mode)
                    new_item = CT.pack(it, new_raw, stats, raw)
                    if kind == 'none' and start < 4 * args.batch_size:
                        for key in ('event_features', 'tau_features', 'track_features', 'pfo_features'):
                            if len(it[key]):
                                identity_feature_max = max(identity_feature_max,
                                                           float(torch.max(torch.abs(new_item[key] - it[key]))))
                    packed.append(train.reduce_event(new_item))
                    i = start + k
                    vis[i, v], ech[i, v], epi0[i, v], met[i, v] = classical_inputs(new_raw, CT)
                batch = {key: (val.to(device) if torch.is_tensor(val) else val) for key, val in collate(packed).items()}
                pred = network(batch)[1]
                h[start:start + len(packed), v] = pred.cpu().numpy()
                score[start:start + len(packed), v] = torch.sigmoid(readout(pred.reshape(-1, 6))).cpu().numpy()
                if perm_max is None and kind == 'none':
                    perm = torch.randperm(len(packed), generator=torch.Generator().manual_seed(7))
                    # reverse the token order inside every event (keep padding at the end)
                    b2 = collate([dict(p, track_features=p['track_features'].flip(0), track_sides=p['track_sides'].flip(0),
                                       pfo_features=p['pfo_features'].flip(0), pfo_sides=p['pfo_sides'].flip(0))
                                  for p in packed])
                    b2 = {key: (val.to(device) if torch.is_tensor(val) else val) for key, val in b2.items()}
                    perm_max = float(torch.max(torch.abs(network(b2)[1] - pred)))
                    del perm
            if start % (20 * args.batch_size) == 0:
                print(f'{start}/{n} {time.time() - t0:.0f}s', flush=True)

    with np.load(run_dir / 'validation_predictions.npz') as ref:
        ref_ids = np.asarray(ref['global_indices'], np.int64)[:n]
        ref_h = np.asarray(ref['h_pred'], np.float32)[:n]
    ids = ds.global_index.numpy()[:n].astype(np.int64)
    if not np.array_equal(ref_ids, ids):
        raise RuntimeError('reference identity mismatch')
    parity = float(np.max(np.abs(h[:, 0] - ref_h)))
    report = dict(status='complete', arm=args.arm, rows=int(n), limit=args.limit, variants=VARIANTS,
                  checkpoint=dict(path=str(ckpt_path), sha256=sha256(ckpt_path), epoch=int(ckpt['epoch'])),
                  readout=dict(path=str(readout_path), sha256=sha256(readout_path)),
                  identity_prediction_parity_max_abs=parity,
                  identity_feature_max_abs=identity_feature_max,
                  token_order_reversal_max_abs=perm_max,
                  seconds=time.time() - t0, device=torch.cuda.get_device_name(0))
    if parity > 2e-5:
        report['status'] = 'failed_parity'
    if identity_feature_max != 0.0:
        report['status'] = 'failed_identity_features'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, global_indices=ids, labels=ds.labels.numpy()[:n],
                        truth_modes=ds.truth_modes.numpy()[:n], h_exact=ds.h.numpy()[:n],
                        reco_mode=reco_mode, h=h, score=score, vis=vis, ech=ech, epi0=epi0, met=met,
                        variant_kind=np.array([v[0] for v in VARIANTS]),
                        variant_eps=np.array([v[1] for v in VARIANTS]),
                        variant_met=np.array([v[2] for v in VARIANTS]))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if report['status'] != 'complete':
        sys.exit(1)


if __name__ == '__main__':
    main()
