"""Input interventions on the fixed main-line learner (gen3pi teacher, p11 full22, seed 42): does the h of one tau use
the geometry (IP / SV) of the other tau?

Variants (validation only, the selected checkpoint, nothing retrained):
  identity        stored inputs; must reproduce validation_predictions.npz (parity check)
  shuffle_other_s the 22 geometry channels of side 1-s permuted among events whose side 1-s has the same reco mode and
                  core-track multiplicity (as p10 full_shuffle, but one side only); side s keeps its own.  The h of
                  side s is then read.  Done for s = 0 and s = 1.
  shuffle_own_s   the same for side s itself (control: what the own geometry is worth)
  --own-edits     on reco-pi sides only (both sides at once), track-0 block of the 22 channels
                  (u.n, u.r, log|IP|, avail, t.n/0.01, t.r/0.01):
    rot90         (u.n, u.r) -> (-u.r, u.n): the measured IP direction rotated by +90 deg about k
    flip          (u.n, u.r) -> -(u.n, u.r)
    ipmag_const   log|IP| -> its median over reco-pi sides (direction kept)
    offset_zero   (t.n, t.r) -> 0 (track direction put on the visible axis)
  --token-edits   on reco-pi sides only: the per-track token columns d0 (5) and z0 sin(theta) (6) of every track of that side
                  set to 0 (stored, standardised value), geometry kept: tok_d0z0_zero
Output: npz with h_pred per variant (rows in the stored validation order), json with the parity check.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('GEO_DIM', '22')
os.environ.setdefault('GEO_KEY', 'full')
os.environ.setdefault('GEO_FEATURES', '/home/rbaba/hz-beyond-ceiling-20260923/artifacts/geo_full22.npz')
os.environ.setdefault('H_TARGETS_DIR', '/home/rbaba/gen3pi-teacher-20260924/artifacts/targets')
os.environ.setdefault('P11_DIR', '/home/rbaba/hz-beyond-ceiling-20260923/scripts')
os.environ.setdefault('SEED', '42')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_point_h as TP  # noqa: E402  (patches the dataset class: gen3pi targets, 22 geometry channels)

P = TP.P
T = P.T
CKPT = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/p11/full22_s42')
LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')


def main():
    out = Path(sys.argv[1])
    device = torch.device('cuda')
    ds = T.RecoHDataset('validation')
    model = P.M.RecoModel('geo16').to(device)
    state = torch.load(CKPT / 'checkpoint.pt', map_location='cpu')['model_state_dict']
    model.load_state_dict(state)
    geo0 = ds.geometry_tau.clone()
    ids = ds.global_index.numpy()
    with np.load(LADDER) as d:
        look = {int(g): i for i, g in enumerate(d['global_indices'])}
        r = np.array([look[int(g)] for g in ids])
        rmode = np.where(d['reco_h_available_side'], d['reco_h_mode'], -1)[r]
    ncore = np.asarray(np.load('/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz')['n_core'])[ids]
    rng = np.random.default_rng(20261023)

    def run():
        loader = DataLoader(ds, batch_size=2048, num_workers=4, collate_fn=T.collate_reco_h)
        _, pred = T.evaluate(model, loader, device)
        return pred

    res = {}
    base = run()
    ref = np.load(CKPT / 'validation_predictions.npz')
    assert np.array_equal(base['global_indices'], ref['global_indices'])
    parity = float(np.max(np.abs(base['h_pred'] - ref['h_pred'])))
    res['identity'] = base['h_pred']
    for s in (0, 1):
        for name, side in (('other', 1 - s), ('own', s)):
            key = rmode[:, side] * 10 + np.minimum(ncore[:, side], 3)
            perm = np.arange(len(key))
            for k in np.unique(key):
                w = np.flatnonzero(key == k)
                perm[w] = w[rng.permutation(len(w))]
            g = geo0.clone()
            g[:, side] = geo0[perm, side]
            ds.geometry_tau = g
            res[f'shuffle_{name}_{s}'] = run()['h_pred']
            ds.geometry_tau = geo0
    if '--own-edits' in sys.argv:
        pi = torch.as_tensor(rmode == 0)
        med = float(torch.median(geo0[:, :, 2][pi]))
        for name in ('rot90', 'flip', 'ipmag_const', 'offset_zero'):
            g = geo0.clone()
            blk = g[:, :, 0:6].clone()
            if name == 'rot90':
                blk[..., 0], blk[..., 1] = -geo0[:, :, 1], geo0[:, :, 0]
            elif name == 'flip':
                blk[..., 0:2] = -geo0[:, :, 0:2]
            elif name == 'ipmag_const':
                blk[..., 2] = med
            else:
                blk[..., 4:6] = 0.0
            g[:, :, 0:6] = torch.where(pi[..., None], blk, geo0[:, :, 0:6])
            ds.geometry_tau = g
            res[name] = run()['h_pred']
            ds.geometry_tau = geo0
    if '--token-edits' in sys.argv:
        saved = {k: v['track_features'].clone() for k, v in ds._data.items()}
        for i in range(len(ids)):
            if not (rmode[i] == 0).any():
                continue
            d = ds._data[ds._sample[i]]
            pos = int(ds._row_position[i])
            a, b = int(d['track_offsets'][pos]), int(d['track_offsets'][pos + 1])
            sides = d['track_sides'][a:b].numpy()
            for sd in np.flatnonzero(rmode[i] == 0):
                rows_ = a + np.flatnonzero(sides == sd)
                d['track_features'][rows_, 5] = 0.0
                d['track_features'][rows_, 6] = 0.0
        res['tok_d0z0_zero'] = run()['h_pred']
        for k, v in saved.items():
            ds._data[k]['track_features'] = v
    np.savez_compressed(out, global_indices=base['global_indices'], **res)
    out.with_suffix('.json').write_text(json.dumps({'parity_max_abs': parity, 'rows': int(len(ids))}, indent=1))
    print(json.dumps({'parity_max_abs': parity}))


if __name__ == '__main__':
    main()
