"""Input interventions on the fixed main-line learner (gen3pi teacher, p11 full22, seed 42)
on reco-rho sides (rmode == 1) and reco-pi sides (rmode == 0) for side-by-side comparison.

Does the learner's transverse polarimeter on rho come from the charged pion IP direction,
the visible decay plane (PFO), or both?

Variants:
  identity          stored inputs; parity check with validation_predictions.npz
  shuffle_own_rho   the 22 geometry channels of reco-rho sides permuted among rho sides
  shuffle_own_pi    the 22 geometry channels of reco-pi sides permuted among pi sides
  rot90_rho         (u.n, u.r) -> (-u.r, u.n) on reco-rho sides (IP rotated +90 deg)
  rot90_pi          (u.n, u.r) -> (-u.r, u.n) on reco-pi sides
  flip_rho          (u.n, u.r) -> -(u.n, u.r) on reco-rho sides
  flip_pi           (u.n, u.r) -> -(u.n, u.r) on reco-pi sides
  ipmag_const_rho   log|IP| -> median over reco-rho sides
  ipmag_const_pi    log|IP| -> median over reco-pi sides
  tok_d0z0_zero_rho track token d0 and z0 sin(theta) set to 0 on reco-rho sides
  tok_d0z0_zero_pi  track token d0 and z0 sin(theta) set to 0 on reco-pi sides
  pfo_zero_rho      neutral PFO features set to 0 on reco-rho sides
  pfo_shuffle_rho   neutral PFO tokens permuted among reco-rho events
"""
import argparse
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

import train_point_h as TP  # noqa: E402

P = TP.P
T = P.T
CKPT = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/p11/full22_s42')
LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', default=2048, type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    ds = T.RecoHDataset('validation')
    model = P.M.RecoModel('geo16').to(device)
    state = torch.load(CKPT / 'checkpoint.pt', map_location='cpu')['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    geo0 = ds.geometry_tau.clone()
    ids = ds.global_index.numpy()
    with np.load(LADDER) as d:
        look = {int(g): i for i, g in enumerate(d['global_indices'])}
        r = np.array([look[int(g)] for g in ids])
        rmode = np.where(d['reco_h_available_side'], d['reco_h_mode'], -1)[r]
    ncore = np.asarray(np.load('/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz')['n_core'])[ids]
    rng = np.random.default_rng(20261024)

    def run():
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=T.collate_reco_h)
        _, pred = T.evaluate(model, loader, device)
        return pred

    res = {}
    print('Evaluating identity...')
    base = run()
    ref = np.load(CKPT / 'validation_predictions.npz')
    assert np.array_equal(base['global_indices'], ref['global_indices'])
    parity = float(np.max(np.abs(base['h_pred'] - ref['h_pred'])))
    print(f'Parity max abs error: {parity:.6e}')
    res['identity'] = base['h_pred']

    # 1. Geometry shuffle on own side
    for mode_name, m_val in (('rho', 1), ('pi', 0)):
        print(f'Evaluating shuffle_own_{mode_name}...')
        g = geo0.clone()
        for s in (0, 1):
            mask = (rmode[:, s] == m_val)
            w = np.flatnonzero(mask)
            if len(w) > 1:
                perm = w[rng.permutation(len(w))]
                g[w, s] = geo0[perm, s]
        ds.geometry_tau = g
        res[f'shuffle_own_{mode_name}'] = run()['h_pred']
        ds.geometry_tau = geo0

    # 2. IP direction edits (rot90, flip, ipmag_const)
    for mode_name, m_val in (('rho', 1), ('pi', 0)):
        m_mask = torch.as_tensor(rmode == m_val)
        med = float(torch.median(geo0[:, :, 2][m_mask]))
        for edit_name in ('rot90', 'flip', 'ipmag_const', 'offset_zero'):
            print(f'Evaluating {edit_name}_{mode_name}...')
            g = geo0.clone()
            blk = g[:, :, 0:6].clone()
            if edit_name == 'rot90':
                blk[..., 0], blk[..., 1] = -geo0[:, :, 1], geo0[:, :, 0]
            elif edit_name == 'flip':
                blk[..., 0:2] = -geo0[:, :, 0:2]
            elif edit_name == 'ipmag_const':
                blk[..., 2] = med
            elif edit_name == 'offset_zero':
                blk[..., 4:6] = 0.0
            g[:, :, 0:6] = torch.where(m_mask[..., None], blk, geo0[:, :, 0:6])
            ds.geometry_tau = g
            res[f'{edit_name}_{mode_name}'] = run()['h_pred']
            ds.geometry_tau = geo0

    # 3. Track token d0/z0 zeroed
    saved_tracks = {k: v['track_features'].clone() for k, v in ds._data.items()}
    for mode_name, m_val in (('rho', 1), ('pi', 0)):
        print(f'Evaluating tok_d0z0_zero_{mode_name}...')
        for i in range(len(ids)):
            if not (rmode[i] == m_val).any():
                continue
            d_shard = ds._data[ds._sample[i]]
            pos = int(ds._row_position[i])
            a, b = int(d_shard['track_offsets'][pos]), int(d_shard['track_offsets'][pos + 1])
            sides = d_shard['track_sides'][a:b].numpy()
            for sd in np.flatnonzero(rmode[i] == m_val):
                rows_ = a + np.flatnonzero(sides == sd)
                d_shard['track_features'][rows_, 5] = 0.0
                d_shard['track_features'][rows_, 6] = 0.0
        res[f'tok_d0z0_zero_{mode_name}'] = run()['h_pred']
        for k, v in saved_tracks.items():
            ds._data[k]['track_features'] = v.clone()

    # 4. Neutral PFO zeroed and shuffled on rho
    saved_pfo = {k: v['pfo_features'].clone() for k, v in ds._data.items()}
    print('Evaluating pfo_zero_rho...')
    for i in range(len(ids)):
        if not (rmode[i] == 1).any():
            continue
        d_shard = ds._data[ds._sample[i]]
        pos = int(ds._row_position[i])
        a, b = int(d_shard['pfo_offsets'][pos]), int(d_shard['pfo_offsets'][pos + 1])
        sides = d_shard['pfo_sides'][a:b].numpy()
        for sd in np.flatnonzero(rmode[i] == 1):
            rows_ = a + np.flatnonzero(sides == sd)
            d_shard['pfo_features'][rows_] = 0.0
    res['pfo_zero_rho'] = run()['h_pred']
    for k, v in saved_pfo.items():
        ds._data[k]['pfo_features'] = v.clone()

    print('Evaluating pfo_shuffle_rho...')
    # Shuffle PFO features among rho events
    # Gather all PFO indices belonging to rho sides
    rho_pfo_locs = []
    for i in range(len(ids)):
        if not (rmode[i] == 1).any():
            continue
        s_name = ds._sample[i]
        pos = int(ds._row_position[i])
        d_shard = ds._data[s_name]
        a, b = int(d_shard['pfo_offsets'][pos]), int(d_shard['pfo_offsets'][pos + 1])
        sides = d_shard['pfo_sides'][a:b].numpy()
        for sd in np.flatnonzero(rmode[i] == 1):
            rows_ = a + np.flatnonzero(sides == sd)
            for r_ in rows_:
                rho_pfo_locs.append((s_name, r_))

    if len(rho_pfo_locs) > 1:
        perm_idx = rng.permutation(len(rho_pfo_locs))
        # Extract features
        feats = [ds._data[s_name]['pfo_features'][r_].clone() for s_name, r_ in rho_pfo_locs]
        for idx_to, idx_from in enumerate(perm_idx):
            s_name, r_ = rho_pfo_locs[idx_to]
            ds._data[s_name]['pfo_features'][r_] = feats[idx_from]
        res['pfo_shuffle_rho'] = run()['h_pred']
        for k, v in saved_pfo.items():
            ds._data[k]['pfo_features'] = v.clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, global_indices=base['global_indices'], **res)
    meta = {'parity_max_abs': parity, 'rows': int(len(ids)), 'keys': list(res.keys())}
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    print('Done. Saved to', args.output)


if __name__ == '__main__':
    main()
