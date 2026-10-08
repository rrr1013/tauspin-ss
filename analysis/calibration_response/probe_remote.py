"""Probe the packed validation features before writing the calibration transform.

Checks, on a few thousand validation events of the 9/30 ATLAS cohort:
  * relative features (dEta, dPhi, ptFraction) against the absolute track/PFO/tau values,
  * event features against the absolute tau/MET values,
  * the visible-tau 3-momentum against sum(core tracks) + sum(pi0-flagged PFOs),
  * PFO content (pi0 flag, distance to the nearest track) per reco decay mode.

usage (lxgpu02, .venv-gpu): python probe_remote.py --checkout DIR --stats feature_stats.json --limit 3000
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkout', type=Path, required=True)
    ap.add_argument('--stats', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=3000)
    args = ap.parse_args()

    os.environ['SAMPLE'] = 'atlas'
    os.environ['GEO_FEATURES'] = '/home/rbaba/meeting-followup-20260930/ipsv/atlas_geometry.npz'
    os.environ['GEO_KEY'] = 'none'
    os.environ['GEO_DIM'] = '22'
    os.environ['SEED'] = '42'
    sys.path.insert(0, str(args.checkout / 'analysis/meeting_followup/ipsv'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_ipsv as train  # noqa: E402
    import calib_transform as CT  # noqa: E402

    stats = CT.Stats(json.loads(args.stats.read_text()))
    ds = train.AtlasReduced('validation')
    rows = np.arange(min(args.limit, len(ds)))
    out = {k: [] for k in ('trk_deta', 'trk_dphi', 'trk_frac', 'pfo_deta', 'pfo_dphi', 'pfo_frac',
                           'ev', 'sum_vs_tau', 'n_pfo', 'n_pi0', 'mode', 'pfo_dr_trk', 'pfo_pi0')}
    for i in rows:
        item = train.Geo.__getitem__(ds, int(i))   # unreduced
        raw = CT.unpack(item, stats)
        tau = raw['tau']
        for s in (0, 1):
            m = raw['trk_side'] == s
            t = raw['trk'][m]
            out['trk_deta'].append(np.abs(t['dEta'] - (t['eta'] - tau['eta'][s])).max(initial=0))
            out['trk_dphi'].append(np.abs(CT.wrap(t['dPhi'] - (t['phi'] - tau['phi'][s]))).max(initial=0))
            out['trk_frac'].append(np.abs(t['ptFraction'] - t['pt'] / tau['pt'][s]).max(initial=0)
                                   / max(1e-9, np.abs(t['ptFraction']).max(initial=1e-9)))
            m2 = raw['pfo_side'] == s
            p = raw['pfo'][m2]
            out['pfo_deta'].append(np.abs(p['dEta'] - (p['eta'] - tau['eta'][s])).max(initial=0))
            out['pfo_dphi'].append(np.abs(CT.wrap(p['dPhi'] - (p['phi'] - tau['phi'][s]))).max(initial=0))
            out['pfo_frac'].append(np.abs(p['ptFraction'] - p['pt'] / tau['pt'][s]).max(initial=0)
                                   / max(1e-9, np.abs(p['ptFraction']).max(initial=1e-9)))
            core = t[t['isCore'] > 0.5]
            pi0 = p[p['isPi0'] > 0.5]
            vec = CT.p3(core['pt'], core['eta'], core['phi']).sum(0) + CT.p3(pi0['pt'], pi0['eta'], pi0['phi']).sum(0)
            tv = CT.p3(tau['pt'][s:s + 1], tau['eta'][s:s + 1], tau['phi'][s:s + 1])[0]
            out['sum_vs_tau'].append([np.linalg.norm(vec) / np.linalg.norm(tv),
                                      np.arccos(np.clip(vec @ tv / np.linalg.norm(vec) / np.linalg.norm(tv), -1, 1))])
            out['n_pfo'].append(len(p))
            out['n_pi0'].append(len(pi0))
            out['mode'].append(int(item['tau_decay_mode'][s]))
            if len(p) and len(t):
                pv = CT.p3(p['pt'], p['eta'], p['phi'])
                tv3 = CT.p3(t['pt'], t['eta'], t['phi'])
                cosang = (pv / np.linalg.norm(pv, axis=1, keepdims=True)) @ (tv3 / np.linalg.norm(tv3, axis=1, keepdims=True)).T
                out['pfo_dr_trk'].extend(np.arccos(np.clip(cosang.max(1), -1, 1)).tolist())
                out['pfo_pi0'].extend(p['isPi0'].tolist())
        ev_rebuilt = CT.event_features_raw(raw)
        out['ev'].append(np.abs(ev_rebuilt - raw['event_raw']).max())
    res = {}
    for k in ('trk_deta', 'trk_dphi', 'trk_frac', 'pfo_deta', 'pfo_dphi', 'pfo_frac', 'ev'):
        a = np.asarray(out[k])
        res[k] = dict(max=float(a.max()), p99=float(np.quantile(a, 0.99)), median=float(np.median(a)))
    sv = np.asarray(out['sum_vs_tau'])
    mode = np.asarray(out['mode'])
    res['sum_vs_tau'] = {int(m): dict(n=int((mode == m).sum()),
                                      pt_ratio_median=float(np.median(sv[mode == m, 0])),
                                      pt_ratio_p16_p84=np.quantile(sv[mode == m, 0], [0.16, 0.84]).tolist(),
                                      angle_median_mrad=float(1e3 * np.median(sv[mode == m, 1])),
                                      n_pi0_mean=float(np.mean(np.asarray(out['n_pi0'])[mode == m])),
                                      n_pfo_mean=float(np.mean(np.asarray(out['n_pfo'])[mode == m])))
                         for m in np.unique(mode)}
    dr = np.asarray(out['pfo_dr_trk'])
    pi0 = np.asarray(out['pfo_pi0'])
    res['pfo_nearest_track_angle_mrad'] = dict(
        pi0_median=float(1e3 * np.median(dr[pi0 > 0.5])), nonpi0_median=float(1e3 * np.median(dr[pi0 < 0.5])),
        frac_below_1mrad_nonpi0=float(np.mean(dr[pi0 < 0.5] < 1e-3)), frac_pi0=float(np.mean(pi0 > 0.5)))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
