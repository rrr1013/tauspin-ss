"""Evaluate learner input interventions on rho and pi sides.
Calculates weighted correlation of (hn, hr, hk) with exact h for each intervention variant,
broken down by mode (rho, pi, 3pi), and outputs a summary JSON and markdown table.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN
from hquality import wcorr
import posterior as CP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--posterior-ref', default=Path('/home/rbaba/taukin-gap-20261023/out/val_track_dl/posterior.npz'), type=Path)
    args = parser.parse_args()

    I = np.load(args.input)
    ref = np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')
    hx = ref['h'].astype(float)
    P = np.load(args.posterior_ref)
    assert np.array_equal(P['global_indices'], I['global_indices'])

    w = P['weights'].astype(float)
    modes, rm, av = P['modes'], P['reco_h_mode'].astype(int), P['reco_h_available_side']
    W = np.broadcast_to(w[:, None], av.shape)

    modes_map = {
        'pi': (0, 0),
        'rho': (1, 1),
        '3pi': (3, 3)
    }

    results = {}
    print(f"{'Variant':22s} | {'pi hn':6s} {'pi hr':6s} {'pi hk':6s} | {'rho hn':6s} {'rho hr':6s} {'rho hk':6s} | {'3pi hn':6s} {'3pi hr':6s} {'3pi hk':6s}")
    print("-" * 88)

    for v in I.files:
        if v == 'global_indices':
            continue
        h = I[v]
        results[v] = {}
        row_str = f"{v:22s} | "
        for g in ('pi', 'rho', '3pi'):
            tm, rmode_val = modes_map[g]
            msk = av & (rm == rmode_val) & (modes == tm)
            corrs = [wcorr(h[..., c][msk], hx[..., c][msk], W[msk]) for c in range(3)]
            results[v][g] = {'hn': float(corrs[0]), 'hr': float(corrs[1]), 'hk': float(corrs[2]), 'sides': int(msk.sum())}
            row_str += f"{corrs[0]:.3f} {corrs[1]:.3f} {corrs[2]:.3f} | "
        print(row_str)

    # Detailed breakdown for rho by |IP| sextile
    tracks = dict(np.load(CP.TRACKS))
    m = CP.P12.ip_measurements(tracks, P['global_indices'], 0)
    msk_rho = av & (rm == 1) & (modes == 1)
    x = np.log10(np.maximum(m['mag_um'], 1e-3))
    e = np.quantile(x[msk_rho], np.linspace(0, 1, 7))

    rho_by_ip = {}
    for v in I.files:
        if v == 'global_indices':
            continue
        h = I[v]
        rho_by_ip[v] = []
        for i in range(6):
            s = msk_rho & (x >= e[i]) & (x <= e[i + 1])
            corrs = [wcorr(h[..., c][s], hx[..., c][s], W[s]) for c in range(3)]
            rho_by_ip[v].append({
                'ip_bin': i,
                'ip_range_um': [float(10**e[i]), float(10**e[i+1])],
                'corrs': [float(c) for c in corrs],
                'sides': int(s.sum())
            })
    results['rho_by_ip_sextile'] = rho_by_ip

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Summary saved to {args.output}")


if __name__ == '__main__':
    main()
