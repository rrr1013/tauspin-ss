"""Why does the IP gain for pi sides grow with the boost?

Same PV-referenced 3-D IP construction as hz_beyond_ceiling/p2 and
cp_mixing/p3 (leading core track, along-track component removed).  For each
1-prong side we measure, per beta*gamma sextile and truth mode:
  * |IP| (um),
  * the azimuth error of the IP direction around the track w.r.t. the truth
    tau direction component perpendicular to the track,
  * the cone angle tau-track and the implied transverse tau-direction error
    cone * 2 sin(|dphi|/2), compared with the cone itself.
"""
import json
import os

import numpy as np

from boost_map import load

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'cp_mixing', 'data')


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def perp(v, axis):
    return v - np.sum(v * axis, -1, keepdims=True) * axis


def main():
    D = load()
    L = np.load(os.path.join(DATA, 'ladder_validation.npz'))
    T = np.load(os.path.join(DATA, 'ip_tracks.npz'))
    ids = L['global_indices']
    pv, t = T['pv'], T['trk']
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    beam = np.array([*np.median(pv[:, :2], axis=0), -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))])
    tl = t[ids, :, 0]                                              # (N, 2, 7)
    pvi = pv[ids]
    theta, phi, d0, z0, pt = tl[..., 1], tl[..., 2], tl[..., 4], tl[..., 5], tl[..., 0]
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip = perp(pca - pvi[:, None, :], tdir)
    tau = L['truth_visible_tau_lab4'][..., :3] + L['truth_neutrino_lab4'][..., :3]
    tau_hat = unit(tau)
    tperp = perp(tau_hat, tdir)
    cross = np.sum(np.cross(unit(tperp), unit(ip)) * tdir, -1)
    dphi = np.arctan2(cross, np.sum(unit(tperp) * unit(ip), -1))
    cone = np.arccos(np.clip(np.sum(tau_hat * tdir, -1), -1, 1))
    ipmag = np.linalg.norm(ip, axis=-1) * 1e3
    n_core = T['n_core'][ids]
    bg = D['bg']
    out = {}
    for mcode, name in ((0, 'pi'), (1, 'rho')):
        m = (D['modes'] == mcode) & (n_core == 1) & np.isfinite(d0) & np.isfinite(dphi)
        e = np.quantile(bg[m], np.linspace(0, 1, 7))
        rows = []
        for i in range(6):
            s = m & (bg >= e[i]) & (bg <= e[i + 1])
            ad = np.abs(dphi[s])
            implied = cone[s] * 2 * np.sin(ad / 2)
            rows.append({'bg_med': float(np.median(bg[s])), 'n': int(s.sum()),
                         'track_pt_med': float(np.median(pt[s])),
                         'ip_um_med': float(np.median(ipmag[s])),
                         'dphi_abs_med_deg': float(np.degrees(np.median(ad))),
                         'mean_cos_dphi': float(np.mean(np.cos(dphi[s]))),
                         'frac_within_30deg': float(np.mean(ad < np.pi / 6)),
                         'cone_mrad_med': float(1e3 * np.median(cone[s])),
                         'implied_err_over_cone_med': float(np.median(implied / cone[s]))})
        out[name] = rows
        print(name)
        for r in rows:
            print('  ' + '  '.join(f'{k} {v:.3g}' for k, v in r.items()))
    with open(os.path.join(HERE, 'results_ip.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
