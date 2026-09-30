"""ATLAS full-reco sample: the same 22-channel geometry blocks as build_simple_inputs.py.

local / local_shuffle are copied unchanged from hz_beyond_ceiling p10 v2 (geo_full22.npz keys
'full' / 'full_shuffle', the arms of the 2026-09-23/24 runs).  legacy and lab are new and use
the same per-track IP reconstruction as p10 (perigee + beam spot, minus PV, track component
removed) and the PV->SV vector of the 2026-09-20 geometry audit:
  legacy  per track (d0/50um, z0/50mm, avail) with the ntuple's beam-spot z0 + SV lab block
  lab     per track (PV-referenced IP x, y, z /50um, avail) + SV lab block
  SV lab block (slots 12-15): PV->SV x, y, z /1mm (clipped +-50), avail
Also writes validation arrays for the resolution comparison.

usage: python build_atlas_geometry.py OUT.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

BC = Path('/home/rbaba/hz-beyond-ceiling-20260923/artifacts')
LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
AUDIT = Path('/home/rbaba/tauspin-ip-sv-analysis-corrected-20260920/geometry_audit.npz')
G = 22


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def main(out_path):
    trk = np.load(BC / 'ip_tracks.npz')
    pv, t = trk['pv'], trk['trk']
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    beam = np.array([*np.median(pv[:, :2], axis=0), -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))])
    theta, phi, d0, z0 = (t[..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip = pca - pv[:, None, None, :]
    ip = ip - np.sum(ip * tdir, -1, keepdims=True) * tdir
    ip_ok = np.isfinite(d0)
    with np.load(AUDIT) as g:
        sv_av = np.asarray(g['sv_available']) > 0
        sv_dir = np.asarray(g['sv_direction'])
        sv_len = np.asarray(g['sv_length'])
    sv_ok = sv_av & (sv_len > 1e-6) & np.isfinite(sv_dir).all(-1)
    geo = np.load(BC / 'geo_full22.npz')
    out, report = {}, {'beam_reference_mm': beam.tolist()}
    for split in ('train', 'validation'):
        ids = geo[f'{split}_global_indices']
        n = len(ids)
        f = {k: np.zeros((n, 2, G), np.float32) for k in ('legacy', 'lab')}
        for j in range(3):
            good = ip_ok[ids, :, j]
            v = np.nan_to_num(ip[ids, :, j])
            f['lab'][..., 4 * j:4 * j + 4] = np.where(good[..., None], np.concatenate(
                [np.clip(v / 0.05, -20, 20), np.ones((n, 2, 1))], -1), 0)
            f['legacy'][..., 4 * j:4 * j + 3] = np.where(good[..., None], np.stack(
                [np.clip(np.nan_to_num(d0[ids, :, j]) / 0.05, -20, 20), np.clip(np.nan_to_num(z0[ids, :, j]) / 50.0, -20, 20),
                 np.ones((n, 2))], -1), 0)
        s = np.nan_to_num(sv_dir[ids] * sv_len[ids][..., None])
        svb = np.concatenate([np.clip(s / 1.0, -50, 50), np.ones((n, 2, 1))], -1) * sv_ok[ids][..., None]
        for k in f:
            f[k][..., 12:16] = svb
        out[f'{split}_global_indices'] = ids
        out[f'{split}_none'] = np.zeros((n, 2, G), np.float32)
        out[f'{split}_legacy'] = f['legacy']
        out[f'{split}_lab'] = f['lab']
        out[f'{split}_local'] = geo[f'{split}_full']
        out[f'{split}_local_shuffle'] = geo[f'{split}_full_shuffle']
        report[split] = dict(rows=int(n), lab_std=f['lab'].reshape(-1, G).std(0).round(2).tolist(),
                             legacy_std=f['legacy'].reshape(-1, G).std(0).round(2).tolist(),
                             sv_available_fraction=float(sv_ok[ids].mean()))
        if split == 'validation':
            with np.load(LADDER / 'validation.npz') as d:
                if not np.array_equal(np.asarray(d['global_indices']), ids):
                    raise RuntimeError('ladder / geometry identity mismatch')
                basis = np.asarray(d['reco_basis'])
                tau_true = unit(np.asarray(d['truth_visible_tau_lab4'])[..., :3] + np.asarray(d['truth_neutrino_lab4'])[..., :3])
                reco_mode = np.asarray(d['reco_h_mode'])
            out['res_ip'] = ip[ids]                 # N, side, track, xyz (mm)
            out['res_ip_ok'] = ip_ok[ids]
            out['res_track_dir'] = tdir[ids]
            out['res_sv'] = s
            out['res_sv_ok'] = sv_ok[ids]
            out['res_tau_true'] = tau_true
            out['res_basis'] = basis
            out['res_reco_mode'] = reco_mode
            out['res_n_core'] = trk['n_core'][ids]
    np.savez_compressed(out_path, **out)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
