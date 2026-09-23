"""Canonical exact-h targets with the 3pi sides from the generator current.

Copies /home/rbaba/reco-h-supervision-20260906-targets/{split}.npz field by
field and replaces only h[row, side] where the truth mode is 3pi.  Truth pions,
pi0 and neutrinos are re-read from the ROOT files of the canonical dataset
(same identity checks as azimuth_nullspace/build_truth_surface.py).  Every pi
and rho side of the touched events is recomputed as a closure test against
the stored h.  With --surface, the cached validation truth surface is also
written with h_ref replaced, for the null-space code.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
from polarimeter import canonical_h, frames  # noqa: E402

TARGETS = Path('/home/rbaba/reco-h-supervision-20260906-targets')
DATASET = Path('/home/rbaba/truth-neutrino-20260906-dataset')
KEYS = ['eventNumber'] + [f'truth_{o}_{f}' for o in ('particle', 'neutrino')
                          for f in ('pdgId', 'tauIndex', 'pt', 'eta', 'phi', 'm')] + ['truth_particle_charge']


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            d.update(b)
    return d.hexdigest()


def four_vector(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack((px, py, pz, np.sqrt(px * px + py * py + pz * pz + m * m)), -1)


def padded(v, width):
    return ak.to_numpy(ak.fill_none(ak.pad_none(v, width, clip=True), 0.0))


def read_truth(labels, file_indices, entry_indices, event_numbers, modes):
    n = len(labels)
    out = {'pions': np.zeros((n, 2, 3, 4)), 'charges': np.zeros((n, 2, 3)),
           'pi0': np.zeros((n, 2, 4)), 'nu4': np.zeros((n, 2, 4))}
    filled = np.zeros(n, bool)
    meta = json.loads((DATASET / 'metadata.json').read_text())
    for label in (0, 1):
        sample = 'H' if label else 'Z'
        for chunk in np.unique(file_indices[labels == label] // 10):
            idx = np.flatnonzero((labels == label) & (file_indices // 10 == chunk))
            with uproot.open(meta['files'][sample][int(chunk)]) as f:
                ev = f['tauspin'].arrays(KEYS, library='ak')[entry_indices[idx].astype(np.int64)]
            if not np.array_equal(ak.to_numpy(ev['eventNumber']).astype(np.int64), event_numbers[idx]):
                raise ValueError('event identity mismatch')
            pdg, side = ev['truth_particle_pdgId'], ev['truth_particle_tauIndex']
            for s in (0, 1):
                md = modes[idx, s]
                n_pi = np.where(md == 3, 3, 1)
                n_z = np.where(md == 1, 1, 0)
                on = side == s
                is_pi, is_z = on & (abs(pdg) == 211), on & (abs(pdg) == 111)
                if not (np.array_equal(ak.to_numpy(ak.sum(is_pi, -1)), n_pi)
                        and np.array_equal(ak.to_numpy(ak.sum(is_z, -1)), n_z)
                        and np.array_equal(ak.to_numpy(ak.sum(on, -1)), n_pi + n_z)):
                    raise ValueError('visible daughters violate the canonical mode pattern')
                act = np.arange(3)[None] < n_pi[:, None]
                p = four_vector(*(padded(ev['truth_particle_' + k][is_pi], 3) for k in ('pt', 'eta', 'phi', 'm')))
                q = padded(ev['truth_particle_charge'][is_pi], 3)
                p[~act] = 0.0
                q[~act] = 0.0
                z = four_vector(*(padded(ev['truth_particle_' + k][is_z], 1) for k in ('pt', 'eta', 'phi', 'm')))[:, 0]
                z[n_z == 0] = 0.0
                on_nu = ev['truth_neutrino_tauIndex'] == s
                if not np.array_equal(ak.to_numpy(ak.sum(on_nu, -1)), np.ones(len(idx), int)):
                    raise ValueError('expected one truth neutrino per side')
                nu = four_vector(*(padded(ev['truth_neutrino_' + k][on_nu], 1) for k in ('pt', 'eta', 'phi', 'm')))[:, 0]
                out['pions'][idx, s], out['charges'][idx, s] = p, q
                out['pi0'][idx, s], out['nu4'][idx, s] = z, nu
            filled[idx] = True
            print(json.dumps({'sample': sample, 'chunk': int(chunk), 'rows': len(idx)}), flush=True)
    if not filled.all():
        raise ValueError('rows not filled')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'validation'), required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--surface', type=Path, help='truth_surface.npz to rewrite (validation only)')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    src = TARGETS / f'{args.split}.npz'
    with np.load(src) as d:
        data = {k: d[k] for k in d.files}
    valid = data['h_valid'].astype(bool)
    modes = data['modes']
    rows = np.flatnonzero(valid & (modes == 3).any(1))
    truth = read_truth(data['labels'][rows], data['file_indices'][rows], data['entry_indices'][rows],
                       data['event_numbers'][rows], modes[rows])
    fr = frames(truth['pions'], truth['pi0'], truth['nu4'])
    h_old = data['h'][rows].astype(np.float64)
    h_new = h_old.copy()
    closure = {}
    cos3 = []
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, modes[rows], truth['charges'], s, md)
            if md == 3:
                h_new[sel, s] = h
                cos3.append(np.sum(h * h_old[sel, s], -1))
            else:
                dev = np.abs(h - h_old[sel, s]).max() if len(sel) else 0.0
                closure[f'mode{md}_side{s}'] = {'n': int(len(sel)), 'max_abs_dev': float(dev)}
                if dev > 1e-5:
                    raise RuntimeError(f'closure failed mode {md} side {s}: {dev}')
    cos3 = np.concatenate(cos3)
    new_h = data['h'].copy()
    new_h[rows] = h_new.astype(data['h'].dtype)
    data['h'] = new_h
    np.savez_compressed(args.output_dir / f'{args.split}.npz', **data)
    audit = {
        'split': args.split, 'source': str(src), 'source_sha256': sha256_file(src),
        'rows': int(len(valid)), 'h_valid_rows': int(valid.sum()),
        'events_with_3pi_side': int(len(rows)), 'replaced_3pi_sides': int(len(cos3)),
        'cos_old_new_3pi': {'mean': float(cos3.mean()), 'median': float(np.median(cos3)),
                            'frac_negative': float(np.mean(cos3 < 0))},
        'pi_rho_closure_on_touched_events': closure,
        'unchanged_fields': sorted(k for k in data if k != 'h'),
        'current': 'TauDecay UFO Kuehn-Santamaria (rho770 + rho1465, beta=-0.145); pi and rho unchanged',
        'hostname': platform.node(), 'script_sha256': sha256_file(__file__),
        'elapsed_seconds': time.time() - started, 'test_split_loaded': False,
    }
    if args.surface:
        with np.load(args.surface) as s:
            surf = {k: s[k] for k in s.files}
        pos = {int(v): i for i, v in enumerate(data['global_indices'])}
        take = np.asarray([pos[int(v)] for v in surf['global_indices']])
        if not np.array_equal(data['labels'][take], surf['labels']):
            raise RuntimeError('surface label mismatch')
        keep3 = surf['modes'] == 3
        dev = np.abs(np.where(keep3[..., None], 0.0, surf['h_ref'] - data['h'][take])).max()
        if dev > 1e-6:
            raise RuntimeError(f'surface pi/rho h differs from targets: {dev}')
        surf['h_ref'] = data['h'][take].astype(surf['h_ref'].dtype)
        np.savez_compressed(args.output_dir / 'truth_surface_gen3pi.npz', **surf)
        audit['surface'] = {'source': str(args.surface), 'source_sha256': sha256_file(args.surface),
                            'rows': int(len(take))}
    (args.output_dir / f'{args.split}.audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
