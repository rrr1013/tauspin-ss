"""Cache the truth visible daughters and truth neutrinos of the canonical
exact-h validation cohort, so the kinematic null space can be constructed
without re-reading ROOT for every study.

Inputs are read only.  The cohort, the decay-mode definition and the exact-h
reference vector all come from the existing canonical artifacts:

  /home/rbaba/reco-h-supervision-20260906-targets/validation.npz   (cohort + h)
  /home/rbaba/truth-neutrino-20260906-dataset/metadata.json        (ROOT files)

Four-vectors are (px,py,pz,E) in GeV.  Sides are tau- (index 0) then tau+
(index 1), matching every existing artifact in this project.

Per side the charged-pion slots are stored in ntuple order and zero padded to
three; the pi0 is stored as a separate single four-vector.  This is exactly the
layout `HybridPolarimeter` expects, so the same vectorised polarimeter that the
existing runs use can be fed an arbitrary neutrino momentum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

COHORT = Path('/home/rbaba/reco-h-supervision-20260906-targets/validation.npz')
DATASET = Path('/home/rbaba/truth-neutrino-20260906-dataset')

# mode code -> sorted |pdgId| of the visible daughters, as fixed by the
# canonical exact-h builder (reco_h_supervision_20260906/build_targets.py).
PATTERNS = {0: [211], 1: [111, 211], 3: [211, 211, 211]}

KEYS = ['eventNumber'] + [
    f'truth_{obj}_{field}'
    for obj in ('particle', 'neutrino')
    for field in ('pdgId', 'tauIndex', 'pt', 'eta', 'phi', 'm')
] + ['truth_particle_charge']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def four_vector(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """(...,) kinematics -> (...,4) four-vector with energy from the stored mass."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.stack((px, py, pz, energy), axis=-1)


def padded(values: ak.Array, width: int) -> np.ndarray:
    """Jagged -> dense (n,width), zero filled.  Overflow is not silently cut:
    the caller asserts the multiplicity first."""
    return ak.to_numpy(ak.fill_none(ak.pad_none(values, width, clip=True), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=int, default=0,
                        help='only the first N cohort rows; output is marked as a probe')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with np.load(COHORT) as handle:
        valid = np.flatnonzero(handle['h_valid'])
        if args.probe:
            valid = valid[: args.probe]
        cols = {key: handle[key][valid].copy()
                for key in ('labels', 'file_indices', 'entry_indices', 'event_numbers',
                            'global_indices', 'weights', 'modes')}
        h_ref = handle['h'][valid].copy()
    rows = len(valid)
    if not args.probe and rows != 59390:
        raise ValueError(f'unexpected canonical validation cohort size {rows}')
    if len(np.unique(cols['global_indices'])) != rows:
        raise ValueError('duplicate identity in cohort')

    meta = json.loads((DATASET / 'metadata.json').read_text())

    out = {
        'pions': np.zeros((rows, 2, 3, 4)),
        'pion_charges': np.zeros((rows, 2, 3)),
        'pion_counts': np.zeros((rows, 2), np.int32),
        'pi0': np.zeros((rows, 2, 4)),
        'pi0_counts': np.zeros((rows, 2), np.int32),
        'nu4': np.zeros((rows, 2, 4)),
    }
    filled = np.zeros(rows, bool)
    source_files: list[str] = []

    for label in (0, 1):
        sample = 'H' if label else 'Z'
        chunks = np.unique(cols['file_indices'][cols['labels'] == label] // 10)
        for chunk in chunks:
            index = np.flatnonzero((cols['labels'] == label) & (cols['file_indices'] // 10 == chunk))
            path = meta['files'][sample][int(chunk)]
            source_files.append(path)
            entries = cols['entry_indices'][index].astype(np.int64)
            with uproot.open(path) as handle:
                arrays = handle['tauspin'].arrays(KEYS, library='ak')
            event = arrays[entries]
            if not np.array_equal(ak.to_numpy(event['eventNumber']).astype(np.int64),
                                  cols['event_numbers'][index]):
                raise ValueError(f'event identity mismatch in {path}')

            pdg = event['truth_particle_pdgId']
            side = event['truth_particle_tauIndex']
            charge = event['truth_particle_charge']
            nu_pdg = event['truth_neutrino_pdgId']
            nu_side = event['truth_neutrino_tauIndex']

            for s in (0, 1):
                mode = cols['modes'][index, s]
                on_side = side == s
                is_pion = on_side & (abs(pdg) == 211)
                is_pi0 = on_side & (abs(pdg) == 111)
                n_pion = ak.to_numpy(ak.sum(is_pion, axis=-1))
                n_pi0 = ak.to_numpy(ak.sum(is_pi0, axis=-1))
                n_all = ak.to_numpy(ak.sum(on_side, axis=-1))
                expect_pion = np.where(mode == 3, 3, 1)
                expect_pi0 = np.where(mode == 1, 1, 0)
                if not (np.array_equal(n_pion, expect_pion)
                        and np.array_equal(n_pi0, expect_pi0)
                        and np.array_equal(n_all, expect_pion + expect_pi0)):
                    raise ValueError('visible daughter pattern violates the canonical mode pattern')

                pion = four_vector(*(padded(event['truth_particle_' + f][is_pion], 3)
                                     for f in ('pt', 'eta', 'phi', 'm')))
                pion_charge = padded(charge[is_pion], 3)
                # Zero-padded slots must stay exactly zero four-vectors.
                slot_active = np.arange(3)[None, :] < expect_pion[:, None]
                pion[~slot_active] = 0.0
                pion_charge[~slot_active] = 0.0
                expected_charge = np.where(s == 0, -1.0, 1.0)
                if not np.allclose(pion_charge.sum(axis=-1), expected_charge):
                    raise ValueError('summed visible charge does not match the side convention')
                if not np.array_equal(np.sign(padded(pdg[is_pion], 3)) * slot_active,
                                      pion_charge * slot_active):
                    raise ValueError('pion charge and pdgId sign disagree')

                pi0 = four_vector(*(padded(event['truth_particle_' + f][is_pi0], 1)
                                    for f in ('pt', 'eta', 'phi', 'm')))[:, 0, :]
                pi0[expect_pi0 == 0] = 0.0

                on_nu = nu_side == s
                if not np.array_equal(ak.to_numpy(ak.sum(on_nu, axis=-1)), np.ones(len(index), int)):
                    raise ValueError('expected exactly one truth neutrino per side')
                nu_id = padded(nu_pdg[on_nu], 1)[:, 0]
                if not np.all(nu_id == (16 if s == 0 else -16)):
                    raise ValueError('truth neutrino pdgId does not match the side convention')
                nu = four_vector(*(padded(event['truth_neutrino_' + f][on_nu], 1)
                                   for f in ('pt', 'eta', 'phi', 'm')))[:, 0, :]

                out['pions'][index, s] = pion
                out['pion_charges'][index, s] = pion_charge
                out['pion_counts'][index, s] = expect_pion
                out['pi0'][index, s] = pi0
                out['pi0_counts'][index, s] = expect_pi0
                out['nu4'][index, s] = nu
            filled[index] = True
            print(json.dumps({'sample': sample, 'chunk': int(chunk), 'rows': len(index)}), flush=True)

    if not filled.all():
        raise ValueError('some cohort rows were never filled')
    for key, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'non-finite values in {key}')

    visible = out['pions'].sum(axis=2) + out['pi0']
    tau = visible + out['nu4']
    tau_mass = np.sqrt(np.maximum(tau[..., 3] ** 2 - np.sum(tau[..., :3] ** 2, axis=-1), 0.0))
    nu_mass2 = out['nu4'][..., 3] ** 2 - np.sum(out['nu4'][..., :3] ** 2, axis=-1)

    audit = {
        'cohort': str(COHORT),
        'cohort_sha256': sha256_file(COHORT),
        'dataset_metadata': str(DATASET / 'metadata.json'),
        'rows': rows,
        'probe_rows': args.probe,
        'source_files': source_files,
        'fourvector_order': 'px,py,pz,E in GeV; side 0 = tau-, side 1 = tau+',
        'pion_slot_rule': 'charged pions in ntuple order, zero padded to three slots',
        'mode_patterns': {str(k): v for k, v in PATTERNS.items()},
        'truth_tau_mass_GeV_quantiles': np.quantile(tau_mass, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        'truth_nu_mass_squared_GeV2_max_abs': float(np.max(abs(nu_mass2))),
        'visible_mass_GeV_quantiles': np.quantile(
            np.sqrt(np.maximum(visible[..., 3] ** 2 - np.sum(visible[..., :3] ** 2, axis=-1), 0.0)),
            [0, 0.5, 1]).tolist(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'awkward': ak.__version__, 'uproot': uproot.__version__},
        'hostname': platform.node(),
        'script_sha256': sha256_file(Path(__file__)),
        'elapsed_seconds': time.time() - started,
    }
    np.savez_compressed(args.output, h_ref=h_ref, **cols, **out)
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
