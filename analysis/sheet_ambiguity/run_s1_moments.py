"""Per-event polarimeter moments on the null space, resolved by mass-shell sheet.

Same solution set, same measures, same polarimeter and same reference frame as
`met_nullspace/run_m1_moments.py`; the only additions are the conditionings that
the discrete ambiguity needs:

  O0   the whole solution set                      (reproduces the previous run)
  O1   restricted to the truth sheet               (discrete ambiguity resolved)
  O2a  correct mass-shell root on side 0 only
  O2b  correct mass-shell root on side 1 only
  Sx   reweighted by an idealised tau-direction measurement of resolution sigma

`O3`, which fixes the continuous coordinate at the truth transverse momentum and
leaves only the four sheets, is four hypotheses per event and is done in one go
after the loop.

The direction noise comes from its own generator, so the main stream - and
therefore the draws, and therefore every O0 number - is bit-identical to the
previous run at the same seed.

Read-only inputs: the cached truth surface and the existing polarimeter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'met_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms
import nullspace as ns
import sheetspace as ss

BASE_MEASURES = ('flat', 'solid', 'solid_clipped')
ORACLES = ('o0', 'o1', 'o2a', 'o2b')
DEFAULT_SIGMA_MRAD = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
# Vertex position resolutions, converted per event into a direction resolution
# sigma_x / L with L the tau decay length.  Applied to three-prong sides only:
# there the three tracks define a vertex.  The one-prong impact-parameter
# geometry is a different construction and is deliberately not modelled here.
DEFAULT_VERTEX_UM = (10.0, 20.0, 30.0, 50.0, 100.0)
THREE_PRONG_MODE = 3
TAU_CTAU_UM = 87.03


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bank_names(sigmas: tuple[float, ...], vertices: tuple[float, ...]) -> list[str]:
    names = [f'{measure}_{oracle}' for measure in BASE_MEASURES for oracle in ORACLES]
    # One bank per sheet, so that any sheet-selection rule can be applied after
    # the fact: if the posterior is a mixture of the sheet-conditional measures
    # with weights p_s, so are its first and second moments.  That makes the
    # discrete channel exactly computable at any direction resolution, without
    # the Monte Carlo grid having to resolve the measurement.
    names += [f'flat_sheet{index}' for index in range(4)]
    names += [f'flat_sigma{sigma:g}' for sigma in sigmas]
    names += [f'flat_oneside{sigma:g}' for sigma in sigmas]
    names += [f'flat_vertex{micron:g}' for micron in vertices]
    return names


class Accumulator:
    """Weighted first and second moments of h, one bank per (measure, conditioning)."""

    def __init__(self, rows: int, names: list[str]):
        self.names = names
        self.sum_w = {n: np.zeros(rows) for n in names}
        self.sum_w2 = {n: np.zeros(rows) for n in names}
        self.first = {n: np.zeros((rows, 2, 3)) for n in names}
        self.second = {n: np.zeros((rows, 3, 3)) for n in names}
        self.t_own = {n: np.zeros(rows) for n in names}
        self.t_square = {n: np.zeros(rows) for n in names}
        self.sheet_w = {n: np.zeros((rows, 4)) for n in names}
        # Flat O0 only: how far each sheet's tau directions sit from the truth one.
        self.sheet_angle = np.zeros((rows, 4, 2))
        self.sheet_angle2 = np.zeros((rows, 4, 2))
        self.count = np.zeros(rows, dtype=np.int64)

    def add(self, name: str, weight: np.ndarray, h: np.ndarray, block: np.ndarray,
            sheet: np.ndarray, reference_t: np.ndarray, t_own: np.ndarray) -> None:
        """`weight` (H,R), `h` (H,R,2,3) in the common reference frame."""
        self.sum_w[name][block] += weight.sum(axis=0)
        self.sum_w2[name][block] += (weight ** 2).sum(axis=0)
        self.first[name][block] += np.einsum('hr,hrsi->rsi', weight, h)
        self.second[name][block] += np.einsum('hr,hri,hrj->rij', weight,
                                              h[:, :, 0], h[:, :, 1])
        self.t_own[name][block] += np.einsum('hr,hr->r', weight, t_own)
        self.t_square[name][block] += np.einsum('hr,hr->r', weight, reference_t ** 2)
        for index in range(4):
            rows_of_sheet = sheet == index
            if not rows_of_sheet.any():
                continue
            self.sheet_w[name][block, index] += weight[rows_of_sheet].sum(axis=0)

    def add_geometry(self, weight: np.ndarray, block: np.ndarray, sheet: np.ndarray,
                     angle: np.ndarray) -> None:
        """`angle` (H,R,2): hypothesis tau direction against the truth direction."""
        for index in range(4):
            rows_of_sheet = sheet == index
            if not rows_of_sheet.any():
                continue
            part = weight[rows_of_sheet][..., None]
            self.sheet_angle[block, index] += (part * angle[rows_of_sheet]).sum(axis=0)
            self.sheet_angle2[block, index] += (part * angle[rows_of_sheet] ** 2).sum(axis=0)

    def normalised(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {'count': self.count}
        for name in self.names:
            norm = np.maximum(self.sum_w[name], 1e-300)
            out[f'first_{name}'] = self.first[name] / norm[:, None, None]
            out[f'second_{name}'] = self.second[name] / norm[:, None, None]
            out[f't_own_{name}'] = self.t_own[name] / norm
            out[f't_square_{name}'] = self.t_square[name] / norm
            out[f'sheet_weight_{name}'] = self.sheet_w[name] / norm[:, None]
            out[f'sum_w_{name}'] = self.sum_w[name]
            out[f'ess_{name}'] = np.where(
                self.sum_w2[name] > 0.0,
                self.sum_w[name] ** 2 / np.maximum(self.sum_w2[name], 1e-300), 0.0)
        flat_norm = np.maximum(self.sheet_w['flat_o0'], 1e-300)[..., None]
        out['sheet_angle_mean'] = self.sheet_angle / flat_norm
        out['sheet_angle_rms'] = np.sqrt(np.maximum(self.sheet_angle2 / flat_norm, 0.0))
        return out


def build_sheets(visible: np.ndarray, nu_t_pair: np.ndarray) -> dict:
    """Four longitudinal sheets above fixed transverse momenta on both sides."""
    solved = ns.solve_nu_z(visible[None], nu_t_pair, np.asarray(ms.UPSTREAM_TAU_MASS))
    nz, valid_root = solved['nz'], solved['valid']
    return {
        'nu': np.stack([np.concatenate(
            (nu_t_pair, np.stack((nz[..., 0, i], nz[..., 1, j]), axis=-1)[..., None]), axis=-1)
            for i, j in ss.SHEETS], axis=1),
        'valid': np.stack([valid_root[..., 0, i] & valid_root[..., 1, j] for i, j in ss.SHEETS],
                          axis=1),
        'discriminant': solved['discriminant'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', default='sheet')
    parser.add_argument('--proposal-lattice', type=int, default=64)
    parser.add_argument('--keep', type=int, default=128)
    parser.add_argument('--row-chunk', type=int, default=1500)
    parser.add_argument('--hypothesis-chunk', type=int, default=64)
    parser.add_argument('--clip-quantile', type=float, default=0.99)
    parser.add_argument('--rows', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260919,
                        help='the previous run seed, so that O0 reproduces exactly')
    parser.add_argument('--sigma-mrad', type=float, nargs='*', default=list(DEFAULT_SIGMA_MRAD))
    parser.add_argument('--vertex-um', type=float, nargs='*', default=list(DEFAULT_VERTEX_UM))
    parser.add_argument('--nn-root', type=Path, default=ns.DEFAULT_NN_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    sigmas = tuple(args.sigma_mrad)
    vertices = tuple(args.vertex_um)
    names = bank_names(sigmas, vertices)
    surface = ns.load_surface(args.surface)
    polarimeter, origin = ns.load_polarimeter(args.nn_root)
    n_rows = len(surface['modes']) if not args.rows else args.rows

    truth_nu = surface['nu4'][:n_rows][..., :3]
    visible = surface['visible'][:n_rows]
    met = truth_nu[:, 0, :2] + truth_nu[:, 1, :2]
    rng = np.random.default_rng(args.seed)
    rng_dir = np.random.default_rng(args.seed + 1)

    truth_direction = ss.tau_direction(visible, truth_nu)
    tangent_first, tangent_second = ss.tangent_frame(truth_direction)
    direction_noise = rng_dir.normal(size=(n_rows, 2, 2))
    label = ss.truth_sheet(visible, truth_nu)

    # Per-side direction resolution a decay-vertex measurement would give.
    tau4 = visible + np.concatenate(
        (truth_nu, np.linalg.norm(truth_nu, axis=-1)[..., None]), axis=-1)
    tau_mass = np.sqrt(np.maximum(ns.minkowski_dot(tau4, tau4), 1e-12))
    boost = np.linalg.norm(tau4[..., :3], axis=-1) / tau_mass
    decay_length_um = rng_dir.exponential(boost * TAU_CTAU_UM)
    has_vertex = surface['modes'][:n_rows] == THREE_PRONG_MODE

    # (name -> per-side 1/sigma^2 in rad^-2); zero means that side is unmeasured.
    inverse_variance: dict[str, np.ndarray] = {}
    for sigma in sigmas:
        radians = sigma * 1e-3
        inverse_variance[f'flat_sigma{sigma:g}'] = np.full((n_rows, 2), radians ** -2)
        one_side = np.zeros((n_rows, 2))
        one_side[:, 0] = radians ** -2
        inverse_variance[f'flat_oneside{sigma:g}'] = one_side
    for micron in vertices:
        radians = 1e-6 * micron / np.maximum(decay_length_um * 1e-6, 1e-12)
        inverse_variance[f'flat_vertex{micron:g}'] = np.where(
            has_vertex, radians ** -2, 0.0)

    accumulator = Accumulator(n_rows, names)
    reference_basis = np.zeros((n_rows, 3, 3))
    truth_h_reference = np.zeros((n_rows, 2, 3))
    truth_h_closure = np.zeros(n_rows)
    region_area = np.zeros(n_rows)
    acceptance = np.zeros(n_rows)
    retained = np.zeros(n_rows, dtype=np.int64)
    min_angle2 = np.zeros(n_rows)
    root_offset = np.zeros((n_rows, 2, 2))
    o3_first = np.zeros((n_rows, 2, 3))
    o3_second = np.zeros((n_rows, 3, 3))
    o3_sheet_weight = np.zeros((n_rows, 4))
    o3_valid = np.zeros(n_rows, dtype=np.int64)
    root_angle = np.zeros((n_rows, 2))

    for start in range(0, n_rows, args.row_chunk):
        stop = min(start + args.row_chunk, n_rows)
        block = np.arange(start, stop)
        rows = len(block)
        vis = visible[block]
        met_block = met[block]
        nu_truth = truth_nu[block]
        unit = ms.lattice(args.proposal_lattice, rows, rng).transpose(1, 0, 2)

        drawn = ms.sample_region(vis, met_block, unit)
        index = ms.compact(drawn['accepted'], args.keep, rng)
        nu_t = np.take_along_axis(drawn['nu_t'], np.maximum(index, 0)[..., None], axis=0)
        nu_t_pair = np.stack((nu_t, met_block[None] - nu_t), axis=-2)
        sheets = build_sheets(vis, nu_t_pair)
        sheets['valid'] &= (index >= 0)[:, None, :]
        acceptance[block] = drawn['accepted'].mean(axis=0)
        region_area[block] = acceptance[block] * drawn['spec']['area']
        retained[block] = (index >= 0).sum(axis=0)

        nu = sheets['nu']                                       # (S,4,R,2,3)
        ok = sheets['valid']                                    # (S,4,R)
        accumulator.count[block] = ok.sum(axis=(0, 1))

        per_draw = ms.solid_angle_weight(sheets['discriminant'])
        per_draw = np.where(ok.any(axis=1) & np.isfinite(per_draw), per_draw, 0.0)
        limit = np.zeros(rows)
        for r in range(rows):
            positive = per_draw[:, r][per_draw[:, r] > 0.0]
            limit[r] = np.quantile(positive, args.clip_quantile) if positive.size else 0.0
        weight_solid = np.where(ok, np.broadcast_to(per_draw[:, None], ok.shape), 0.0)
        base = {'flat': ok.astype(np.float64).reshape(-1, rows),
                'solid': weight_solid.reshape(-1, rows),
                'solid_clipped': np.minimum(weight_solid, limit[None, None]).reshape(-1, rows)}

        # Reference frame: the flat-measure average tau pair, hypotheses only.
        tau_sum = np.zeros((rows, 2, 4))
        for sheet in range(4):
            nu4 = np.concatenate(
                (nu[:, sheet], np.linalg.norm(nu[:, sheet], axis=-1)[..., None]), axis=-1)
            tau_sum += np.sum(np.where(ok[:, sheet][..., None, None], vis[None] + nu4, 0.0),
                              axis=0)
        basis = ns.pair_basis(tau_sum / np.maximum(accumulator.count[block], 1)[:, None, None])
        reference_basis[block] = basis

        truth_result = ns.evaluate_h(polarimeter, surface, nu_truth[None], block)
        truth_h_closure[block] = np.abs(
            truth_result['h'][0] - surface['h_ref'][block].astype(np.float64)).max(axis=(-1, -2))
        truth_h_reference[block] = ms.to_reference_basis(
            truth_result['h'], truth_result['hypothesis_basis'], basis[None])[0]

        flat_nu = nu.reshape(-1, rows, 2, 3)
        flat_ok = ok.reshape(-1, rows)
        flat_nu = np.where(flat_ok[..., None, None],
                           flat_nu, np.broadcast_to(nu_truth[None], flat_nu.shape))

        # Geometry and the direction likelihood, once for every hypothesis.
        #
        # Every hypothesis and every measurement is written in the tangent plane
        # of the truth direction, where the measurement is exactly the isotropic
        # Gaussian it is meant to be.  The tangent coordinates are sin(angle)
        # times a unit azimuth, so at the few tens of milliradians involved here
        # they are the angular displacement to better than one part in 10^4.
        hypothesis_direction = ss.tau_direction(vis[None], flat_nu)
        angle = ss.opening_angle(hypothesis_direction, truth_direction[block][None])
        offset = np.stack(
            (np.sum(hypothesis_direction * tangent_first[block][None], axis=-1),
             np.sum(hypothesis_direction * tangent_second[block][None], axis=-1)),
            axis=-1)                                            # (H,R,2sides,2)
        angle2 = np.where(flat_ok, np.sum(angle * angle, axis=-1), np.inf)
        closest = angle2.min(axis=0)
        min_angle2[block] = np.where(np.isfinite(closest), closest, 0.0)

        sheet_axis = np.arange(flat_ok.shape[0]) % 4
        truth_sheet_block = label['sheet'][block]
        truth_root = label['root'][block]
        conditioning = {
            'o0': np.ones(flat_ok.shape, dtype=bool),
            'o1': sheet_axis[:, None] == truth_sheet_block[None, :],
            'o2a': (sheet_axis[:, None] // 2) == truth_root[None, :, 0],
            'o2b': (sheet_axis[:, None] % 2) == truth_root[None, :, 1],
        }
        sigma_weight = {}
        for name, inverse in inverse_variance.items():
            width = np.where(inverse > 0.0, 1.0 / np.sqrt(np.maximum(inverse, 1e-300)), 0.0)
            measurement = width[block][..., None] * direction_noise[block]   # (R,2sides,2)
            residual = offset - measurement[None]
            quadratic = np.einsum('hrsc,rs->hr', residual ** 2, inverse[block])
            quadratic = np.where(flat_ok, quadratic, np.inf)
            shift = quadratic.min(axis=0)
            shift = np.where(np.isfinite(shift), shift, 0.0)
            excess = np.maximum(np.nan_to_num(quadratic - shift[None], nan=np.inf), 0.0)
            sigma_weight[name] = base['flat'] * np.exp(-0.5 * np.minimum(excess, 1400.0))

        for h_start in range(0, flat_nu.shape[0], args.hypothesis_chunk):
            h_stop = min(h_start + args.hypothesis_chunk, flat_nu.shape[0])
            piece = slice(h_start, h_stop)
            result = ns.evaluate_h(polarimeter, surface, flat_nu[piece], block)
            good = flat_ok[piece] & result['valid']
            h_reference = ms.to_reference_basis(result['h'], result['hypothesis_basis'],
                                                basis[None])
            t_own = ms.transverse_statistic(result['h'])
            reference_t = ms.transverse_statistic(h_reference)
            sheet_here = sheet_axis[piece]
            for measure in BASE_MEASURES:
                for oracle in ORACLES:
                    weight = np.where(good & conditioning[oracle][piece],
                                      base[measure][piece], 0.0)
                    accumulator.add(f'{measure}_{oracle}', weight, h_reference, block,
                                    sheet_here, reference_t, t_own)
            for index in range(4):
                accumulator.add(f'flat_sheet{index}',
                                np.where(good & (sheet_here[:, None] == index),
                                         base['flat'][piece], 0.0),
                                h_reference, block, sheet_here, reference_t, t_own)
            for name, weights in sigma_weight.items():
                accumulator.add(name, np.where(good, weights[piece], 0.0), h_reference,
                                block, sheet_here, reference_t, t_own)
            accumulator.add_geometry(np.where(good, base['flat'][piece], 0.0), block,
                                     sheet_here, angle[piece])

        # O3: the truth transverse momentum, four sheets, nothing else free.
        nu_t_free = nu_truth[:, 0, :2]
        nu_t_truth_pair = np.stack((nu_t_free, met_block - nu_t_free), axis=-2)
        o3 = build_sheets(vis, nu_t_truth_pair[None])
        o3_ok = o3['valid'][0]                                   # (4,R)
        o3_nu = np.where(o3_ok[..., None, None], o3['nu'][0],
                         np.broadcast_to(nu_truth[None], o3['nu'][0].shape))
        o3_result = ns.evaluate_h(polarimeter, surface, o3_nu, block)
        o3_good = o3_ok & o3_result['valid']
        o3_h = ms.to_reference_basis(o3_result['h'], o3_result['hypothesis_basis'], basis[None])
        o3_weight = o3_good.astype(np.float64)
        o3_norm = np.maximum(o3_weight.sum(axis=0), 1e-300)
        o3_first[block] = np.einsum('hr,hrsi->rsi', o3_weight, o3_h) / o3_norm[:, None, None]
        o3_second[block] = np.einsum('hr,hri,hrj->rij', o3_weight,
                                     o3_h[:, :, 0], o3_h[:, :, 1]) / o3_norm[:, None, None]
        o3_sheet_weight[block] = (o3_weight / o3_norm[None]).T
        o3_valid[block] = o3_good.sum(axis=0)
        o3_direction = ss.tau_direction(vis[None], o3['nu'][0])
        root_angle[block] = np.stack(
            (ss.opening_angle(o3_direction[0, :, 0], o3_direction[2, :, 0]),
             ss.opening_angle(o3_direction[0, :, 1], o3_direction[1, :, 1])), axis=-1)
        # Where the other root points, in the same tangent plane, so that the
        # sheet-only posterior can use the real geometry instead of an isotropy
        # argument.  Side 0 flips root 0, side 1 flips root 1.
        other = np.stack((o3_direction[2, :, 0], o3_direction[1, :, 1]), axis=-2)
        truth_root_block = label['root'][block]
        flip = truth_root_block == 1
        same = np.stack((o3_direction[0, :, 0], o3_direction[0, :, 1]), axis=-2)
        other = np.where(flip[..., None], same, other)
        root_offset[block] = np.stack(
            (np.sum(other * tangent_first[block], axis=-1),
             np.sum(other * tangent_second[block], axis=-1)), axis=-1)

        print(json.dumps({'rows_done': int(stop), 'elapsed_s': round(time.time() - started, 1)}),
              flush=True)

    payload = accumulator.normalised()
    payload.update({
        'reference_basis': reference_basis,
        'truth_h_canonical': surface['h_ref'][:n_rows].astype(np.float64),
        'truth_h_reference': truth_h_reference,
        'truth_h_closure': truth_h_closure,
        'truth_direction': truth_direction,
        'direction_noise': direction_noise,
        'min_angle2': min_angle2,
        'labels': surface['labels'][:n_rows],
        'weights': surface['weights'][:n_rows].astype(np.float64),
        'modes': surface['modes'][:n_rows],
        'global_indices': surface['global_indices'][:n_rows],
        'region_area': region_area,
        'acceptance': acceptance,
        'retained': retained,
        'truth_sheet': label['sheet'],
        'truth_root': label['root'],
        'root_residual_chosen': label['residual_chosen'],
        'root_residual_other': label['residual_other'],
        'root_separation': label['root_separation'],
        'root_angle': root_angle,
        'root_offset': root_offset,
        'decay_length_um': decay_length_um,
        'has_vertex': has_vertex,
        'vertex_um': np.asarray(vertices),
        'o3_first': o3_first,
        'o3_second': o3_second,
        'o3_sheet_weight': o3_sheet_weight,
        'o3_valid': o3_valid,
        'sigma_mrad': np.asarray(sigmas),
    })
    np.savez_compressed(args.output / f'{args.tag}_moments.npz', **payload)

    summary = {
        'tag': args.tag,
        'rows': int(n_rows),
        'sigma_mrad': list(sigmas),
        'proposal_lattice': args.proposal_lattice,
        'retained_per_event': args.keep,
        'clip_quantile': args.clip_quantile,
        'seed': args.seed,
        'inputs': {'surface': str(args.surface), 'surface_sha256': sha256_file(args.surface),
                   'hybrid_polarimeter': str(origin),
                   'hybrid_polarimeter_sha256': sha256_file(origin)},
        'closure': {
            'truth_h_max_abs_deviation': float(np.nanmax(truth_h_closure)),
            'truth_h_median_abs_deviation': float(np.nanmedian(truth_h_closure)),
            'root_residual_chosen_max_GeV': float(np.nanmax(label['residual_chosen'])),
            'root_residual_other_median_GeV': float(np.nanmedian(label['residual_other'])),
            'root_separation_quantiles_GeV': np.quantile(
                label['root_separation'], [0.01, 0.05, 0.5, 0.95]).tolist(),
            'near_degenerate_fraction_0p1GeV': float(
                (label['root_separation'] < 0.1).any(axis=-1).mean()),
            'both_roots_valid_fraction': float(label['both_roots_valid'].all(axis=-1).mean()),
        },
        'region': {
            'proposal_acceptance_mean': float(acceptance.mean()),
            'region_area_GeV2_quantiles': np.quantile(region_area, [0.05, 0.5, 0.95]).tolist(),
            'retained_quantiles': np.quantile(retained, [0.01, 0.05, 0.5, 0.95]).tolist(),
            'events_with_no_valid_hypothesis': int((payload['count'] == 0).sum()),
            'o3_valid_sheets_mean': float(o3_valid.mean()),
            'root_angle_mrad_quantiles': (1e3 * np.quantile(
                root_angle, [0.05, 0.5, 0.95])).tolist(),
        },
        'effective_sample_size': {
            name: {'mean': float(np.mean(payload[f'ess_{name}'])),
                   'quantiles': np.quantile(payload[f'ess_{name}'], [0.05, 0.5, 0.95]).tolist()}
            for name in names},
        'elapsed_seconds': time.time() - started,
        'hostname': platform.node(),
        'script_sha256': sha256_file(Path(__file__)),
        'sheetspace_sha256': sha256_file(Path(__file__).resolve().parent / 'sheetspace.py'),
        'metspace_sha256': sha256_file(Path(ms.__file__)),
        'nullspace_sha256': sha256_file(Path(ns.__file__)),
    }
    (args.output / f'{args.tag}_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
