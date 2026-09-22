"""Null-space moments with a PV-referenced impact-parameter likelihood.

Copied from met_nullspace/run_m1_moments.py (2026-09-19).  The MET-constrained
two-dimensional solution set is built exactly as before on truth visible
momenta and exact tau-neutrino MET.  Besides the flat measure, each hypothesis
is weighted by the measured reco impact parameter of each tau: the azimuth of
the hypothesis tau direction around the reco leading core track is compared
with the azimuth of the PV-referenced IP, through a response density fitted on
the TRAIN split only (truth tau direction vs reco IP, in |IP| bins).
Additional measures: the legacy (beam-spot z0) IP, a matched shuffle of the
PV IP measurement, and a hypothesis-independent parent-mass constraint
m_tautau ~ 91.19 GeV (same for H and Z), alone and with the IP.

Original docstring:
Per-event polarimeter moments on the MET-constrained null space.

For every event this accumulates, in one common observable frame,

    m-_i = E_mu[h-_i] ,  m+_i = E_mu[h+_i] ,  M_ij = E_mu[h-_i h+_j] ,

from which every estimator of this run follows: the joint pair statistic
E[T] = sum_i s_i M_ii, the marginal one T(m-, m+), their difference, which is
the posterior covariance a per-side point estimate throws away, and the
null-space likelihood ratio, which needs only M because the unpolarised joint
decay density carries the factor 1 + sum_ij C_ij h-_i h+_j.

The reference frame is the n,r,k basis of the measure-averaged tau pair, built
from the hypotheses themselves and therefore using no truth input.  Each
hypothesis is evaluated in its own basis by `HybridPolarimeter` and rotated into
that frame, because averaging vectors written in different bases is meaningless.

Three configurations share this code:

  default        MET constrained, the two-dimensional solution set
  --independent  the same per-side families with the MET constraint dropped
  --met-sigma    MET smeared isotropically, for the resolution requirement

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms
import nullspace as ns

MEASURES = ('flat', 'ip', 'ip_shuffle', 'ip_legacy', 'mass', 'ip_mass')
IP_EDGES_UM = np.array([0, 10, 20, 40, 80, 160, 320, 1e9])
PHI_BINS = 48
PARENT_MASS = 91.1874
SHEETS = ((0, 0), (0, 1), (1, 0), (1, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def track_frame(t: np.ndarray):
    e1 = np.cross(np.broadcast_to(np.array([0.0, 0.0, 1.0]), t.shape), t)
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-300)
    e2 = np.cross(t, e1)
    return e1, e2


def azimuth_in_frame(v: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sum(v * e2, -1), np.sum(v * e1, -1))


def ip_measurements(tracks: dict, ids: np.ndarray) -> dict:
    pv, t, n_core = tracks['pv'], tracks['trk'], tracks['n_core']
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    bxy = np.median(pv[:, :2], axis=0)
    bz = -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))
    theta, phi, d0, z0 = (lead[ids][..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + np.array([bxy[0], bxy[1], bz])
    ip = pca - pv[ids][:, None, :]
    ip -= np.sum(ip * tdir, -1, keepdims=True) * tdir
    legacy = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0 * np.sin(theta)], -1)
    legacy -= np.sum(legacy * tdir, -1, keepdims=True) * tdir
    e1, e2 = track_frame(tdir)
    return {'tdir': tdir, 'e1': e1, 'e2': e2,
            'psi': azimuth_in_frame(ip, e1, e2), 'mag_um': np.linalg.norm(ip, axis=-1) * 1e3,
            'psi_legacy': azimuth_in_frame(legacy, e1, e2),
            'mag_legacy_um': np.linalg.norm(legacy, axis=-1) * 1e3,
            'n_core': n_core[ids], 'available': ok[ids] & (n_core[ids] > 0)}


def fit_response(meas: dict, truth_dir: np.ndarray, psi_key: str, mag_key: str) -> np.ndarray:
    """Histogram density of (psi_truth - psi_meas) per (prong class, |IP| bin); TRAIN only."""
    psi_true = azimuth_in_frame(truth_dir, meas['e1'], meas['e2'])
    delta = np.angle(np.exp(1j * (psi_true - meas[psi_key])))
    cls = (meas['n_core'] >= 2).astype(int)
    b = np.clip(np.digitize(meas[mag_key], IP_EDGES_UM) - 1, 0, len(IP_EDGES_UM) - 2)
    table = np.ones((2, len(IP_EDGES_UM) - 1, PHI_BINS))  # add-one smoothing
    good = meas['available'] & np.isfinite(delta)
    idx = np.floor((delta[good] + np.pi) / (2 * np.pi) * PHI_BINS).astype(int) % PHI_BINS
    np.add.at(table, (cls[good], b[good], idx), 1.0)
    density = table / table.sum(-1, keepdims=True) * PHI_BINS / (2 * np.pi)
    return np.log(density * 2 * np.pi)  # log ratio to uniform


def ip_log_weight(tau_dir: np.ndarray, meas: dict, block: np.ndarray, response: np.ndarray,
                  psi_key: str, mag_key: str) -> np.ndarray:
    """tau_dir (...,R,2,3) -> summed log-likelihood ratio (...,R)."""
    e1, e2 = meas['e1'][block], meas['e2'][block]
    psi_h = azimuth_in_frame(tau_dir, e1, e2)
    delta = np.angle(np.exp(1j * (psi_h - meas[psi_key][block])))
    idx = np.floor((delta + np.pi) / (2 * np.pi) * PHI_BINS).astype(int) % PHI_BINS
    cls = (meas['n_core'][block] >= 2).astype(int)
    b = np.clip(np.digitize(meas[mag_key][block], IP_EDGES_UM) - 1, 0, len(IP_EDGES_UM) - 2)
    value = response[cls, b, idx]
    value = np.where(meas['available'][block], value, 0.0)
    return value.sum(-1)


class Accumulator:
    """Weighted first and second moments of h, one bank per measure."""

    def __init__(self, rows: int):
        self.sum_w = {m: np.zeros(rows) for m in MEASURES}
        self.sum_w2 = {m: np.zeros(rows) for m in MEASURES}
        self.first = {m: np.zeros((rows, 2, 3)) for m in MEASURES}
        self.second = {m: np.zeros((rows, 3, 3)) for m in MEASURES}
        self.t_own = {m: np.zeros(rows) for m in MEASURES}
        self.t_square = {m: np.zeros(rows) for m in MEASURES}
        self.self_pair = {m: np.zeros((rows, 2, 3, 3)) for m in MEASURES}
        # Per mass-shell-root sheet, to separate the discrete ambiguity from the
        # continuous one: the solution set is four sheets over a small region.
        self.sheet_w = {m: np.zeros((rows, 4)) for m in MEASURES}
        self.sheet_t = {m: np.zeros((rows, 4)) for m in MEASURES}
        self.sheet_t2 = {m: np.zeros((rows, 4)) for m in MEASURES}
        self.count = np.zeros(rows, dtype=np.int64)

    def add(self, measure: str, weight: np.ndarray, h: np.ndarray, t_own: np.ndarray,
            block: np.ndarray, sheet: np.ndarray) -> None:
        """`weight` (H,R), `h` (H,R,2,3) in the reference frame, `t_own` (H,R)."""
        self.sum_w[measure][block] += weight.sum(axis=0)
        self.sum_w2[measure][block] += (weight ** 2).sum(axis=0)
        self.first[measure][block] += np.einsum('hr,hrsi->rsi', weight, h)
        self.second[measure][block] += np.einsum('hr,hri,hrj->rij', weight,
                                                 h[:, :, 0], h[:, :, 1])
        self.t_own[measure][block] += np.einsum('hr,hr->r', weight, t_own)
        reference_t = ms.transverse_statistic(h)
        self.t_square[measure][block] += np.einsum('hr,hr->r', weight, reference_t ** 2)
        for side in (0, 1):
            self.self_pair[measure][block, side] += np.einsum(
                'hr,hri,hrj->rij', weight, h[:, :, side], h[:, :, side])
        for index in range(4):
            rows_of_sheet = sheet == index
            if not rows_of_sheet.any():
                continue
            part = weight[rows_of_sheet]
            self.sheet_w[measure][block, index] += part.sum(axis=0)
            self.sheet_t[measure][block, index] += (part * reference_t[rows_of_sheet]).sum(axis=0)
            self.sheet_t2[measure][block, index] += (
                part * reference_t[rows_of_sheet] ** 2).sum(axis=0)

    def normalised(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {'count': self.count}
        for measure in MEASURES:
            norm = np.maximum(self.sum_w[measure], 1e-300)
            out[f'first_{measure}'] = self.first[measure] / norm[:, None, None]
            out[f'second_{measure}'] = self.second[measure] / norm[:, None, None]
            out[f't_own_{measure}'] = self.t_own[measure] / norm
            out[f't_square_{measure}'] = self.t_square[measure] / norm
            out[f'self_pair_{measure}'] = self.self_pair[measure] / norm[:, None, None, None]
            sheet_norm = np.maximum(self.sheet_w[measure], 1e-300)
            out[f'sheet_weight_{measure}'] = self.sheet_w[measure] / norm[:, None]
            out[f'sheet_t_{measure}'] = self.sheet_t[measure] / sheet_norm
            out[f'sheet_t_square_{measure}'] = self.sheet_t2[measure] / sheet_norm
            out[f'sum_w_{measure}'] = self.sum_w[measure]
            out[f'ess_{measure}'] = np.where(
                self.sum_w2[measure] > 0.0,
                self.sum_w[measure] ** 2 / np.maximum(self.sum_w2[measure], 1e-300), 0.0)
        return out


def build_sheets(visible: np.ndarray, met: np.ndarray, nu_t_pair: np.ndarray) -> dict:
    """Four longitudinal sheets above fixed transverse momenta on both sides."""
    solved = ns.solve_nu_z(visible[None], nu_t_pair, np.asarray(ms.UPSTREAM_TAU_MASS))
    nz, valid_root = solved['nz'], solved['valid']
    return {
        'nu': np.stack([np.concatenate(
            (nu_t_pair, np.stack((nz[..., 0, i], nz[..., 1, j]), axis=-1)[..., None]), axis=-1)
            for i, j in SHEETS], axis=1),
        'valid': np.stack([valid_root[..., 0, i] & valid_root[..., 1, j] for i, j in SHEETS],
                          axis=1),
        'discriminant': solved['discriminant'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', default='met')
    parser.add_argument('--independent', action='store_true',
                        help='drop the MET constraint and draw the two sides independently')
    parser.add_argument('--met-sigma', type=float, default=0.0,
                        help='isotropic Gaussian smearing of MET in GeV')
    parser.add_argument('--proposal-lattice', type=int, default=40,
                        help='jittered lattice side; its square is drawn from the proposal')
    parser.add_argument('--keep', type=int, default=128,
                        help='accepted draws retained per event for the polarimeter')
    parser.add_argument('--row-chunk', type=int, default=1500)
    parser.add_argument('--hypothesis-chunk', type=int, default=64)
    parser.add_argument('--clip-quantile', type=float, default=0.99)
    parser.add_argument('--rows', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--nn-root', type=Path, default=ns.DEFAULT_NN_ROOT)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--ladder-train', type=Path, default=Path(
        '/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/train.npz'))
    parser.add_argument('--mass-sigma', type=float, default=3.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    surface = ns.load_surface(args.surface)
    polarimeter, origin = ns.load_polarimeter(args.nn_root)
    n_rows = len(surface['modes']) if not args.rows else args.rows

    truth_nu = surface['nu4'][:n_rows][..., :3]
    visible = surface['visible'][:n_rows]
    met_exact = truth_nu[:, 0, :2] + truth_nu[:, 1, :2]
    rng = np.random.default_rng(args.seed)
    met = met_exact.copy()
    if args.met_sigma > 0.0:
        met += rng.normal(0.0, args.met_sigma, size=met.shape)

    tracks = dict(np.load(args.tracks))
    with np.load(args.ladder_train) as tr:
        train_ids = np.asarray(tr['global_indices'])
        train_truth = tr['truth_visible_tau_lab4'][..., :3] + tr['truth_neutrino_lab4'][..., :3]
    train_truth /= np.linalg.norm(train_truth, axis=-1, keepdims=True)
    train_meas = ip_measurements(tracks, train_ids)
    response = fit_response(train_meas, train_truth, 'psi', 'mag_um')
    response_legacy = fit_response(train_meas, train_truth, 'psi_legacy', 'mag_legacy_um')
    meas = ip_measurements(tracks, surface['global_indices'][:n_rows].astype(np.int64))
    shuffle_rng = np.random.default_rng(args.seed + 7)
    meas['psi_shuffle'] = meas['psi'].copy()
    meas['mag_shuffle_um'] = meas['mag_um'].copy()
    key = (np.minimum(meas['n_core'], 3) * 10
           + np.clip(np.digitize(meas['mag_um'], IP_EDGES_UM) - 1, 0, 6)).reshape(-1)
    flat_psi = meas['psi_shuffle'].reshape(-1)
    for k in np.unique(key):
        where = np.flatnonzero(key == k)
        flat_psi[where] = flat_psi[shuffle_rng.permutation(where)]
    meas['psi_shuffle'] = flat_psi.reshape(meas['psi'].shape)
    # sanity: truth-direction log-likelihood ratio on this cohort (should be > 0 on average)
    truth_dir_rows = surface['tau'][:n_rows, :, :3]
    truth_dir_rows = truth_dir_rows / np.linalg.norm(truth_dir_rows, axis=-1, keepdims=True)
    all_rows = np.arange(n_rows)
    truth_llr = {name: float(np.mean(ip_log_weight(truth_dir_rows, meas, all_rows, resp, pk, mk)))
                 for name, resp, pk, mk in (('ip', response, 'psi', 'mag_um'),
                                            ('ip_shuffle', response, 'psi_shuffle', 'mag_um'),
                                            ('ip_legacy', response_legacy, 'psi_legacy', 'mag_legacy_um'))}
    print(json.dumps({'truth_mean_log_likelihood_ratio': truth_llr}), flush=True)
    accumulator = Accumulator(n_rows)
    reference_basis = np.zeros((n_rows, 3, 3))
    truth_h_reference = np.zeros((n_rows, 2, 3))
    truth_h_closure = np.zeros(n_rows)
    proposal_area = np.zeros(n_rows)
    region_area = np.zeros(n_rows)
    acceptance = np.zeros(n_rows)
    retained = np.zeros(n_rows, dtype=np.int64)
    sheet_multiplicity = np.zeros(n_rows)
    ellipse_geometry = np.zeros((n_rows, 2, 2))

    for start in range(0, n_rows, args.row_chunk):
        stop = min(start + args.row_chunk, n_rows)
        block = np.arange(start, stop)
        rows = len(block)
        vis = visible[block]
        met_block = met[block]
        nu_truth = truth_nu[block]
        unit = ms.lattice(args.proposal_lattice, rows, rng).transpose(1, 0, 2)

        if args.independent:
            frame0 = ms.ellipse_frame(vis[:, 0])
            frame1 = ms.ellipse_frame(vis[:, 1])
            other = ms.lattice(args.proposal_lattice, rows, rng).transpose(1, 0, 2)
            nu_t_pair = np.stack((ms.disk_to_plane(frame0, ms.unit_disk(unit)),
                                  ms.disk_to_plane(frame1, ms.unit_disk(other))),
                                 axis=-2)[:args.keep]
            sheets = build_sheets(vis, met_block, nu_t_pair)
            acceptance[block] = 1.0
            proposal_area[block] = frame0['area']
            region_area[block] = frame0['area']
            retained[block] = nu_t_pair.shape[0]
            frames = (frame0, frame1)
        else:
            drawn = ms.sample_region(vis, met_block, unit)
            index = ms.compact(drawn['accepted'], args.keep, rng)
            nu_t = np.take_along_axis(drawn['nu_t'], np.maximum(index, 0)[..., None], axis=0)
            nu_t_pair = np.stack((nu_t, met_block[None] - nu_t), axis=-2)
            sheets = build_sheets(vis, met_block, nu_t_pair)
            sheets['valid'] &= (index >= 0)[:, None, :]
            acceptance[block] = drawn['accepted'].mean(axis=0)
            proposal_area[block] = drawn['spec']['area']
            region_area[block] = acceptance[block] * drawn['spec']['area']
            retained[block] = (index >= 0).sum(axis=0)
            frames = (drawn['spec']['ellipse0'], drawn['spec']['ellipse1'])
        for side, frame in enumerate(frames):
            ellipse_geometry[block, side, 0] = frame['semi_major']
            ellipse_geometry[block, side, 1] = frame['semi_minor']

        nu = sheets['nu']                                       # (S,4,R,2,3)
        ok = sheets['valid']                                    # (S,4,R)
        accumulator.count[block] = ok.sum(axis=(0, 1))
        sheet_multiplicity[block] = ok.sum(axis=(0, 1)) / np.maximum(retained[block], 1)

        # The solid-angle weight depends on the draw, not on the sheet, so the
        # clip has to be set on the distinct per-draw values: taking the
        # quantile over the four-fold replicated array puts it on the maximum.
        per_draw = ms.solid_angle_weight(sheets['discriminant'])
        per_draw = np.where(ok.any(axis=1) & np.isfinite(per_draw), per_draw, 0.0)
        limit = np.zeros(rows)
        for r in range(rows):
            positive = per_draw[:, r][per_draw[:, r] > 0.0]
            limit[r] = np.quantile(positive, args.clip_quantile) if positive.size else 0.0
        weight_solid = np.broadcast_to(per_draw[:, None], ok.shape).copy()
        weight_solid = np.where(ok, weight_solid, 0.0)
        tau_h = vis[None, None, :, :, :3] + nu                  # (S,4,R,2,3)
        tau_dir = tau_h / np.maximum(np.linalg.norm(tau_h, axis=-1, keepdims=True), 1e-300)
        log_ip = ip_log_weight(tau_dir, meas, block, response, 'psi', 'mag_um')
        log_sh = ip_log_weight(tau_dir, meas, block, response, 'psi_shuffle', 'mag_um')
        log_lg = ip_log_weight(tau_dir, meas, block, response_legacy, 'psi_legacy', 'mag_legacy_um')
        e_nu = np.linalg.norm(nu, axis=-1)
        pair4 = np.concatenate((tau_h.sum(-2), (vis[None, None, :, :, 3] + e_nu).sum(-1)[..., None]), -1)
        m_h = np.sqrt(np.maximum(pair4[..., 3] ** 2 - np.sum(pair4[..., :3] ** 2, -1), 0.0))
        log_m = -0.5 * ((m_h - PARENT_MASS) / args.mass_sigma) ** 2

        def bank(log_w):
            log_w = np.where(ok, log_w, -np.inf)
            peak = np.max(log_w.reshape(-1, rows), axis=0)
            peak = np.where(np.isfinite(peak), peak, 0.0)
            w = np.exp(log_w - peak[None, None])
            return np.where(ok, w, 0.0).reshape(-1, rows)
        banks = {'flat': ok.astype(np.float64).reshape(-1, rows),
                 'ip': bank(log_ip), 'ip_shuffle': bank(log_sh), 'ip_legacy': bank(log_lg),
                 'mass': bank(log_m), 'ip_mass': bank(log_ip + log_m)}

        # Reference frame: the flat-measure average tau pair, hypotheses only.
        tau_sum = np.zeros((rows, 2, 4))
        for sheet in range(4):
            nu4 = np.concatenate(
                (nu[:, sheet], np.linalg.norm(nu[:, sheet], axis=-1)[..., None]), axis=-1)
            tau_sum += np.sum(np.where(ok[:, sheet][..., None, None], vis[None] + nu4, 0.0),
                              axis=0)
        basis = ns.pair_basis(tau_sum / np.maximum(accumulator.count[block], 1)[:, None, None])
        reference_basis[block] = basis

        # Truth closure, and the size of the frame choice, in one extra hypothesis.
        truth_result = ns.evaluate_h(polarimeter, surface, nu_truth[None], block)
        truth_h_closure[block] = np.abs(
            truth_result['h'][0] - surface['h_ref'][block].astype(np.float64)).max(axis=(-1, -2))
        truth_h_reference[block] = ms.to_reference_basis(
            truth_result['h'], truth_result['hypothesis_basis'], basis[None])[0]

        flat_nu = nu.reshape(-1, rows, 2, 3)
        flat_ok = ok.reshape(-1, rows)
        flat_nu = np.where(flat_ok[..., None, None],
                           flat_nu, np.broadcast_to(nu_truth[None], flat_nu.shape))

        for h_start in range(0, flat_nu.shape[0], args.hypothesis_chunk):
            h_stop = min(h_start + args.hypothesis_chunk, flat_nu.shape[0])
            result = ns.evaluate_h(polarimeter, surface, flat_nu[h_start:h_stop], block)
            good = flat_ok[h_start:h_stop] & result['valid']
            h_reference = ms.to_reference_basis(result['h'], result['hypothesis_basis'],
                                                basis[None])
            t_own = ms.transverse_statistic(result['h'])
            sheet_index = np.arange(h_start, h_stop) % 4
            for measure, weights in banks.items():
                accumulator.add(measure, np.where(good, weights[h_start:h_stop], 0.0),
                                h_reference, t_own, block, sheet_index)
        print(json.dumps({'rows_done': int(stop), 'elapsed_s': round(time.time() - started, 1)}),
              flush=True)

    payload = accumulator.normalised()
    payload.update({
        'reference_basis': reference_basis,
        'truth_h_canonical': surface['h_ref'][:n_rows].astype(np.float64),
        'truth_h_reference': truth_h_reference,
        'truth_h_closure': truth_h_closure,
        'labels': surface['labels'][:n_rows],
        'weights': surface['weights'][:n_rows].astype(np.float64),
        'modes': surface['modes'][:n_rows],
        'global_indices': surface['global_indices'][:n_rows],
        'proposal_area': proposal_area,
        'region_area': region_area,
        'acceptance': acceptance,
        'retained': retained,
        'ellipse_geometry': ellipse_geometry,
        'sheet_multiplicity': sheet_multiplicity,
        'met_used': met,
        'met_exact': met_exact,
    })
    np.savez_compressed(args.output / f'{args.tag}_moments.npz', **payload)

    summary = {
        'tag': args.tag,
        'rows': int(n_rows),
        'independent': bool(args.independent),
        'met_sigma_GeV': args.met_sigma,
        'proposal_lattice': args.proposal_lattice,
        'proposal_draws_per_event': int(args.proposal_lattice ** 2),
        'retained_per_event': args.keep,
        'clip_quantile': args.clip_quantile,
        'seed': args.seed,
        'inputs': {'surface': str(args.surface), 'surface_sha256': sha256_file(args.surface),
                   'hybrid_polarimeter': str(origin),
                   'hybrid_polarimeter_sha256': sha256_file(origin)},
        'closure': {
            'truth_h_max_abs_deviation': float(np.nanmax(truth_h_closure)),
            'truth_h_median_abs_deviation': float(np.nanmedian(truth_h_closure)),
        },
        'region': {
            'proposal_acceptance_mean': float(acceptance.mean()),
            'proposal_acceptance_quantiles': np.quantile(acceptance, [0.05, 0.5, 0.95]).tolist(),
            'region_area_GeV2_quantiles': np.quantile(region_area, [0.05, 0.5, 0.95]).tolist(),
            'ellipse_semi_major_GeV_median': float(np.median(ellipse_geometry[..., 0])),
            'ellipse_semi_minor_GeV_median': float(np.median(ellipse_geometry[..., 1])),
            'retained_quantiles': np.quantile(retained, [0.01, 0.05, 0.5, 0.95]).tolist(),
            'sheet_multiplicity_mean': float(sheet_multiplicity.mean()),
            'valid_hypotheses_quantiles': np.quantile(
                payload['count'], [0.01, 0.05, 0.5, 0.95]).tolist(),
            'events_with_no_valid_hypothesis': int((payload['count'] == 0).sum()),
        },
        'effective_sample_size': {
            measure: {'mean': float(np.mean(payload[f'ess_{measure}'])),
                      'quantiles': np.quantile(payload[f'ess_{measure}'],
                                               [0.05, 0.5, 0.95]).tolist()}
            for measure in MEASURES},
        'elapsed_seconds': time.time() - started,
        'hostname': platform.node(),
        'script_sha256': sha256_file(Path(__file__)),
        'metspace_sha256': sha256_file(Path(__file__).resolve().parent / 'metspace.py'),
        'nullspace_sha256': sha256_file(Path(ns.__file__)),
    }
    (args.output / f'{args.tag}_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
