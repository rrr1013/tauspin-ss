"""Oracle ladder, headroom shares and the direction-resolution requirement.

Nothing here re-runs the polarimeter.  Every number is a function of the stored
per-event moments, so the oracles and the resolution scans are all evaluated on
exactly the same events and the same draws.

Two resolution scans are reported and they answer different questions.

`grid`
    The full Bayesian reweighting of every hypothesis by an idealised tau
    direction measurement.  It uses the measurement for everything it is worth,
    both to pick the mass-shell root and to squeeze the continuous coordinate,
    but it is limited by how finely the Monte Carlo draws sample the region, so
    its effective sample size has to be read next to it.

`sheet`
    The same measurement used *only* to choose the root.  The posterior is then
    a mixture of the four sheet-conditional measures, and since moments of a
    mixture are mixtures of moments, this is exact at every resolution with no
    Monte Carlo limit at all.  Its two limits are checks: sigma -> 0 must give
    the truth-sheet oracle and sigma -> infinity must give the unconditioned
    solution set.

The difference between the two curves is what the measurement buys beyond the
discrete choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'met_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms
import run_m2_report as m2

MODE_NAMES = m2.MODE_NAMES
ORACLES = ('o0', 'o1', 'o2a', 'o2b')
MEASURES = ('flat', 'solid', 'solid_clipped')

# Quoted from the runs that produced them, not recomputed here.
REFERENCE_AUC = {
    'met_nullspace_ceiling_likelihood_ratio': 0.62061,   # MET null space run, flat
    'met_nullspace_joint_T': 0.61877,
    'met_nullspace_marginal_T': 0.61807,
    'truth_h_likelihood_ratio': 0.71858,
    'truth_h_transverse_statistic': 0.70640,
    'azimuth_only_marginalised': 0.6214,                 # azimuth run, scan C
    'full_reco_point_h_fixed_readout': 0.617348,         # rho diagnostic run, weighted
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def bank_estimators(data: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    out = ms.moment_estimators(data[f'first_{name}'][:, 0], data[f'first_{name}'][:, 1],
                               data[f'second_{name}'])
    out['t_own'] = data[f't_own_{name}']
    return out


def moment_estimators_from(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    return ms.moment_estimators(first[:, 0], first[:, 1], second)


def truth_estimators(h: np.ndarray) -> dict[str, np.ndarray]:
    """Exact polarimeter vectors: the second moment is just the outer product."""
    second = np.einsum('ri,rj->rij', h[:, 0], h[:, 1])
    return ms.moment_estimators(h[:, 0], h[:, 1], second)


def sheet_prior(data: dict[str, np.ndarray]) -> np.ndarray:
    """Weight each sheet carries under the unconditioned flat measure, (R,4)."""
    weight = np.stack([data[f'sum_w_flat_sheet{index}'] for index in range(4)], axis=-1)
    total = np.maximum(weight.sum(axis=1, keepdims=True), 1e-300)
    return weight / total


def sheet_posterior(data: dict[str, np.ndarray], sigma_rad: np.ndarray) -> np.ndarray:
    """Posterior over the four sheets given one tau-direction measurement per side.

    Everything lives in the tangent plane of the truth direction, the same frame
    `run_s1_moments` uses.  The truth root sits at the origin of that plane by
    construction and the other root sits at the stored offset, so this uses the
    real geometry rather than an isotropy argument; the measurement is the same
    stored noise scaled by the resolution, so the two resolution curves see the
    identical measurement.  `sigma_rad` is (R,2) in radians; a zero means that
    side is not measured.  The approximation is that the root separation is taken
    at the truth transverse momentum and held fixed across the region.
    """
    prior = sheet_prior(data)
    epsilon = sigma_rad[..., None] * data['direction_noise']     # (R,2sides,2)
    offset = data['root_offset']                                 # (R,2sides,2)
    inverse = np.where(sigma_rad > 0.0, 1.0 / np.maximum(sigma_rad, 1e-300) ** 2, 0.0)
    distance_correct = np.sum(epsilon ** 2, axis=-1) * inverse
    distance_other = np.sum((offset - epsilon) ** 2, axis=-1) * inverse
    exponent = np.zeros(prior.shape)
    truth_root = data['truth_root']                              # (R,2)
    for index in range(4):
        for side, root in enumerate((index // 2, index % 2)):
            correct = root == truth_root[:, side]
            exponent[:, index] += -0.5 * np.where(correct, distance_correct[:, side],
                                                  distance_other[:, side])
    exponent -= exponent.max(axis=1, keepdims=True)
    posterior = prior * np.exp(np.maximum(exponent, -700.0))
    return posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)


def uniform_sigma(rows: int, sigma_mrad: float, sides: int = 2) -> np.ndarray:
    out = np.zeros((rows, 2))
    out[:, :sides] = sigma_mrad * 1e-3
    return out


def vertex_sigma(data: dict[str, np.ndarray], micron: float) -> np.ndarray:
    """Per-side direction resolution from a decay-vertex position resolution."""
    length_um = np.maximum(data['decay_length_um'], 1e-6)
    return np.where(data['has_vertex'], micron / length_um, 0.0)


def mixed_moments(data: dict[str, np.ndarray], posterior: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Moments of a mixture of the sheet-conditional measures."""
    first = np.zeros(data['first_flat_sheet0'].shape)
    second = np.zeros(data['second_flat_sheet0'].shape)
    for index in range(4):
        weight = posterior[:, index]
        first += weight[:, None, None] * data[f'first_flat_sheet{index}']
        second += weight[:, None, None] * data[f'second_flat_sheet{index}']
    return first, second


def eligible(data: dict[str, np.ndarray]) -> np.ndarray:
    ok = data['count'] > 0
    for name in [f'{m}_{o}' for m in MEASURES for o in ORACLES] + \
                [f'flat_sheet{i}' for i in range(4)]:
        ok &= data[f'sum_w_{name}'] > 0.0
        values = bank_estimators(data, name)
        for key in ('joint', 'marginal', 'likelihood_ratio'):
            ok &= np.isfinite(values[key])
    ok &= np.isfinite(data['root_angle']).all(axis=1)
    ok &= data['o3_valid'] > 0
    return ok


def auc_block(labels, weights, modes, score, mask) -> dict[str, object]:
    return m2.auc_block(labels, weights, modes, score, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--tag', default='hi')
    parser.add_argument('--cross-tag', default='base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load(args.moments / args.tag / f'{args.tag}_moments.npz')
    cross = None
    cross_path = args.moments / args.cross_tag / f'{args.cross_tag}_moments.npz'
    if cross_path.exists():
        cross = load(cross_path)

    labels, weights, modes = data['labels'], data['weights'], data['modes']
    mask = eligible(data)
    if cross is not None:
        mask &= eligible(cross)

    report: dict[str, object] = {
        'tag': args.tag,
        'rows': int(len(labels)),
        'eligible_events': int(mask.sum()),
        'excluded_events': int((~mask).sum()),
        'reference_auc_quoted': REFERENCE_AUC,
        'closure': {
            'truth_h_max_abs_deviation': float(np.nanmax(data['truth_h_closure'])),
            'root_residual_chosen_quantiles_GeV': np.quantile(
                data['root_residual_chosen'], [0.5, 0.95, 0.99, 1.0]).tolist(),
            'root_separation_quantiles_GeV': np.quantile(
                data['root_separation'], [0.01, 0.05, 0.5, 0.95]).tolist(),
            'near_degenerate_fraction_0p1GeV': float(
                (data['root_separation'] < 0.1).any(axis=-1)[mask].mean()),
            'o3_valid_sheets_mean': float(data['o3_valid'][mask].mean()),
            'o3_four_sheets_fraction': float((data['o3_valid'][mask] == 4).mean()),
        },
    }

    # ---- how much the strict cohort costs ----------------------------------
    # The ladder needs every sheet to be populated, because a sheet with no
    # hypotheses has no conditional moments to mix.  That is a real restriction
    # and it removes events where the kinematics already killed one root, so the
    # unconditioned ceiling is quoted on both cohorts.
    loose = data['count'] > 0
    for key in ('joint', 'marginal', 'likelihood_ratio'):
        loose &= np.isfinite(bank_estimators(data, 'flat_o0')[key])
    report['cohort_sensitivity'] = {
        'strict_events': int(mask.sum()),
        'loose_events': int(loose.sum()),
        'flat_o0_likelihood_ratio_strict': auc_block(
            labels, weights, modes, bank_estimators(data, 'flat_o0')['likelihood_ratio'],
            mask)['auc_unweighted'],
        'flat_o0_likelihood_ratio_loose': auc_block(
            labels, weights, modes, bank_estimators(data, 'flat_o0')['likelihood_ratio'],
            loose)['auc_unweighted'],
        'flat_o0_joint_loose': auc_block(
            labels, weights, modes, bank_estimators(data, 'flat_o0')['joint'],
            loose)['auc_unweighted'],
        'flat_o0_marginal_loose': auc_block(
            labels, weights, modes, bank_estimators(data, 'flat_o0')['marginal'],
            loose)['auc_unweighted'],
        'empty_sheet_events': int((loose & ~mask).sum()),
    }

    # ---- the oracle ladder -------------------------------------------------
    truth_reference = truth_estimators(data['truth_h_reference'])
    truth_canonical = truth_estimators(data['truth_h_canonical'].astype(np.float64))
    o3 = moment_estimators_from(data['o3_first'], data['o3_second'])

    ladder: dict[str, object] = {}
    scores: dict[str, np.ndarray] = {}
    for measure in MEASURES:
        for oracle in ORACLES:
            values = bank_estimators(data, f'{measure}_{oracle}')
            for key in ('joint', 'marginal', 'likelihood_ratio'):
                name = f'{measure}_{oracle}_{key}'
                scores[name] = values[key]
                ladder[name] = auc_block(labels, weights, modes, values[key], mask)
    for key in ('joint', 'marginal', 'likelihood_ratio'):
        scores[f'o3_{key}'] = o3[key]
        ladder[f'o3_{key}'] = auc_block(labels, weights, modes, o3[key], mask)
        scores[f'o4_{key}'] = truth_reference[key]
        ladder[f'o4_{key}'] = auc_block(labels, weights, modes, truth_reference[key], mask)
        scores[f'o4canonical_{key}'] = truth_canonical[key]
        ladder[f'o4canonical_{key}'] = auc_block(labels, weights, modes,
                                                 truth_canonical[key], mask)
    report['oracle_ladder'] = ladder

    # ---- primary endpoint and the headroom shares --------------------------
    primary = {}
    for key in ('likelihood_ratio', 'joint'):
        primary[f'o1_minus_o0_{key}'] = m2.bootstrap_auc_difference(
            labels, scores[f'flat_o1_{key}'], scores[f'flat_o0_{key}'], mask,
            draws=args.bootstrap, seed=args.seed)
        primary[f'o3_minus_o0_{key}'] = m2.bootstrap_auc_difference(
            labels, scores[f'o3_{key}'], scores[f'flat_o0_{key}'], mask,
            draws=args.bootstrap, seed=args.seed)
        primary[f'o2a_minus_o0_{key}'] = m2.bootstrap_auc_difference(
            labels, scores[f'flat_o2a_{key}'], scores[f'flat_o0_{key}'], mask,
            draws=args.bootstrap, seed=args.seed)
    report['primary_endpoint'] = primary

    shares = {}
    for measure in MEASURES:
        base_auc = ladder[f'{measure}_o0_likelihood_ratio']['auc_unweighted']
        top_auc = ladder['o4_likelihood_ratio']['auc_unweighted']
        headroom = top_auc - base_auc
        shares[measure] = {
            'base': base_auc, 'ceiling': top_auc, 'headroom': headroom,
            'f_discrete': (ladder[f'{measure}_o1_likelihood_ratio']['auc_unweighted']
                           - base_auc) / headroom,
            'f_one_side': (ladder[f'{measure}_o2a_likelihood_ratio']['auc_unweighted']
                           - base_auc) / headroom,
        }
    shares['flat']['f_continuous'] = (
        (ladder['o3_likelihood_ratio']['auc_unweighted']
         - ladder['flat_o0_likelihood_ratio']['auc_unweighted'])
        / (ladder['o4_likelihood_ratio']['auc_unweighted']
           - ladder['flat_o0_likelihood_ratio']['auc_unweighted']))
    report['headroom_shares'] = shares
    report['headroom_shares_note'] = (
        'f_discrete and f_continuous are not required to sum to one; conditioning '
        'on the sheet and conditioning on the continuous coordinate are neither '
        'commuting nor additive operations.')

    # ---- resolution requirement -------------------------------------------
    def grid_entry(name: str) -> dict[str, object]:
        values = bank_estimators(data, name)
        entry = {'bank': name,
                 'ess_mean': float(data[f'ess_{name}'][mask].mean()),
                 'ess_median': float(np.median(data[f'ess_{name}'][mask])),
                 'ess_median_by_pair': {
                     label: float(np.median(data[f'ess_{name}'][
                         mask & (modes == code).all(axis=1)]))
                     for code, label in MODE_NAMES.items()
                     if (mask & (modes == code).all(axis=1)).sum() >= 200},
                 'p_truth_sheet': float(np.take_along_axis(
                     data[f'sheet_weight_{name}'], data['truth_sheet'][:, None],
                     axis=1)[mask, 0].mean())}
        for key in ('joint', 'marginal', 'likelihood_ratio'):
            entry[key] = auc_block(labels, weights, modes, values[key], mask)
        return entry

    def sheet_entry(sigma_rad: np.ndarray) -> dict[str, object]:
        posterior = sheet_posterior(data, sigma_rad)
        first, second = mixed_moments(data, posterior)
        values = moment_estimators_from(first, second)
        entry = {'p_truth_sheet': float(np.take_along_axis(
            posterior, data['truth_sheet'][:, None], axis=1)[mask, 0].mean())}
        for key in ('joint', 'marginal', 'likelihood_ratio'):
            entry[key] = auc_block(labels, weights, modes, values[key], mask)
        return entry

    rows = len(labels)
    sigmas = [float(value) for value in data['sigma_mrad']]
    vertices = [float(value) for value in data['vertex_um']]

    curves: dict[str, list] = {'two_sided': [], 'one_sided': [], 'vertex': []}
    for sigma in sigmas:
        both = dict(sigma_mrad=sigma, **grid_entry(f'flat_sigma{sigma:g}'))
        both['sheet_only'] = sheet_entry(uniform_sigma(rows, sigma, sides=2))
        curves['two_sided'].append(both)
        one = dict(sigma_mrad=sigma, **grid_entry(f'flat_oneside{sigma:g}'))
        one['sheet_only'] = sheet_entry(uniform_sigma(rows, sigma, sides=1))
        curves['one_sided'].append(one)
    for micron in vertices:
        entry = dict(vertex_um=micron, **grid_entry(f'flat_vertex{micron:g}'))
        per_side = vertex_sigma(data, micron)
        entry['sheet_only'] = sheet_entry(per_side)
        measured = per_side[mask] > 0.0
        entry['sigma_alpha_mrad_median_measured_sides'] = float(
            1e3 * np.median(per_side[mask][measured]))
        entry['measured_side_fraction'] = float(measured.mean())
        curves['vertex'].append(entry)
    report['resolution_curves'] = curves

    # A denser sheet-only scan: it is exact at any resolution, so it costs nothing.
    fine = []
    for sigma in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 200.0):
        entry = dict(sigma_mrad=sigma, **sheet_entry(uniform_sigma(rows, sigma, sides=2)))
        fine.append(entry)
    report['sheet_only_resolution_curve'] = fine

    # Limit checks of the sheet mixture against the two oracles it must reproduce.
    first_tiny, second_tiny = mixed_moments(data, sheet_posterior(
        data, uniform_sigma(rows, 1e-6)))
    first_huge, second_huge = mixed_moments(data, sheet_posterior(
        data, uniform_sigma(rows, 1e6)))
    report['sheet_mixture_limits'] = {
        'sigma_to_zero_auc': auc_block(labels, weights, modes,
                                       moment_estimators_from(first_tiny, second_tiny)[
                                           'likelihood_ratio'], mask)['auc_unweighted'],
        'o1_auc': ladder['flat_o1_likelihood_ratio']['auc_unweighted'],
        'sigma_to_infinity_auc': auc_block(labels, weights, modes,
                                           moment_estimators_from(first_huge, second_huge)[
                                               'likelihood_ratio'], mask)['auc_unweighted'],
        'o0_auc': ladder['flat_o0_likelihood_ratio']['auc_unweighted'],
    }

    # ---- geometry of the discrete ambiguity --------------------------------
    root_angle_mrad = 1e3 * data['root_angle']
    geometry = {
        'root_angle_mrad_quantiles': np.quantile(
            root_angle_mrad[mask], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
        'intra_sheet_angle_mrad_quantiles': np.quantile(
            1e3 * np.take_along_axis(
                data['sheet_angle_rms'], data['truth_sheet'][:, None, None].repeat(2, axis=2),
                axis=1)[mask, 0], [0.05, 0.5, 0.95]).tolist(),
        'by_mode': {},
    }
    for code, name in MODE_NAMES.items():
        side_mask = mask[:, None] & (modes == code)
        if side_mask.sum() < 200:
            continue
        geometry['by_mode'][name] = {
            'sides': int(side_mask.sum()),
            'root_angle_mrad_quantiles': np.quantile(
                root_angle_mrad[side_mask], [0.05, 0.5, 0.95]).tolist(),
            'intra_sheet_angle_mrad_median': float(np.median(
                1e3 * np.take_along_axis(
                    data['sheet_angle_rms'],
                    data['truth_sheet'][:, None, None].repeat(2, axis=2),
                    axis=1)[:, 0][side_mask])),
        }
    report['geometry'] = geometry

    # ---- how much of the spread on the solution set is discrete ------------
    spread = {}
    for measure in ('flat',):
        joint = np.einsum('i,rii->r', ms.SPIN_SIGNATURE, data[f'second_{measure}_o0'])
        total = np.maximum(data[f't_square_{measure}_o0'] - joint ** 2, 0.0)
        sheet_w = data[f'sheet_weight_{measure}_o0']
        per_sheet_mean = np.stack(
            [np.einsum('i,rii->r', ms.SPIN_SIGNATURE, data[f'second_flat_sheet{i}'])
             for i in range(4)], axis=1)
        between = (sheet_w * (per_sheet_mean - joint[:, None]) ** 2).sum(axis=1)
        good = mask & (total > 0)
        spread[measure] = {
            'total_std_quantiles': np.quantile(np.sqrt(total[good]), [0.05, 0.5, 0.95]).tolist(),
            'between_sheet_fraction_mean': float((between[good] / total[good]).mean()),
            'between_sheet_fraction_median': float(np.median(between[good] / total[good])),
        }
    report['spread_decomposition'] = spread

    # ---- cross check against the coarser draw budget -----------------------
    if cross is not None:
        cross_block = {}
        for oracle in ORACLES:
            values = bank_estimators(cross, f'flat_{oracle}')
            cross_block[f'flat_{oracle}_likelihood_ratio'] = auc_block(
                cross['labels'], cross['weights'], cross['modes'],
                values['likelihood_ratio'], mask)['auc_unweighted']
        cross_block['ess_flat_o0'] = float(cross['ess_flat_o0'][mask].mean())
        cross_block['truth_h_max_abs_deviation'] = float(np.nanmax(cross['truth_h_closure']))
        report['cross_check_' + args.cross_tag] = cross_block

    sheet_mean_t = np.stack(
        [np.einsum('i,rii->r', ms.SPIN_SIGNATURE, data[f'second_flat_sheet{i}'])
         for i in range(4)], axis=1)

    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / 'scores.npz',
                        sheet_mean_t=sheet_mean_t,
                        t_square_o0=data['t_square_flat_o0'],
                        mask=mask, labels=labels, modes=modes, weights=weights,
                        root_angle=data['root_angle'],
                        root_offset=data['root_offset'],
                        decay_length_um=data['decay_length_um'],
                        has_vertex=data['has_vertex'],
                        truth_sheet=data['truth_sheet'],
                        sheet_weight_o0=data['sheet_weight_flat_o0'],
                        sheet_angle_rms=data['sheet_angle_rms'],
                        root_separation=data['root_separation'],
                        **{f'score_{k}': v for k, v in scores.items()})
    print(json.dumps({k: report[k] for k in
                      ('eligible_events', 'closure', 'headroom_shares',
                       'primary_endpoint', 'sheet_mixture_limits')}, indent=2))


if __name__ == '__main__':
    main()
