"""Estimators, AUCs and figures from the per-event null-space moments.

Everything this run compares is a function of the three stored moment objects
per event, so nothing here re-runs the polarimeter.  The primary comparison is
the joint pair estimator against the marginal one on exactly the same events and
the same draws; their difference is the posterior covariance between the two
sides that a per-side point estimate discards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms

MODE_NAMES = {0: '1p0n', 1: '1p1n', 3: '3p0n'}
COMPONENTS = ('n', 'r', 'k')
MEASURES = ('flat', 'solid', 'solid_clipped')

# Published reference points, quoted not recomputed.
REFERENCE_AUC = {
    'truth_h_transverse_statistic': 0.7064,      # azimuth null space run, unweighted, 59,390
    'azimuth_only_marginalised': 0.6214,         # same run, scan C, truth E_tau known
    'full_reco_point_h_fixed_readout': 0.617348,  # rho diagnostic run, weighted, 56,262
}


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ROC AUC by the rank identity with tie averaging."""
    order = np.argsort(scores, kind='mergesort')
    scores = scores[order]
    labels = labels[order]
    weights = weights[order].astype(np.float64)
    positive = weights * (labels == 1)
    negative = weights * (labels == 0)
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return float('nan')
    group = np.cumsum(np.r_[True, np.diff(scores) != 0]) - 1
    negative_per_group = np.bincount(group, weights=negative)
    below = np.r_[0.0, np.cumsum(negative_per_group)[:-1]]
    rank_weight = below[group] + 0.5 * negative_per_group[group]
    return float((positive * rank_weight).sum() / (total_positive * total_negative))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind='mergesort')
        out = np.empty(len(v))
        out[order] = np.arange(len(v), dtype=np.float64)
        return out
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1])


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def estimators(data: dict[str, np.ndarray], measure: str) -> dict[str, np.ndarray]:
    out = ms.moment_estimators(data[f'first_{measure}'][:, 0],
                               data[f'first_{measure}'][:, 1],
                               data[f'second_{measure}'])
    out['t_own'] = data[f't_own_{measure}']
    return out


def spread_decomposition(data: dict[str, np.ndarray], measure: str) -> dict[str, np.ndarray]:
    """How much the pair statistic still moves on the solution set, split into
    the part that separates the four mass-shell-root sheets and the part inside
    a sheet.  Missing keys mean the moments predate this decomposition."""
    if f't_square_{measure}' not in data:
        return {}
    joint = np.einsum('i,rii->r', ms.SPIN_SIGNATURE, data[f'second_{measure}'])
    total = np.maximum(data[f't_square_{measure}'] - joint ** 2, 0.0)
    sheet_w = data[f'sheet_weight_{measure}']
    sheet_t = data[f'sheet_t_{measure}']
    sheet_t2 = data[f'sheet_t_square_{measure}']
    within = (sheet_w * np.maximum(sheet_t2 - sheet_t ** 2, 0.0)).sum(axis=1)
    between = (sheet_w * (sheet_t - joint[:, None]) ** 2).sum(axis=1)
    return {'total_variance': total, 'within_sheet': within, 'between_sheet': between,
            'sheet_weight': sheet_w, 'sheet_mean': sheet_t}


def monte_carlo_covariance_noise(data: dict[str, np.ndarray], measure: str) -> np.ndarray:
    """Expected sampling error of the per-event covariance estimate, (R,3).

    The covariance is estimated from a finite number of draws, so a non-zero
    spread is expected even when the true covariance vanishes.  This is the
    scale that spread has to be compared against.
    """
    if f'self_pair_{measure}' not in data:
        return np.zeros(data['count'].shape + (3,))
    first = data[f'first_{measure}']
    self_pair = data[f'self_pair_{measure}']
    variance = np.stack([self_pair[:, side][:, [0, 1, 2], [0, 1, 2]] - first[:, side] ** 2
                         for side in (0, 1)], axis=1)
    variance = np.maximum(variance, 0.0)
    effective = np.maximum(data[f'ess_{measure}'], 1.0)[:, None]
    return np.sqrt(variance[:, 0] * variance[:, 1] / effective)


def eligible_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    ok = data['count'] > 0
    for measure in MEASURES:
        values = estimators(data, measure)
        for key in ('joint', 'marginal', 'likelihood_ratio'):
            ok &= np.isfinite(values[key])
        ok &= data[f'sum_w_{measure}'] > 0.0
    return ok


def auc_block(labels, weights, modes, score, mask) -> dict[str, object]:
    out = {
        'events': int(mask.sum()),
        'auc_unweighted': weighted_auc(labels[mask], score[mask], np.ones(int(mask.sum()))),
        'auc_weighted': weighted_auc(labels[mask], score[mask], weights[mask]),
    }
    for code, name in MODE_NAMES.items():
        pair = mask & (modes == code).all(axis=1)
        if pair.sum() >= 200:
            out[f'auc_{name}x{name}'] = weighted_auc(
                labels[pair], score[pair], np.ones(int(pair.sum())))
            out[f'events_{name}x{name}'] = int(pair.sum())
    return out


class RankedScore:
    """AUC for a fixed score under changing event weights.

    The bootstrap resamples events, not scores, so the sort and the tie grouping
    are computed once and only the weights move.
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray):
        self.order = np.argsort(scores, kind='mergesort')
        ordered = scores[self.order]
        self.labels = labels[self.order]
        self.group = np.cumsum(np.r_[True, np.diff(ordered) != 0]) - 1
        self.groups = int(self.group[-1]) + 1

    def auc(self, weights: np.ndarray) -> float:
        w = weights[self.order].astype(np.float64)
        positive = w * (self.labels == 1)
        negative = w * (self.labels == 0)
        total_positive, total_negative = positive.sum(), negative.sum()
        if total_positive <= 0 or total_negative <= 0:
            return float('nan')
        per_group = np.bincount(self.group, weights=negative, minlength=self.groups)
        below = np.r_[0.0, np.cumsum(per_group)[:-1]]
        return float((positive * (below[self.group] + 0.5 * per_group[self.group])).sum()
                     / (total_positive * total_negative))


def bootstrap_auc_difference(labels, score_a, score_b, mask, draws=1000,
                             seed=20260919) -> dict[str, float]:
    """Paired event bootstrap of AUC(a) - AUC(b) on the identical event set."""
    rng = np.random.default_rng(seed)
    index = np.flatnonzero(mask)
    ranked_a = RankedScore(labels[index], score_a[index])
    ranked_b = RankedScore(labels[index], score_b[index])
    unit = np.ones(len(index))
    point = ranked_a.auc(unit) - ranked_b.auc(unit)
    samples = np.empty(draws)
    for draw in range(draws):
        weight = np.bincount(rng.integers(0, len(index), len(index)),
                             minlength=len(index)).astype(np.float64)
        samples[draw] = ranked_a.auc(weight) - ranked_b.auc(weight)
    return {'difference': float(point),
            'ci_low': float(np.quantile(samples, 0.025)),
            'ci_high': float(np.quantile(samples, 0.975)),
            'p_two_sided_sign': float(2.0 * min((samples <= 0).mean(), (samples >= 0).mean())),
            'bootstrap_draws': draws, 'seed': seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    met = load(args.moments / 'met' / 'met_moments.npz')
    indep = load(args.moments / 'indep' / 'indep_moments.npz')
    sigmas = {}
    for path in sorted((args.moments).glob('sigma*/sigma*_moments.npz')):
        sigmas[float(path.stem.replace('sigma', '').replace('_moments', ''))] = load(path)

    labels, weights, modes = met['labels'], met['weights'], met['modes']
    truth_canonical = ms.transverse_statistic(met['truth_h_canonical'])
    truth_reference = ms.transverse_statistic(met['truth_h_reference'])
    # The optimal discriminant at truth level, so the ladder compares like with
    # like: the null-space likelihood ratio is optimal for its own information
    # set, and this is optimal for the exact one.
    truth_outer = np.einsum('ri,rj->rij', met['truth_h_canonical'][:, 0],
                            met['truth_h_canonical'][:, 1])
    truth_likelihood = (np.log(np.maximum(1.0 + np.einsum('ij,rij->r', ms.C_HIGGS, truth_outer),
                                          1e-12))
                        - np.log(np.maximum(1.0 + np.einsum('ij,rij->r', ms.C_Z, truth_outer),
                                            1e-12)))
    primary = eligible_mask(met)

    report: dict[str, object] = {
        'rows': int(len(labels)),
        'primary_cohort': int(primary.sum()),
        'dropped_no_valid_hypothesis': int((~primary).sum()),
        'reference_auc_quoted': REFERENCE_AUC,
        'truth_closure_max': float(np.max(met['truth_h_closure'])),
        'basis_check': {
            'truth_T_canonical_basis': auc_block(labels, weights, modes, truth_canonical, primary),
            'truth_T_reference_basis': auc_block(labels, weights, modes, truth_reference, primary),
            'truth_likelihood_ratio': auc_block(labels, weights, modes,
                                               truth_likelihood, primary),
            'correlation_canonical_reference': float(
                np.corrcoef(truth_canonical[primary], truth_reference[primary])[0, 1]),
        },
        'geometry': {
            'ellipse_semi_major_GeV_quantiles': np.quantile(
                met['ellipse_geometry'][..., 0], [0.05, 0.5, 0.95]).tolist(),
            'ellipse_semi_minor_GeV_quantiles': np.quantile(
                met['ellipse_geometry'][..., 1], [0.05, 0.5, 0.95]).tolist(),
            'region_area_GeV2_quantiles': np.quantile(
                met['region_area'], [0.05, 0.5, 0.95]).tolist(),
            'acceptance_quantiles': np.quantile(met['acceptance'], [0.05, 0.5, 0.95]).tolist(),
            'retained_quantiles': np.quantile(met['retained'], [0.01, 0.05, 0.5]).tolist(),
        },
        'measures': {},
        'independent_control': {},
        'met_resolution': {},
    }

    for measure in MEASURES:
        values = estimators(met, measure)
        block: dict[str, object] = {
            'joint': auc_block(labels, weights, modes, values['joint'], primary),
            'marginal': auc_block(labels, weights, modes, values['marginal'], primary),
            'likelihood_ratio': auc_block(labels, weights, modes,
                                          values['likelihood_ratio'], primary),
            'likelihood_ratio_measured_c': auc_block(
                labels, weights, modes, values['likelihood_ratio_measured_c'], primary),
            'joint_minus_marginal': bootstrap_auc_difference(
                labels, values['joint'], values['marginal'], primary, args.bootstrap),
            'likelihood_minus_joint': bootstrap_auc_difference(
                labels, values['likelihood_ratio'], values['joint'], primary, args.bootstrap),
            'correlation_with_truth': {
                name: {'pearson': float(np.corrcoef(values[name][primary],
                                                    truth_canonical[primary])[0, 1]),
                       'spearman': spearman(values[name][primary], truth_canonical[primary])}
                for name in ('joint', 'marginal', 'likelihood_ratio')},
            'effective_sample_size_mean': float(np.mean(met[f'ess_{measure}'][primary])),
        }
        noise = monte_carlo_covariance_noise(met, measure)[primary]
        spread = spread_decomposition(met, measure)
        if spread:
            block['pair_statistic_spread'] = {
                'total_std_mean': float(np.mean(np.sqrt(spread['total_variance'][primary]))),
                'total_std_quantiles': np.quantile(
                    np.sqrt(spread['total_variance'][primary]), [0.05, 0.5, 0.95]).tolist(),
                'between_sheet_fraction_mean': float(np.mean(
                    spread['between_sheet'][primary]
                    / np.maximum(spread['total_variance'][primary], 1e-12))),
                'sheet_weight_mean': spread['sheet_weight'][primary].mean(axis=0).tolist(),
            }
        covariance = values['covariance_diagonal'][primary]
        block['posterior_covariance'] = {
            component: {
                'monte_carlo_noise_mean': float(noise[:, i].mean()) if noise.any() else None,
                'mean': float(covariance[:, i].mean()),
                'std': float(covariance[:, i].std()),
                'quantiles': np.quantile(covariance[:, i], [0.05, 0.5, 0.95]).tolist(),
                'mean_H': float(covariance[labels[primary] == 1, i].mean()),
                'mean_Z': float(covariance[labels[primary] == 0, i].mean()),
            } for i, component in enumerate(COMPONENTS)}
        block['second_moment_mean'] = {
            'H': met[f'second_{measure}'][primary][labels[primary] == 1].mean(axis=0).tolist(),
            'Z': met[f'second_{measure}'][primary][labels[primary] == 0].mean(axis=0).tolist(),
        }
        report['measures'][measure] = block

    control = eligible_mask(indep) & primary
    for measure in MEASURES:
        values = estimators(indep, measure)
        report['independent_control'][measure] = {
            'joint': auc_block(labels, weights, modes, values['joint'], control),
            'marginal': auc_block(labels, weights, modes, values['marginal'], control),
            'likelihood_ratio': auc_block(labels, weights, modes,
                                          values['likelihood_ratio'], control),
            'null_covariance_mean': [
                float(values['covariance_diagonal'][control][:, i].mean()) for i in range(3)],
        }
    report['independent_control']['cohort'] = int(control.sum())

    high = args.moments / 'met_hi' / 'met_hi_moments.npz'
    if high.exists():
        # Four times the retained draws per event.  If the joint-minus-marginal
        # difference is an artefact of finite sampling it must move here.
        dense = load(high)
        both = primary & eligible_mask(dense)
        report['convergence'] = {'cohort': int(both.sum()), 'retained_per_event': 512}
        for measure in MEASURES:
            values = estimators(dense, measure)
            report['convergence'][measure] = {
                'joint_auc': weighted_auc(labels[both], values['joint'][both],
                                          np.ones(int(both.sum()))),
                'marginal_auc': weighted_auc(labels[both], values['marginal'][both],
                                             np.ones(int(both.sum()))),
                'likelihood_auc': weighted_auc(labels[both], values['likelihood_ratio'][both],
                                               np.ones(int(both.sum()))),
                'joint_minus_marginal': bootstrap_auc_difference(
                    labels, values['joint'], values['marginal'], both, args.bootstrap),
                'effective_sample_size_mean': float(np.mean(dense[f'ess_{measure}'][both])),
                'posterior_covariance_std': [
                    float(values['covariance_diagonal'][both][:, i].std()) for i in range(3)],
                'monte_carlo_noise_mean': [
                    float(monte_carlo_covariance_noise(dense, measure)[both][:, i].mean())
                    for i in range(3)],
            }

    common = primary.copy()
    for data in sigmas.values():
        common &= eligible_mask(data)
    if not sigmas:
        report['met_resolution']['note'] = 'no smeared-MET configurations were present'
    report['met_resolution']['common_cohort'] = int(common.sum())
    curve = []
    for sigma in [0.0] + sorted(sigmas):
        data = met if sigma == 0.0 else sigmas[sigma]
        values = estimators(data, 'flat')
        own = eligible_mask(data)
        curve.append({
            'sigma_GeV': sigma,
            'eligible_fraction': float(own.mean()),
            'joint_auc_common': weighted_auc(labels[common], values['joint'][common],
                                             np.ones(int(common.sum()))),
            'marginal_auc_common': weighted_auc(labels[common], values['marginal'][common],
                                                np.ones(int(common.sum()))),
            'likelihood_auc_common': weighted_auc(labels[common],
                                                  values['likelihood_ratio'][common],
                                                  np.ones(int(common.sum()))),
            'joint_correlation_common': float(np.corrcoef(
                values['joint'][common], truth_canonical[common])[0, 1]),
            'region_area_median': float(np.median(data['region_area'][common])),
        })
    report['met_resolution']['curve'] = curve

    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
