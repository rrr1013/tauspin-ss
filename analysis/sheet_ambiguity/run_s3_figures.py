"""Figures for the discrete mass-shell ambiguity.

Everything is drawn from `report.json` and `scores.npz`; no estimator is
recomputed here.  Line style and marker carry the same information as colour so
the panels survive a greyscale print.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODE_NAMES = {0: r'$\pi$ (1p0n)', 1: r'$\rho$ (1p1n)', 3: r'$3\pi$ (3p0n)'}
PAIR_KEYS = (('auc_1p0nx1p0n', r'$\pi\times\pi$'),
             ('auc_1p1nx1p1n', r'$\rho\times\rho$'),
             ('auc_3p0nx3p0n', r'$3\pi\times3\pi$'))
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.axisbelow': True})


def load(output: Path) -> tuple[dict, dict[str, np.ndarray]]:
    report = json.loads((output / 'report.json').read_text())
    with np.load(output / 'scores.npz') as handle:
        scores = {key: handle[key] for key in handle.files}
    return report, scores


def figure_ladder(report: dict, output: Path) -> None:
    """AUC of every conditioning, overall and per decay-mode pair."""
    ladder = report['oracle_ladder']
    rows = [
        ('flat_o0_likelihood_ratio', 'O0  nothing known\n(kinematic ceiling)'),
        ('flat_o2a_likelihood_ratio', r'O2a  correct root, side $\tau^-$'),
        ('flat_o2b_likelihood_ratio', r'O2b  correct root, side $\tau^+$'),
        ('flat_o1_likelihood_ratio', 'O1  correct root, both sides\n(discrete resolved)'),
        ('o3_likelihood_ratio', r'O3  truth $\nu_T$ known'
                                '\n(continuous resolved)'),
        ('o4_likelihood_ratio', 'O4  truth neutrinos\n(nothing left free)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={'width_ratios': [1.15, 1.0]})
    y = np.arange(len(rows))[::-1]

    ax = axes[0]
    values = [ladder[key]['auc_unweighted'] for key, _ in rows]
    ax.barh(y, values, height=0.6, color='0.82', edgecolor='0.25')
    for position, value in zip(y, values):
        ax.text(value + 0.002, position, f'{value:.4f}', va='center', fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in rows], fontsize=8)
    ax.set_xlim(0.55, max(values) + 0.03)
    ax.set_xlabel('H/Z AUC (unweighted, likelihood ratio on the conditioned set)')
    ax.set_title(f"all events, N = {report['eligible_events']:,}", fontsize=9)
    # The trained full-reco readout is deliberately not drawn here: it is a
    # different cohort, a different weighting and an estimator that may use
    # non-spin handles, so putting it on this axis would invite a comparison
    # this figure cannot support.

    ax = axes[1]
    markers = ('o', 's', '^')
    lines = ('-', '--', ':')
    for (pair_key, pair_label), marker, style in zip(PAIR_KEYS, markers, lines):
        pair_values = [ladder[key].get(pair_key, np.nan) for key, _ in rows]
        ax.plot(pair_values, y, marker=marker, ls=style, ms=5, label=pair_label)
    overall = [ladder[key]['auc_unweighted'] for key, _ in rows]
    ax.plot(overall, y, marker='D', ls='-', lw=2.0, color='0.2', ms=4, label='all events')
    ax.set_yticks(y)
    ax.set_yticklabels(['O0', 'O2a', 'O2b', 'O1', 'O3', 'O4'])
    ax.set_xlabel('H/Z AUC (unweighted)')
    ax.set_title('by truth decay-mode pair', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Resolving the discrete mass-shell root, and what it is worth\n'
                 'truth visible + exact MET, flat measure, theory spin-correlation matrices',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output / 'figure_1_oracle_ladder.png', bbox_inches='tight')
    plt.close(fig)


def figure_resolution(report: dict, output: Path) -> None:
    """What a tau-direction measurement of resolution sigma buys."""
    curves = report['resolution_curves']
    fine = report['sheet_only_resolution_curve']
    ladder = report['oracle_ladder']
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    def series(entries, key, path=('likelihood_ratio', 'auc_unweighted')):
        out = []
        for entry in entries:
            node = entry
            for step in path:
                node = node[step]
            out.append((entry[key], node))
        return np.array(out).T

    ax = axes[0]
    x, y = series(fine, 'sigma_mrad')
    ax.plot(x, y, 'o-', ms=4, color='0.15',
            label='used only to pick the root\n(exact at any resolution)')
    grid = curves['two_sided']
    gx, gy = series(grid, 'sigma_mrad')
    trusted = np.array([entry['ess_median'] >= 20.0 for entry in grid])
    ax.plot(gx[trusted], gy[trusted], 's--', ms=5, color='C0',
            label=r'used fully (grid, ESS$_{\rm med}\geq$20)')
    if (~trusted).any():
        ax.plot(gx[~trusted], gy[~trusted], 'x:', ms=6, color='C0', alpha=0.5,
                label='same, grid limited')
    for key, label, style in (('flat_o0_likelihood_ratio', 'O0 ceiling', ':'),
                              ('flat_o1_likelihood_ratio', 'O1 root oracle', '--'),
                              ('o4_likelihood_ratio', 'O4 truth', '-.')):
        value = ladder[key]['auc_unweighted']
        ax.axhline(value, color='0.45', ls=style, lw=1.0)
        ax.text(x.min(), value + 0.0008, f' {label}', fontsize=7.5, color='0.35',
                va='bottom')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma_\alpha$ per transverse component [mrad], both sides')
    ax.set_ylabel('H/Z AUC (unweighted)')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title('the discrete choice is only part of the value', fontsize=9)

    ax = axes[1]
    for name, label, marker, style, colour in (
            ('two_sided', 'both taus measured', 's', '-', 'C0'),
            ('one_sided', r'only the $\tau^-$ side measured', '^', '--', 'C1')):
        entries = curves[name]
        ex, ey = series(entries, 'sigma_mrad')
        keep = np.array([entry['ess_median'] >= 20.0 for entry in entries])
        ax.plot(ex[keep], ey[keep], marker=marker, ls=style, ms=5, color=colour, label=label)
        ax.plot(ex[~keep], ey[~keep], marker='x', ls=':', ms=5, color=colour, alpha=0.5)
    for key, label, style in (('flat_o0_likelihood_ratio', 'O0', ':'),
                              ('flat_o1_likelihood_ratio', 'O1', '--'),
                              ('o4_likelihood_ratio', 'O4', '-.')):
        value = ladder[key]['auc_unweighted']
        ax.axhline(value, color='0.45', ls=style, lw=1.0)
        ax.text(ex.max(), value + 0.0008, f'{label} ', fontsize=7.5, color='0.35',
                va='bottom', ha='right')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma_\alpha$ [mrad]')
    ax.set_ylabel('H/Z AUC (unweighted)')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('one measured side against two', fontsize=9)

    ax = axes[2]
    px, py = series(fine, 'sigma_mrad', ('p_truth_sheet',))
    ax.plot(px, py, 'o-', ms=4, color='0.15', label='P(truth sheet), sheet posterior')
    gx, gp = series(grid, 'sigma_mrad', ('p_truth_sheet',))
    ax.plot(gx, gp, 's--', ms=4, color='C0', label='P(truth sheet), full grid')
    ax.axhline(0.25, color='0.5', ls=':', lw=1.0)
    ax.text(px.min(), 0.265, ' no information (1/4)', fontsize=7.5, color='0.35')
    ax.set_xscale('log')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r'$\sigma_\alpha$ [mrad]')
    ax.set_ylabel('posterior probability of the truth sheet')
    ax.legend(fontsize=8, loc='lower left')
    twin = ax.twinx()
    ex, ee = series(grid, 'sigma_mrad', ('ess_median',))
    twin.plot(ex, ee, '^-.', ms=4, color='C3', alpha=0.8)
    twin.set_yscale('log')
    twin.set_ylabel('median effective hypotheses (grid)', color='C3', fontsize=8)
    twin.tick_params(axis='y', colors='C3')
    twin.grid(False)
    ax.set_title('can the root be identified, and can the grid follow?', fontsize=9)

    fig.suptitle('How well the tau flight direction must be measured\n'
                 'idealised unbiased Gaussian measurement, same noise realisation at '
                 'every $\\sigma_\\alpha$; truth visible + exact MET', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output / 'figure_2_resolution_requirement.png', bbox_inches='tight')
    plt.close(fig)


def figure_vertex(report: dict, scores: dict[str, np.ndarray], output: Path) -> None:
    """The same requirement expressed as a decay-vertex position resolution."""
    entries = report['resolution_curves']['vertex']
    ladder = report['oracle_ladder']
    mask = scores['mask']
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    ax = axes[0]
    length = scores['decay_length_um'][mask][scores['has_vertex'][mask]] * 1e-3
    ax.hist(np.clip(length, 1e-3, 1e3), bins=np.logspace(-3, 3, 80), histtype='step',
            lw=1.6, color='0.15')
    ax.set_xscale('log')
    ax.set_xlabel('tau decay length [mm], three-prong sides')
    ax.set_ylabel('sides per bin')
    ax.set_title(f'median {np.median(length):.2f} mm at this boost', fontsize=9)

    ax = axes[1]
    x = [entry['vertex_um'] for entry in entries]
    ax.plot(x, [entry['likelihood_ratio']['auc_unweighted'] for entry in entries],
            's-', ms=5, color='C0', label='vertex used fully')
    ax.plot(x, [entry['sheet_only']['likelihood_ratio']['auc_unweighted'] for entry in entries],
            'o--', ms=5, color='0.15', label='vertex used only for the root')
    for key, label, style in (('flat_o0_likelihood_ratio', 'O0 ceiling', ':'),
                              ('flat_o1_likelihood_ratio', 'O1 root oracle', '--')):
        value = ladder[key]['auc_unweighted']
        ax.axhline(value, color='0.45', ls=style, lw=1.0)
        ax.text(x[-1], value + 0.0004, f'{label} ', fontsize=7.5, color='0.35',
                va='bottom', ha='right')
    ax.set_xlabel(r'vertex position resolution per coordinate [$\mu$m]')
    ax.set_ylabel('H/Z AUC (unweighted), all events')
    ax.legend(fontsize=8)
    ax.set_title('three-prong sides only get a vertex', fontsize=9)

    ax = axes[2]
    ax.plot(x, [entry['likelihood_ratio'].get('auc_3p0nx3p0n', np.nan) for entry in entries],
            's-', ms=5, color='C0', label=r'$3\pi\times3\pi$, vertex used fully')
    ax.plot(x, [entry['sheet_only']['likelihood_ratio'].get('auc_3p0nx3p0n', np.nan)
                for entry in entries],
            'o--', ms=5, color='0.15', label=r'$3\pi\times3\pi$, root only')
    for key, label, style in (('flat_o0_likelihood_ratio', 'O0', ':'),
                              ('flat_o1_likelihood_ratio', 'O1', '--'),
                              ('o4_likelihood_ratio', 'O4 truth', '-.')):
        value = ladder[key].get('auc_3p0nx3p0n', np.nan)
        ax.axhline(value, color='0.45', ls=style, lw=1.0)
        ax.text(x[-1], value + 0.0012, f'{label} ', fontsize=7.5, color='0.35',
                va='bottom', ha='right')
    ax.set_xlabel(r'vertex position resolution per coordinate [$\mu$m]')
    ax.set_ylabel('H/Z AUC (unweighted)')
    ax.legend(fontsize=8, loc='upper right')
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(x)
    top.set_xticklabels([f"{entry['sigma_alpha_mrad_median_measured_sides']:.1f}"
                         for entry in entries], fontsize=7.5)
    top.set_xlabel(r'median $\sigma_\alpha$ on a measured side [mrad]', fontsize=8)
    top.grid(False)
    ax.set_title(r'the mode with the vertex: $3\pi\times3\pi$', fontsize=9, pad=26)

    fig.suptitle('The requirement as a vertex resolution\n'
                 'geometric scaling on truth kinematics: exact primary vertex, unbiased '
                 'isotropic vertex fit, decay length drawn from each side own '
                 'proper-time exponential; not a detector simulation', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output / 'figure_6_vertex_resolution.png', bbox_inches='tight')
    plt.close(fig)


def figure_geometry(report: dict, scores: dict[str, np.ndarray], output: Path) -> None:
    """The angular size of the discrete ambiguity, against its continuous partner."""
    mask = scores['mask']
    modes = scores['modes']
    separation = 1e3 * scores['root_angle']
    intra = 1e3 * np.take_along_axis(
        scores['sheet_angle_rms'],
        scores['truth_sheet'][:, None, None].repeat(2, axis=2), axis=1)[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    bins = np.logspace(-2, 2.3, 70)
    for code, (marker, style) in zip(MODE_NAMES, (('o', '-'), ('s', '--'), ('^', ':'))):
        side = mask[:, None] & (modes == code)
        if side.sum() < 200:
            continue
        counts, edges = np.histogram(np.clip(separation[side], bins[0], bins[-1]), bins=bins)
        centre = np.sqrt(edges[:-1] * edges[1:])
        ax.plot(centre, counts / counts.sum(), ls=style, marker=marker, ms=2.5, lw=1.3,
                markevery=6, label=f'{MODE_NAMES[code]}, {side.sum():,} sides')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta\alpha$ between the two mass-shell roots [mrad]')
    ax.set_ylabel('fraction of tau sides per bin')
    ax.legend(fontsize=8)
    ax.set_title('how far apart the two roots point', fontsize=9)

    ax = axes[1]
    both = np.repeat(mask[:, None], 2, axis=1)
    finite = both & np.isfinite(separation) & np.isfinite(intra) & (intra > 0)
    ax.hexbin(np.log10(np.clip(separation[finite], 1e-2, 1e3)),
              np.log10(np.clip(intra[finite], 1e-3, 1e3)), gridsize=60, bins='log',
              cmap='Greys', mincnt=1)
    limits = np.array([-2.0, 2.3])
    ax.plot(limits, limits, ls='--', color='C3', lw=1.2, label='equal')
    ax.set_xlabel(r'$\log_{10}\,\Delta\alpha_{\rm between\ roots}$ [mrad]')
    ax.set_ylabel(r'$\log_{10}\,$rms spread of the tau direction'
                  '\n' r'within the truth sheet [mrad]')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('discrete separation vs continuous spread', fontsize=9)

    ax = axes[2]
    order = np.sort(separation[both & np.isfinite(separation)])
    fraction = 1.0 - np.arange(len(order)) / len(order)
    ax.plot(order, fraction, color='0.15', lw=1.4)
    labels = []
    for sigma, style in ((1.0, ':'), (3.0, '--'), (10.0, '-.')):
        ax.axvline(2.0 * sigma, color='0.45', ls=style, lw=1.1)
        above = float((order > 2.0 * sigma).mean())
        labels.append(f'$\\sigma_\\alpha$={sigma:g} mrad: {above:.0%} of sides '
                      f'have $\\Delta\\alpha>2\\sigma_\\alpha$')
    ax.text(0.03, 0.06, '\n'.join(labels), transform=ax.transAxes, fontsize=7.2,
            color='0.2', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.7', lw=0.6))
    ax.set_xlim(1e-1, 2e2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta\alpha$ between the two roots [mrad]')
    ax.set_ylabel(r'fraction of sides with larger $\Delta\alpha$')
    ax.set_title('what a given resolution can separate', fontsize=9)

    fig.suptitle('Geometry of the discrete ambiguity: the two mass-shell roots send the tau '
                 'in measurably different directions\n'
                 'evaluated at the truth transverse momentum; '
                 f"{report['closure']['near_degenerate_fraction_0p1GeV']:.1%} of events have "
                 'the two roots within 0.1 GeV in $n_z$ and no meaningful label',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output / 'figure_3_ambiguity_geometry.png', bbox_inches='tight')
    plt.close(fig)


def figure_event_level(report: dict, scores: dict[str, np.ndarray], output: Path) -> None:
    """Which events the root oracle actually helps."""
    mask = scores['mask']
    truth = scores['score_o4_joint'][mask]
    base = scores['score_flat_o0_joint'][mask]
    oracle = scores['score_flat_o1_joint'][mask]
    separation = 1e3 * scores['root_angle'][mask].min(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    bins = np.linspace(0, 2.0, 60)
    ax.hist(np.abs(base - truth), bins=bins, histtype='step', lw=1.6, color='0.15',
            label=f'O0, median {np.median(np.abs(base - truth)):.3f}')
    ax.hist(np.abs(oracle - truth), bins=bins, histtype='step', lw=1.6, ls='--',
            color='C0', label=f'O1, median {np.median(np.abs(oracle - truth)):.3f}')
    ax.set_xlabel(r'$|\,\mathbb{E}[T] - T_{\rm truth}|$')
    ax.set_ylabel('events per bin')
    ax.legend(fontsize=8)
    ax.set_title('per-event error of the pair statistic', fontsize=9)

    ax = axes[1]
    improvement = np.abs(base - truth) - np.abs(oracle - truth)
    edges = np.quantile(separation, np.linspace(0, 1, 11))
    centre, median, low, high = [], [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        inside = (separation >= start) & (separation < stop)
        if inside.sum() < 50:
            continue
        centre.append(np.median(separation[inside]))
        median.append(np.median(improvement[inside]))
        low.append(np.quantile(improvement[inside], 0.25))
        high.append(np.quantile(improvement[inside], 0.75))
    ax.fill_between(centre, low, high, color='0.85', label='interquartile range')
    ax.plot(centre, median, 'o-', ms=4, color='0.15', label='median')
    ax.axhline(0.0, color='C3', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel(r'smaller $\Delta\alpha$ of the two sides [mrad]')
    ax.set_ylabel(r'error reduction from the root oracle')
    ax.legend(fontsize=8)
    ax.set_title('the gain follows the geometry', fontsize=9)

    ax = axes[2]
    for score, label, style, colour in (
            (base, 'O0 nothing known', ':', '0.15'),
            (oracle, 'O1 root oracle', '--', 'C0'),
            (scores['score_o3_joint'][mask], r'O3 truth $\nu_T$', '-.', 'C1'),
            (scores['score_o4_joint'][mask], 'O4 truth neutrinos', '-', 'C3')):
        error = np.sort(np.abs(score - truth))
        ax.plot(error, np.arange(len(error)) / len(error), ls=style, lw=1.5,
                color=colour, label=label)
    ax.set_xlabel(r'$|\,\mathbb{E}[T] - T_{\rm truth}|$')
    ax.set_ylabel('cumulative fraction of events')
    ax.set_xlim(0, 1.6)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('the same ladder, event by event', fontsize=9)

    fig.suptitle('Where the discrete ambiguity costs something\n'
                 f"{report['eligible_events']:,} events, truth visible + exact MET, "
                 'flat measure', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output / 'figure_4_event_level_gain.png', bbox_inches='tight')
    plt.close(fig)


def figure_sheet_structure(report: dict, scores: dict[str, np.ndarray], output: Path) -> None:
    """How the pair statistic is distributed over the four sheets."""
    mask = scores['mask']
    sheet_mean = scores['sheet_mean_t'][mask]
    weight = scores['sheet_weight_o0'][mask]
    truth_sheet = scores['truth_sheet'][mask]
    truth = scores['score_o4_joint'][mask]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    joint = (sheet_mean * weight).sum(axis=1)
    total = np.maximum(scores['t_square_o0'][mask] - joint ** 2, 0.0)
    between = (weight * (sheet_mean - joint[:, None]) ** 2).sum(axis=1)
    good = total > 0
    fraction = between[good] / total[good]
    ax.hist(fraction, bins=np.linspace(0, 1, 60), histtype='step', lw=1.6, color='0.15')
    ax.axvline(np.median(fraction), color='C3', ls='--', lw=1.2,
               label=f'median {np.median(fraction):.3f}')
    ax.axvline(fraction.mean(), color='C0', ls='-.', lw=1.2,
               label=f'mean {fraction.mean():.3f}')
    ax.set_xlabel(r'share of ${\rm Var}[T]$ that sits between the four sheets')
    ax.set_ylabel('events per bin')
    ax.legend(fontsize=8)
    ax.set_title('the discrete part of the remaining spread', fontsize=9)

    ax = axes[1]
    on_truth = np.take_along_axis(sheet_mean, truth_sheet[:, None], axis=1)[:, 0]
    others = np.where(np.arange(4)[None] == truth_sheet[:, None], np.nan, sheet_mean)
    ax.hist(on_truth - truth, bins=np.linspace(-2, 2, 70), histtype='step', lw=1.6,
            color='0.15', label='truth sheet')
    ax.hist(np.nanmean(others, axis=1) - truth, bins=np.linspace(-2, 2, 70),
            histtype='step', lw=1.6, ls='--', color='C0', label='mean of the other three')
    ax.axvline(0.0, color='C3', ls=':', lw=1.0)
    ax.set_xlabel(r'$\mathbb{E}[T\,|\,\rm sheet] - T_{\rm truth}$')
    ax.set_ylabel('events per bin')
    ax.legend(fontsize=8)
    ax.set_title('the truth sheet is the one that tracks truth', fontsize=9)

    ax = axes[2]
    curve = report['sheet_only_resolution_curve']
    probability = np.array([entry['p_truth_sheet'] for entry in curve])
    auc = np.array([entry['likelihood_ratio']['auc_unweighted'] for entry in curve])
    ax.plot(probability, auc, 'o-', ms=4, color='0.15')
    for entry, p, a in zip(curve, probability, auc):
        if entry['sigma_mrad'] in (0.1, 1.0, 3.0, 10.0, 50.0):
            ax.annotate(f"{entry['sigma_mrad']:g} mrad", (p, a), fontsize=7.5,
                        textcoords='offset points', xytext=(4, -8))
    ax.set_xlabel('posterior probability of the truth sheet')
    ax.set_ylabel('H/Z AUC (unweighted)')
    ax.set_title('AUC is nearly linear in how often the root is right', fontsize=9)

    fig.suptitle('Structure of the four sheets and the price of getting the root wrong',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output / 'figure_5_sheet_structure.png', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, scores = load(args.output)
    figure_ladder(report, args.output)
    figure_resolution(report, args.output)
    figure_geometry(report, scores, args.output)
    figure_event_level(report, scores, args.output)
    figure_sheet_structure(report, scores, args.output)
    figure_vertex(report, scores, args.output)
    print('figures written to', args.output)


if __name__ == '__main__':
    main()
