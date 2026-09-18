"""Figures for the MET-constrained null-space run.

Reads the per-event moments and the report, and rebuilds the illustrative
region geometry directly from the cached truth surface, which needs kinematics
only.  PNG for reading and SVG for the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms
import nullspace as ns
from run_m2_report import (MODE_NAMES, REFERENCE_AUC, RankedScore, estimators,
                           eligible_mask, load, monte_carlo_covariance_noise,
                           spread_decomposition, weighted_auc)

plt.rcParams.update({
    'figure.dpi': 140, 'savefig.dpi': 220, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 8,
    'axes.titlesize': 9.5, 'figure.constrained_layout.use': True,
})

COLOUR = {'joint': '#1b6ca8', 'marginal': '#c1440e', 'likelihood': '#4a7c1f',
          'independent': '#6c6c6c', 'truth': '#111111'}
STYLE = {'joint': ('-', 'o'), 'marginal': ('--', 's'), 'likelihood': (':', '^'),
         'independent': ('-.', 'D'), 'truth': ('-', '*')}


def save(fig, output: Path, name: str) -> None:
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'{name}.{suffix}')
    plt.close(fig)
    print(f'wrote {name}')


def region_outline(visible, met, n_angles=721):
    """Boundary of one side's allowed ellipse in the side-0 transverse plane."""
    ellipse = ms.ellipse_frame(visible)
    angle = np.linspace(0.0, 2.0 * np.pi, n_angles)
    points = (ellipse['centre'][None]
              + np.cos(angle)[:, None] * ellipse['axis_major'][None]
              + np.sin(angle)[:, None] * ellipse['axis_minor'][None])
    return points if met is None else met[None] - points


def figure_geometry(met_data, surface, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    geometry = met_data['ellipse_geometry']
    ax = axes[0, 0]
    ax.hexbin(np.log10(geometry[..., 0].ravel()), np.log10(geometry[..., 1].ravel()),
              gridsize=70, bins='log', cmap='Blues', mincnt=1,
              extent=(0.0, 4.5, -0.6, 1.2))
    for ratio in (2, 3, 4):
        line = np.array([0.0, 4.5])
        ax.plot(line, line - ratio, color='#c1440e', lw=0.9, ls='--')
        ax.annotate(f'aspect $10^{ratio}$', xy=(4.4, 4.4 - ratio), fontsize=7,
                    color='#c1440e', ha='right', va='bottom')
    ax.set_xlabel(r'$\log_{10}$ semi-major axis  $c\sqrt{a}/m_{\rm vis}^2$  [GeV]')
    ax.set_ylabel(r'$\log_{10}$ semi-minor axis  $c/m_{\rm vis}$  [GeV]')
    ax.set_title('(a) each side allows a needle, not a disc\n'
                 f'median {np.median(geometry[..., 0]):.0f} GeV long, '
                 f'{np.median(geometry[..., 1]):.2f} GeV wide')

    ax = axes[0, 1]
    modes = met_data['modes']
    bins = np.logspace(-2, 3.5, 60)
    ax.hist(met_data['region_area'], bins=bins, histtype='step', color=COLOUR['truth'],
            lw=1.6, label=f"all  ({len(modes):,})")
    for code, name in MODE_NAMES.items():
        pair = (modes == code).all(axis=1)
        ax.hist(met_data['region_area'][pair], bins=bins, histtype='step', lw=1.2,
                linestyle=['--', ':', '-.'][list(MODE_NAMES).index(code)],
                label=f'{name}x{name}  ({int(pair.sum()):,})')
    ax.set_xscale('log')
    ax.set_xlabel(r'area of the MET-constrained solution set in $\nu^-_T$  [GeV$^2$]')
    ax.set_ylabel('events / bin')
    ax.set_title('(b) what MET leaves undetermined\n'
                 f"median {np.median(met_data['region_area']):.1f} GeV$^2$")
    ax.legend(loc='upper right')

    # One representative event: the two needles and where they cross.
    area = met_data['region_area']
    choice = int(np.argsort(np.abs(area - np.median(area)))[0])
    visible = surface['visible'][choice]
    nu_truth = surface['nu4'][choice][..., :3]
    met_event = nu_truth[0, :2] + nu_truth[1, :2]
    for ax, zoom in ((axes[1, 0], False), (axes[1, 1], True)):
        first = region_outline(visible[0], None)
        second = region_outline(visible[1], met_event)
        ax.plot(first[:, 0], first[:, 1], color=COLOUR['joint'], lw=1.2,
                label=r'$\tau^-$ mass shell')
        ax.plot(second[:, 0], second[:, 1], color=COLOUR['marginal'], lw=1.2, ls='--',
                label=r'$\tau^+$ mass shell + MET')
        rng = np.random.default_rng(1)
        unit = ms.lattice(70, 1, rng).transpose(1, 0, 2)
        drawn = ms.sample_region(visible[None], met_event[None], unit)
        inside = drawn['nu_t'][drawn['accepted'][:, 0], 0]
        ax.scatter(inside[:, 0], inside[:, 1], s=2.0, color='#7fb069',
                   label=f'solution set ({len(inside)} draws)', zorder=3)
        ax.scatter(*nu_truth[0, :2], marker='*', s=140, color=COLOUR['truth'],
                   zorder=4, label='truth')
        ax.set_xlabel(r'$\nu^-_x$  [GeV]')
        ax.set_ylabel(r'$\nu^-_y$  [GeV]')
        if zoom:
            half = max(4.0, 3.0 * np.sqrt(max(area[choice], 1e-3)))
            centre = inside.mean(axis=0) if len(inside) else nu_truth[0, :2]
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_title(f'(d) the same event, zoomed\n'
                         f'area {area[choice]:.1f} GeV$^2$')
        else:
            ax.set_title(f'(c) one median-area event (row {choice})\n'
                         'the two needles and their intersection')
            ax.legend(loc='best')
    save(fig, output, 'figure_1_nullspace_geometry')


def auc_with_error(labels, score, mask, draws=400, seed=20260919):
    index = np.flatnonzero(mask)
    ranked = RankedScore(labels[index], score[index])
    point = ranked.auc(np.ones(len(index)))
    rng = np.random.default_rng(seed)
    samples = np.array([ranked.auc(np.bincount(rng.integers(0, len(index), len(index)),
                                               minlength=len(index)).astype(float))
                        for _ in range(draws)])
    return point, np.quantile(samples, 0.025), np.quantile(samples, 0.975)


def figure_ladder(met_data, indep_data, output: Path) -> None:
    labels, modes = met_data['labels'], met_data['modes']
    primary = eligible_mask(met_data)
    control = primary & eligible_mask(indep_data)
    values = estimators(met_data, 'flat')
    control_values = estimators(indep_data, 'flat')
    truth = ms.transverse_statistic(met_data['truth_h_canonical'])

    truth_outer = np.einsum('ri,rj->rij', met_data['truth_h_canonical'][:, 0],
                            met_data['truth_h_canonical'][:, 1])
    truth_likelihood = (np.log(np.maximum(1.0 + np.einsum('ij,rij->r', ms.C_HIGGS, truth_outer),
                                          1e-12))
                        - np.log(np.maximum(1.0 + np.einsum('ij,rij->r', ms.C_Z, truth_outer),
                                            1e-12)))
    entries = [
        ('exact truth $h$, likelihood ratio', truth_likelihood, primary,
         COLOUR['truth'], ('-', 'P')),
        ('exact truth $h$, $T$', truth, primary, COLOUR['truth'], STYLE['truth']),
        (r'null-space likelihood ratio', values['likelihood_ratio'], primary,
         COLOUR['likelihood'], STYLE['likelihood']),
        (r'joint  $E[T]$', values['joint'], primary, COLOUR['joint'], STYLE['joint']),
        (r'marginal  $T(E[h^-],E[h^+])$', values['marginal'], primary,
         COLOUR['marginal'], STYLE['marginal']),
        ('no MET, marginal', control_values['marginal'], control,
         COLOUR['independent'], STYLE['independent']),
    ]
    groups = [('all', np.ones(len(labels), bool))]
    groups += [(f'{name}x{name}', (modes == code).all(axis=1))
               for code, name in MODE_NAMES.items()]

    fig, axes = plt.subplots(1, len(groups), figsize=(12.4, 4.2), sharey=True)
    for ax, (group_name, group_mask) in zip(axes, groups):
        for position, (name, score, mask, colour, (line, marker)) in enumerate(entries):
            use = mask & group_mask
            if use.sum() < 200:
                continue
            point, low, high = auc_with_error(labels, score, use)
            ax.errorbar([point], [position], xerr=[[point - low], [high - point]],
                        fmt=marker, color=colour, capsize=3, ms=6, lw=1.4)
        ax.axvline(0.5, color='#999999', lw=0.8, ls=':')
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([name for name, *_ in entries])
        ax.invert_yaxis()
        ax.set_xlabel('H/Z AUC (unweighted)')
        ax.set_title(f'{group_name}  ({int((primary & group_mask).sum()):,})')
    for ax in axes:
        ax.axvline(REFERENCE_AUC['azimuth_only_marginalised'], color='#b08968', lw=1.0, ls='--')
    axes[0].legend(handles=[Line2D([], [], color='#b08968', ls='--',
                                   label='azimuth-only marginalised, 0.621\n(previous run, truth '
                                         r'$E_\tau$ known)')],
                   loc='upper right', fontsize=7)
    fig.suptitle('Transverse spin statistic $T=h^-_n h^+_n + h^-_r h^+_r - h^-_k h^+_k$ '
                 'on the MET-constrained null space\n'
                 'truth visible, exact MET, flat measure; bars are 95% event bootstrap',
                 fontsize=10)
    save(fig, output, 'figure_2_estimator_ladder')


def figure_covariance(met_data, indep_data, output: Path) -> None:
    primary = eligible_mask(met_data)
    noise = monte_carlo_covariance_noise(met_data, 'flat')[primary]
    noise = noise if noise.any() else None
    labels = met_data['labels'][primary]
    values = estimators(met_data, 'flat')
    control = estimators(indep_data, 'flat')
    covariance = values['covariance_diagonal'][primary]
    null = control['covariance_diagonal'][primary & eligible_mask(indep_data)]

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.6))
    bins = np.linspace(-0.35, 0.35, 281)
    for i, component in enumerate(('n', 'r', 'k')):
        ax = axes[i]
        ax.hist(covariance[labels == 1, i], bins=bins, histtype='step', lw=1.5,
                color='#c1440e', label='H')
        ax.hist(covariance[labels == 0, i], bins=bins, histtype='step', lw=1.5,
                color='#1b6ca8', ls='--', label='Z')
        ax.hist(null[:, i], bins=bins, histtype='step', lw=1.1, color='#6c6c6c', ls=':',
                label='no-MET control')
        if noise is not None:
            width = noise[:, i].mean()
            height = max(np.histogram(covariance[:, i], bins=bins)[0].max(), 1)
            centres = 0.5 * (bins[1:] + bins[:-1])
            curve = height * np.exp(-0.5 * (centres / width) ** 2)
            visible = curve >= 0.5
            ax.plot(centres[visible], curve[visible], color='#7a4988', lw=1.4, ls='-.',
                    label=f'sampling error only\n($\\sigma$={width:.3f})' if i == 0 else None)
        ax.axvline(0.0, color='#999999', lw=0.8)
        ax.set_xlabel(rf'$\mathrm{{Cov}}(h^-_{component}, h^+_{component}\,|\,x)$')
        ax.set_ylabel('events / bin' if i == 0 else '')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        ax.set_xlim(-0.12, 0.12)
        ax.set_title(f'{component}:  mean H {covariance[labels == 1, i].mean():+.4f},  '
                     f'Z {covariance[labels == 0, i].mean():+.4f}', fontsize=8.5)
        if i == 0:
            ax.legend(loc='upper left')
    ax = axes[3]
    difference = values['joint'][primary] - values['marginal'][primary]
    ax.hist(difference[labels == 1], bins=np.linspace(-0.6, 0.6, 121), histtype='step',
            lw=1.5, color='#c1440e', label='H')
    ax.hist(difference[labels == 0], bins=np.linspace(-0.6, 0.6, 121), histtype='step',
            lw=1.5, color='#1b6ca8', ls='--', label='Z')
    ax.axvline(0.0, color='#999999', lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel(r'$E[T] - T(E[h^-],E[h^+])$')
    ax.set_title(f'what the marginal estimator drops\nmean H {difference[labels == 1].mean():+.4f}, '
                 f'Z {difference[labels == 0].mean():+.4f}', fontsize=8.5)
    ax.legend(loc='upper left')
    fig.suptitle('Posterior covariance between the two sides on the MET-constrained null space '
                 '(truth visible, exact MET, flat measure)', fontsize=10)
    save(fig, output, 'figure_3_posterior_covariance')


def figure_response(met_data, output: Path) -> None:
    primary = eligible_mask(met_data)
    values = estimators(met_data, 'flat')
    truth = ms.transverse_statistic(met_data['truth_h_canonical'])[primary]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, key, title in ((axes[0], 'joint', r'joint  $E[T]$'),
                           (axes[1], 'marginal', r'marginal  $T(E[h^-],E[h^+])$')):
        estimate = values[key][primary]
        ax.hexbin(truth, estimate, gridsize=70, bins='log', cmap='Blues', mincnt=1,
                  extent=(-1.2, 1.2, np.quantile(estimate, 0.001),
                          np.quantile(estimate, 0.999)))
        ax.set_xlabel(r'truth $T$ (canonical exact $h$)')
        ax.set_ylabel('estimate')
        ax.set_title(f'{title}\nPearson {np.corrcoef(estimate, truth)[0, 1]:.4f}')
    ax = axes[2]
    bins = np.linspace(-1.2, 1.2, 25)
    centres = 0.5 * (bins[1:] + bins[:-1])
    index = np.digitize(truth, bins) - 1
    for key, label in (('joint', r'joint $E[T]$'), ('marginal', 'marginal')):
        estimate = values[key][primary]
        estimate = (estimate - estimate.mean()) / estimate.std()
        profile = np.array([estimate[index == b].mean() if (index == b).sum() > 30 else np.nan
                            for b in range(len(centres))])
        error = np.array([estimate[index == b].std() / max(np.sqrt((index == b).sum()), 1)
                          if (index == b).sum() > 30 else np.nan for b in range(len(centres))])
        line, marker = STYLE[key]
        ax.errorbar(centres, profile, yerr=error, ls=line, marker=marker, ms=6 if key == 'joint' else 4,
                    color=COLOUR[key], label=label,
                    lw=2.4 if key == 'joint' else 1.2, alpha=0.9)
    ax.set_xlabel(r'truth $T$')
    ax.set_ylabel('standardised estimate (mean per bin)')
    ax.set_title('response to the truth pair statistic')
    ax.legend(loc='upper left')
    fig.suptitle('Do the estimators track the truth transverse spin statistic? '
                 f'({int(primary.sum()):,} events, flat measure)', fontsize=10)
    save(fig, output, 'figure_4_response')


def figure_met_curve(report, output: Path) -> None:
    curve = report['met_resolution']['curve']
    sigma = np.array([entry['sigma_GeV'] for entry in curve])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    for key, label, style_key in (('joint_auc_common', r'joint $E[T]$', 'joint'),
                                  ('marginal_auc_common', 'marginal', 'marginal'),
                                  ('likelihood_auc_common', 'likelihood ratio', 'likelihood')):
        line, marker = STYLE[style_key]
        ax.plot(sigma, [entry[key] for entry in curve], ls=line, marker=marker,
                color=COLOUR[style_key], label=label, lw=1.4, ms=5)
    ax.axhline(REFERENCE_AUC['truth_h_transverse_statistic'], color=COLOUR['truth'], lw=1.0,
               ls='-', label=r'exact truth $h$ (0.706)')
    ax.axhline(REFERENCE_AUC['azimuth_only_marginalised'], color='#b08968', lw=1.0, ls='--',
               label='azimuth-only marginalised (0.621)')
    ax.set_xlabel(r'MET resolution $\sigma_{\rm MET}$ per component  [GeV]')
    ax.set_ylabel('H/Z AUC (unweighted, common cohort)')
    ax.set_title(f"common cohort {report['met_resolution']['common_cohort']:,} events")
    ax.legend(loc='best')
    ax = axes[1]
    ax.plot(sigma, [entry['eligible_fraction'] for entry in curve], '-o', color='#1b6ca8',
            label='events with a solution')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r'$\sigma_{\rm MET}$  [GeV]')
    ax.set_ylabel('fraction of events with a non-empty solution set')
    twin = ax.twinx()
    twin.plot(sigma, [entry['region_area_median'] for entry in curve], '--s', color='#c1440e',
              label='median region area')
    twin.set_yscale('log')
    twin.set_ylabel(r'median solution-set area  [GeV$^2$]', color='#c1440e')
    twin.grid(False)
    ax.set_title('smeared MET both shrinks the cohort and inflates the region')
    handles = [Line2D([], [], color='#1b6ca8', marker='o', label='solution exists'),
               Line2D([], [], color='#c1440e', marker='s', ls='--', label='region area')]
    ax.legend(handles=handles, loc='lower left')
    fig.suptitle('How much MET resolution the recovered transverse information needs',
                 fontsize=10)
    save(fig, output, 'figure_5_met_requirement')


def figure_robustness(met_data, report, output: Path) -> None:
    primary = eligible_mask(met_data)
    labels = met_data['labels']
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    ax = axes[0]
    measures = ('flat', 'solid', 'solid_clipped')
    for position, measure in enumerate(measures):
        values = estimators(met_data, measure)
        for key, offset in (('joint', -0.16), ('marginal', 0.0),
                            ('likelihood_ratio', 0.16)):
            point, low, high = auc_with_error(labels, values[key], primary)
            style_key = 'likelihood' if key == 'likelihood_ratio' else key
            line, marker = STYLE[style_key]
            ax.errorbar([position + offset], [point],
                        yerr=[[point - low], [high - point]], fmt=marker,
                        color=COLOUR[style_key], capsize=3, ms=6,
                        label=key.replace('_', ' ') if position == 0 else None)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(['flat in $\\nu^-_T$', 'solid angle', 'solid angle,\n99% clip'])
    ax.set_ylabel('H/Z AUC (unweighted)')
    ax.set_title('measure dependence of the two estimators')
    ax.legend(loc='best')

    ax = axes[1]
    truth_canonical = ms.transverse_statistic(met_data['truth_h_canonical'])[primary]
    truth_reference = ms.transverse_statistic(met_data['truth_h_reference'])[primary]
    ax.hexbin(truth_canonical, truth_reference, gridsize=70, bins='log', cmap='Greys', mincnt=1)
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color='#c1440e', lw=1.0, ls='--')
    ax.set_xlabel(r'truth $T$ in the truth $n,r,k$ basis')
    ax.set_ylabel(r'truth $T$ in the observable reference basis')
    canonical = report['basis_check']['truth_T_canonical_basis']['auc_unweighted']
    reference = report['basis_check']['truth_T_reference_basis']['auc_unweighted']
    ax.set_title(f'frame choice costs {canonical - reference:+.4f} in AUC\n'
                 f'{canonical:.4f} (truth frame) vs {reference:.4f} (reference frame)')
    fig.suptitle('Robustness: does the answer depend on the measure or on the frame?',
                 fontsize=10)
    save(fig, output, 'figure_6_robustness')


def figure_solution_structure(met_data, output: Path) -> None:
    """Where the residual freedom sits: between the four sheets or inside one."""
    primary = eligible_mask(met_data)
    spread = spread_decomposition(met_data, 'flat')
    if not spread:
        print('skipping figure_7: the moments predate the sheet decomposition')
        return
    values = estimators(met_data, 'flat')
    truth = ms.transverse_statistic(met_data['truth_h_canonical'])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

    ax = axes[0]
    total = np.sqrt(spread['total_variance'][primary])
    bins = np.linspace(0.0, 1.2, 80)
    ax.hist(total, bins=bins, histtype='step', lw=1.6, color=COLOUR['truth'],
            label=fr'total, median {np.median(total):.3f}')
    ax.hist(np.sqrt(spread['between_sheet'][primary]), bins=bins, histtype='step', lw=1.3,
            color=COLOUR['joint'], ls='--',
            label=fr"between sheets, median {np.median(np.sqrt(spread['between_sheet'][primary])):.3f}")
    ax.hist(np.sqrt(spread['within_sheet'][primary]), bins=bins, histtype='step', lw=1.3,
            color=COLOUR['marginal'], ls=':',
            label=fr"within a sheet, median {np.median(np.sqrt(spread['within_sheet'][primary])):.3f}")
    ax.set_xlabel(r'per-event spread of $T$ on the solution set')
    ax.set_ylabel('events / bin')
    ax.set_title('(a) how much the pair statistic still moves\n'
                 r'$|T|\leq 1$, so this spread is the whole scale')
    ax.legend(loc='upper right', fontsize=7.5)

    ax = axes[1]
    fraction = spread['between_sheet'][primary] / np.maximum(spread['total_variance'][primary], 1e-12)
    ax.hist(fraction, bins=np.linspace(0.0, 1.0, 60), histtype='step', lw=1.6,
            color=COLOUR['joint'])
    ax.axvline(float(np.median(fraction)), color=COLOUR['marginal'], ls='--', lw=1.2)
    ax.set_xlabel('between-sheet share of the variance of $T$')
    ax.set_ylabel('events / bin')
    ax.set_title('(b) discrete or continuous ambiguity?\n'
                 f'median share {np.median(fraction):.3f}')

    ax = axes[2]
    quality = np.abs(values['joint'][primary] - truth[primary])
    area = met_data['region_area'][primary]
    ax.hexbin(np.log10(np.maximum(area, 1e-3)), quality, gridsize=60, bins='log',
              cmap='Blues', mincnt=1)
    edges = np.quantile(np.log10(np.maximum(area, 1e-3)), np.linspace(0.0, 1.0, 13))
    index = np.clip(np.digitize(np.log10(np.maximum(area, 1e-3)), edges) - 1, 0, 11)
    centres = 0.5 * (edges[1:] + edges[:-1])
    profile = np.array([np.median(quality[index == b]) for b in range(12)])
    ax.plot(centres, profile, '-o', color=COLOUR['marginal'], lw=1.6, ms=4,
            label='median per decile')
    ax.set_xlabel(r'$\log_{10}$ (solution-set area / GeV$^2$)')
    ax.set_ylabel(r'$|E[T] - T_{\rm truth}|$')
    ax.set_title('(c) does a smaller solution set help?')
    ax.legend(loc='upper left')
    fig.suptitle('Structure of the residual ambiguity on the MET-constrained null space',
                 fontsize=10)
    save(fig, output, 'figure_7_solution_structure')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    met_data = load(args.moments / 'met' / 'met_moments.npz')
    indep_data = load(args.moments / 'indep' / 'indep_moments.npz')
    report = json.loads(args.report.read_text())
    surface = ns.load_surface(args.surface)

    figure_geometry(met_data, surface, args.output)
    figure_ladder(met_data, indep_data, args.output)
    figure_covariance(met_data, indep_data, args.output)
    figure_response(met_data, args.output)
    if len(report['met_resolution'].get('curve', [])) > 1:
        figure_met_curve(report, args.output)
    figure_robustness(met_data, report, args.output)
    figure_solution_structure(met_data, args.output)


if __name__ == '__main__':
    main()
