"""Figures for the CP-mixing-angle sensitivity run (reads the p2-p6 outputs)."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import acoplanarity, score, sensitivity, triple, weights  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 160})
PAIRS = ['pi x pi', 'rho x rho', '3pi x 3pi', 'pi x rho', 'pi x 3pi', 'rho x 3pi']
PLABEL = {'pi x pi': 'π×π', 'rho x rho': 'ρ×ρ', '3pi x 3pi': '3π×3π',
          'pi x rho': 'π×ρ', 'pi x 3pi': 'π×3π', 'rho x 3pi': 'ρ×3π'}


def load_all(data):
    D = np.load(data / 'truth_surface_cleo.npz')
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    h = np.array(D['h_ref'], float)
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, hh = canonical_h(fr, D['modes'], D['pion_charges'], s, md)
            h[sel, s] = hh
    return D, h


def shape(ax, x, w, bins, color, ls, label, mk='o'):
    """Area-normalised histogram with sumw2 errors and the best-fit 1 + A cos(x - x0)."""
    n, e = np.histogram(x, bins=bins, range=(0, 2 * np.pi), weights=w)
    n2, _ = np.histogram(x, bins=bins, range=(0, 2 * np.pi), weights=w**2)
    tot, bw = n.sum(), np.diff(e)
    d, err = n / tot / bw, np.sqrt(n2) / tot / bw
    c = 0.5 * (e[1:] + e[:-1])
    ax.errorbar(c, d, yerr=err, fmt=mk, ms=3.6, color=color, elinewidth=1.0, capsize=0,
                label=label, zorder=3)
    M = np.stack([np.ones_like(c), np.cos(c), np.sin(c)], 1)
    W = np.diag(1 / np.maximum(err, 1e-12)**2)
    b = np.linalg.solve(M.T @ W @ M, M.T @ W @ d)
    g = np.linspace(0, 2 * np.pi, 400)
    ax.plot(g, b[0] + b[1] * np.cos(g) + b[2] * np.sin(g), ls, color=color, lw=1.8, zorder=2)
    return float(np.hypot(b[1], b[2]) / b[0]), float(np.arctan2(-b[2], b[1]))


def fig1(data, out):
    """The CP-odd angle itself: how the distribution moves with phi_tau."""
    A = np.load(HERE / 'results' / 'classical_arrays.npz')
    D, h_exact = load_all(data)
    C0 = c_matrix(0.0)
    rows = A['rho-rho__rows']
    PR = np.load(data / 'gen3pi_full22_s42.npz')
    hp = PR['h_pred'].astype(float)[rows]
    he = h_exact[rows]
    zrr = (D['modes'][:, 0] == 1) & (D['modes'][:, 1] == 1) & ~D['labels'].astype(bool)
    zref = {1: acoplanarity(PR['h_pred'].astype(float)[zrr]),
            2: acoplanarity(h_exact[zrr])}
    w45 = weights(he, np.deg2rad(45.), C0)
    one = np.ones(len(rows))
    panels = [('classical φ*$_{CP}$ (reco π$^0$ decay planes)', A['rho-rho__phistar_reco'],
               json_get('classical_full.json', ['channels', 'rho-rho', 'classical_reco_sin', 'sigma_deg'])),
              ('learned reco $h$ (+IP, SV): Δφ$_h$', acoplanarity(hp),
               json_get('classical_full.json', ['channels', 'rho-rho', 'learned_reco_h_Tp', 'sigma_deg'])),
              ('exact $h$: Δφ$_h$', acoplanarity(he),
               json_get('classical_full.json', ['channels', 'rho-rho', 'exact_h_Tp', 'sigma_deg']))]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.9), sharey=True)
    for ax, (title, x, sig) in zip(axes, panels):
        a0, p0 = shape(ax, x, one, 16, BLUE, '-', 'CP-even, $\\phi_\\tau=0$ (as generated)', 'o')
        a1, p1 = shape(ax, x, w45, 16, ORANGE, '--', 'CP-mixed, $\\phi_\\tau=45°$ (reweighted)', 's')
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlabel('angle [rad]')
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                      ['0', 'π/2', 'π', '3π/2', '2π'])
        shift = np.rad2deg(np.mod(p1 - p0 + np.pi, 2 * np.pi) - np.pi)
        note = (f'modulation A = {a0:.3f} → {a1:.3f},  peak moves {shift:+.0f}°\n'
                f'σ($\\phi_\\tau$) = {sig:.2f}° per 10,000 events')
        ax.text(0.03, 0.965, note, transform=ax.transAxes, fontsize=8.6, color=MUTED, va='top')
        ax.set_ylim(0.04, 0.40)
    axes[0].set_ylabel('normalised events / bin width')
    fig.suptitle('Higgs CP mixing angle in ρ×ρ events: the same 7,472 H events seen three ways',
                 fontsize=11.5, color=INK, y=0.99)
    for i, ax in enumerate(axes):
        ax.grid(axis='y', color=GRAY, alpha=0.18, lw=0.6)
        if i in zref:
            az, _ = shape(ax, zref[i], np.ones(len(zref[i])), 16, GRAY, ':',
                          'Z→ττ: no transverse correlation', 'v')
            ax.text(0.97, 0.04, f'instrumental modulation A = {az:.3f}', ha='right',
                    transform=ax.transAxes, fontsize=8.4, color=MUTED)
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', bbox_to_anchor=(0.5, 0.055), ncol=3, fontsize=9)
    fig.text(0.5, 0.008, 'H→ττ, m=91.19 GeV, reco-selected validation cohort, both sides truth ρ and reco ρ.\n'
             'φ$_\\tau$=45° obtained by the exact-$h$ spin weight (ESS 0.40); shapes area-normalised, sumw2 errors, '
             'curves are weighted fits of 1 + A cos(x − x$_0$).',
             ha='center', fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0.01, 0.16, 0.99, 0.95))
    fig.savefig(out / 'fig1_cp_angle_distributions.png')
    plt.close(fig)


def json_get(name, path):
    d = json.loads((HERE / 'results' / name).read_text())
    for k in path:
        d = d[k]
    return d


def fig2(out):
    """One sensitivity ladder: every information level, per channel and overall."""
    from matplotlib.ticker import NullFormatter, NullLocator
    C = json_get('classical_full.json', ['channels'])
    S = json_get('sensitivity.json', [])
    E = S['sensitivity']['exact']
    R = S['sensitivity']['reco']
    series = [('classical φ*$_{CP}$, reconstructed inputs', ORANGE, 'o'),
              ('classical φ*$_{CP}$, perfect inputs', YELLOW, 's'),
              ('learned $h$ from full reco, no geometry', VIOLET, 'v'),
              ('learned $h$ from full reco, + 3 IPs + SV', BLUE, 'D'),
              ('exact $h$, triple product', AQUA, '^'),
              ('exact $h$, optimal statistic', GRAY, '*')]
    rows = [
        ('π×π\n(n=1,346)', [C['pi-pi']['classical_reco_sin'], C['pi-pi']['classical_truth_sin'],
                            C['pi-pi']['learned_reco_h_nogeo_Tp'], C['pi-pi']['learned_reco_h_Tp'],
                            C['pi-pi']['exact_h_Tp'], C['pi-pi']['exact_h_optimal_sigma_deg']]),
        ('ρ×ρ\n(n=7,472)', [C['rho-rho']['classical_reco_sin'], C['rho-rho']['classical_truth_sin'],
                            C['rho-rho']['learned_reco_h_nogeo_Tp'], C['rho-rho']['learned_reco_h_Tp'],
                            C['rho-rho']['exact_h_Tp'], C['rho-rho']['exact_h_optimal_sigma_deg']]),
        ('π×ρ\n(n=6,602)', [C['pi-rho']['classical_reco_sin'], C['pi-rho']['classical_truth_sin'],
                            C['pi-rho']['learned_reco_h_nogeo_Tp'], C['pi-rho']['learned_reco_h_Tp'],
                            C['pi-rho']['exact_h_Tp'], C['pi-rho']['exact_h_optimal_sigma_deg']]),
        ('all decay modes\n(n=29,650)', [None, None, R['base_s43']['Tp'], R['full22_s42']['Tp'],
                                        E['exact:Tp'], E['exact:score(optimal)']]),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    yy = np.arange(len(rows))[::-1]
    for i, (lab, col, mk) in enumerate(series):
        dy = (i - 2.5) * 0.135
        xs, ys, es = [], [], []
        for (rl, vals), yv in zip(rows, yy):
            v = vals[i]
            if v is None:
                continue
            xs.append(v if isinstance(v, float) else v['sigma_deg'])
            es.append(0.0 if isinstance(v, float) else v.get('sigma_deg_se', 0.0))
            ys.append(yv + dy)
        ax.errorbar(xs, ys, xerr=es, fmt=mk, ms=8.5, color=col, elinewidth=1.2, capsize=0,
                    label=lab, zorder=3, ls='none')
    for yv in yy[:-1]:
        ax.axhline(yv - 0.5, color=GRAY, lw=0.6, alpha=0.4)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale('log')
    ax.set_xlabel('σ($\\phi_\\tau$) per 10,000 H→ττ events  [degrees]        ← better')
    ax.set_xlim(0.4, 4.6)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0], ['0.5', '0.7', '1', '1.5', '2', '3', '4'])
    ax.grid(axis='x', color=GRAY, alpha=0.25, lw=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.145), fontsize=8.8, ncol=2)
    ax.text(0.998, 0.985, 'no classical φ*$_{CP}$ construction is implemented for 3-prong sides,\n'
            'so the all-mode row has no classical entry',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.2, color=MUTED)
    fig.suptitle('How precisely the ττ spin correlation fixes the Higgs CP mixing angle',
                 fontsize=12.5, color=INK)
    fig.text(0.5, 0.005, 'Statistical only, per signal event of this simulated sample and selection: no background, '
             'no systematics, no production-rate information.\nThe numbers are meant as a comparison between methods '
             'on identical events, not as a projection for a real measurement.',
             ha='center', fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(out / 'fig2_sensitivity_ladder.png')
    plt.close(fig)


def fig3(data, out):
    """Mode-pair structure: CP is flat at truth level, H/Z is not."""
    S = json_get('sensitivity.json', ['by_mode_pair'])
    mp = json.loads((HERE.parents[0] / 'outputs' / 'mode-pair-auc-origin-20260924'
                     / 'results.json').read_text())['observed']['generator_current']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    x = np.arange(len(PAIRS))
    series = [('exact $h$, triple product', 'exact_Tp', AQUA, '^', -0.16),
              ('exact $h$, optimal statistic', 'exact_optimal', GRAY, '*', 0.0),
              ('reco $h$ (+IP, SV), triple product', 'full22_Tp', BLUE, 's', 0.16)]
    for lab, key, col, mk, dx in series:
        v = [S[p_][key]['sigma_deg'] for p_ in PAIRS]
        e = [S[p_][key].get('sigma_deg_se', 0.0) for p_ in PAIRS]
        a1.errorbar(x + dx, v, yerr=e, fmt=mk, ms=8, color=col, elinewidth=1.2, capsize=0,
                    label=lab, zorder=3)
    a1.set_xticks(x, [PLABEL[p_] for p_ in PAIRS])
    a1.set_ylabel('σ($\\phi_\\tau$) per 10,000 H events  [deg]')
    a1.set_ylim(0.3, 1.95)
    a1.legend(fontsize=8.6, loc='upper left', ncol=1)
    a1.set_title('CP mixing angle  (H events only)', fontsize=10.5, color=INK)
    v = [mp[p_]['auc'] for p_ in PAIRS]
    e = [mp[p_]['se'] for p_ in PAIRS]
    a2.errorbar(x, v, yerr=e, fmt='^', ms=8, color=AQUA, elinewidth=1.2, capsize=0,
                label='exact $h$, fixed-C likelihood ratio', zorder=3)
    a2.set_xticks(x, [PLABEL[p_] for p_ in PAIRS])
    a2.set_ylabel('H/Z weighted AUC')
    a2.set_ylim(0.665, 0.762)
    a2.legend(fontsize=8.6, loc='lower right')
    a2.set_title('H/Z spin discrimination  (H vs Z, 2026-09-24 run)', fontsize=10.5, color=INK)
    for ax in (a1, a2):
        ax.grid(axis='y', color=GRAY, alpha=0.2, lw=0.6)
        ax.set_xlim(-0.55, len(PAIRS) - 0.45)
    fig.suptitle('At truth level the CP angle does not care which decay modes the event has; H/Z does',
                 fontsize=11.5, color=INK)
    fig.text(0.5, 0.005, 'Same validation cohort and the same exact generator-current $h$ in both panels. '
             'Error bars are bootstrap standard deviations; the optimal-statistic points carry no bar.',
             ha='center', fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(out / 'fig3_mode_pair.png')
    plt.close(fig)


def fig4(out):
    """What detector geometry buys: CP vs H/Z, same predicted h."""
    G = json_get('geometry_value.json', [])
    order = ['base_s43', 'sv_s42', 'ip22_s42', 'full_s42', 'full22_s42',
             'full22_shuffle_s42', 'idealip22_s42']
    order = [a for a in order if a in G['arms']]
    lab = [G['arms'][a]['description'] for a in order]
    cp = np.array([G['arms'][a]['cp_effective_lumi_vs_base'] for a in order])
    hz = np.array([G['arms'][a]['hz_effective_lumi_vs_base'] for a in order])
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={'width_ratios': [1.3, 1]})
    real = [i for i, a in enumerate(order) if 'shuffle' not in a and 'ideal' not in a]
    a1.axhline(1, color=GRAY, lw=1, ls=':')
    a1.plot(x[real], cp[real], 'o-', color=BLUE, ms=9, lw=2, label='CP mixing angle $\\phi_\\tau$')
    a1.plot(x[real], hz[real], 's--', color=ORANGE, ms=8, lw=2, label='H/Z spin discrimination')
    extra = [i for i in range(len(order)) if i not in real]
    a1.plot(x[extra], cp[extra], 'o', color=BLUE, ms=9, mfc='white', mew=2)
    a1.plot(x[extra], hz[extra], 's', color=ORANGE, ms=8, mfc='white', mew=2)
    for xi, c, h in zip(x, cp, hz):
        a1.text(xi, c + 0.10, f'{c:.2f}', ha='center', fontsize=8.6, color=BLUE)
        a1.text(xi, h - 0.19, f'{h:.2f}', ha='center', fontsize=8.6, color=ORANGE)
    a1.axvspan(len(real) - 0.5, len(order) - 0.5, color=GRAY, alpha=0.08, lw=0)
    a1.text((len(real) + len(order) - 1) / 2, 3.35, 'control / oracle', ha='center',
            fontsize=8.6, color=MUTED)
    a1.set_xticks(x, lab, rotation=22, ha='right', fontsize=8.6)
    a1.set_ylabel('effective luminosity relative to "no geometry"')
    a1.set_ylim(0.6, 3.55)
    a1.set_xlim(-0.5, len(order) - 0.5)
    a1.legend(fontsize=9.2, loc='upper left')
    a1.set_title('the same predicted $h$, two physics questions', fontsize=10.5, color=INK)
    a1.grid(axis='y', color=GRAY, alpha=0.2, lw=0.6)

    sig = np.array([G['arms'][a]['cp_sigma_deg'] for a in order])
    auc = np.array([G['arms'][a]['hz_auc'] for a in order])
    groups = [('no geometry, + SV only, shuffled control', [i for i, a in enumerate(order)
                                                            if a in ('base_s43', 'sv_s42', 'full22_shuffle_s42')],
               ORANGE, 'o', (8, -14)),
              ('+ IP  (3 arms: IP only, IP+SV v1, IP+SV v2)', [i for i, a in enumerate(order)
                                                               if a in ('ip22_s42', 'full_s42', 'full22_s42')],
               BLUE, 's', (8, 6)),
              ('ideal-IP oracle', [i for i, a in enumerate(order) if a == 'idealip22_s42'],
               VIOLET, 'D', (-10, 14))]
    for glab, idx, col, mk, off in groups:
        if not idx:
            continue
        a2.plot(auc[idx], sig[idx], mk, color=col, ms=9, zorder=3)
        a2.annotate(glab, (np.mean(auc[idx]), np.mean(sig[idx])), textcoords='offset points',
                    xytext=off, fontsize=8.6, color=col)
    a2.plot([G['exact_h']['hz_auc']], [G['exact_h']['cp_sigma_deg']], '*', color=AQUA, ms=16, zorder=3)
    a2.annotate('exact $h$', (G['exact_h']['hz_auc'], G['exact_h']['cp_sigma_deg']),
                textcoords='offset points', xytext=(-8, 12), fontsize=9.2, color=AQUA)
    a2.set_xlabel('H/Z AUC (plug-in fixed-C likelihood ratio)')
    a2.set_ylabel('σ($\\phi_\\tau$) per 10,000 H events  [deg]')
    a2.set_xlim(0.618, 0.745)
    a2.set_ylim(1.85, 0.45)
    a2.grid(color=GRAY, alpha=0.2, lw=0.6)
    a2.set_title('better towards the top right', fontsize=10.5, color=INK)
    fig.suptitle('Impact-parameter information is worth far more to the CP angle than to H/Z tagging',
                 fontsize=12, color=INK)
    fig.text(0.5, 0.005, 'Events with no 3-prong side (37,942; 18,878 H), where the CLEO and generator-current '
             'polarimeters coincide.  SV needs at least two core tracks, hence the null SV-only point.',
             ha='center', fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out / 'fig4_geometry_value.png')
    plt.close(fig)


def fig5(data, out):
    """Audit: linear-response closure, reweighting ESS, Fisher tail, readout saturation."""
    D, h_exact = load_all(data)
    y = D['labels'].astype(bool)
    C0 = c_matrix(0.0)
    hH = h_exact[y]
    S, f = score(hH, C0)
    Tp = triple(hH[:, 0], hH[:, 1])
    fig, ax = plt.subplots(2, 2, figsize=(11.4, 8.0))

    a = ax[0, 0]
    degs = np.arange(-12, 12.5, 1.5)
    fd = [np.average(Tp, weights=weights(hH, np.deg2rad(d), C0)) for d in degs]
    a.plot(degs, fd, 'o', color=BLUE, ms=6, label='reweighted $\\langle T_p\\rangle$ at $\\phi_\\tau$')
    resp = np.cov(np.stack([Tp, S]))[0, 1]
    a.plot(degs, np.mean(Tp) + resp * np.deg2rad(degs), '-', color=ORANGE, lw=2,
           label='linear response Cov($T_p$, $S$)')
    a.set_xlabel('$\\phi_\\tau$ [deg]')
    a.set_ylabel('$\\langle T_p \\rangle = \\langle (h_-\\times h_+)\\cdot\\hat k\\rangle$')
    a.legend(fontsize=8.8)
    a.set_title('(a) the spin reweighting closes on the score', fontsize=10, color=INK)

    a = ax[0, 1]
    ds = np.arange(0, 91, 5.0)
    e = [float(w.sum()**2 / (len(w) * (w**2).sum())) for w in
         (weights(hH, np.deg2rad(d), C0) for d in ds)]
    a.plot(ds, e, 'o-', color=VIOLET, ms=5)
    a.axhline(0.5, color=GRAY, ls=':', lw=1)
    a.set_xlabel('$\\phi_\\tau$ [deg]')
    a.set_ylabel('effective sample size / N')
    a.set_ylim(0, 1.05)
    a.set_title('(b) reweighting cost away from the generated hypothesis', fontsize=10, color=INK)

    a = ax[1, 0]
    order = np.argsort(f)
    fr_ = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    sg, st = [], []
    for fc in fr_:
        keep = np.ones(len(f), bool)
        keep[order[:int(fc * len(f))]] = False
        sg.append(np.rad2deg(1 / np.sqrt(10000 * np.var(S[keep]))))
        st.append(sensitivity(Tp[keep], S[keep], 10000)['sigma_deg'])
    a.plot(100 * fr_, sg, 'o-', color=GRAY, ms=6, label='optimal statistic (Cramér–Rao)')
    a.plot(100 * fr_, st, '^--', color=AQUA, ms=6, label='triple product $T_p$')
    a.set_xlabel('percent of events with the smallest $1+h_-^{T}C_0h_+$ removed')
    a.set_ylabel('σ($\\phi_\\tau$) per 10,000 events [deg]')
    a.legend(fontsize=8.8, loc='center right')
    a.set_title('(c) both statistics lose when the smallest-weight events are cut', fontsize=10, color=INK)

    a = ax[1, 1]
    R = json_get('readout.json', ['arms'])
    for arm, col, mk in (('base_s43', ORANGE, 'o'), ('full22_s42', BLUE, 's')):
        degs_ = [2, 3, 4]
        v = [R[arm][f'fitted_readout_deg{d}']['sigma_deg'] for d in degs_]
        e = [R[arm][f'fitted_readout_deg{d}']['sigma_deg_se'] for d in degs_]
        a.errorbar(degs_, v, yerr=e, fmt=mk + '-', color=col, ms=7, elinewidth=1.1, capsize=0,
                   label=f'{arm}: readout fitted on the train split')
        a.axhline(R[arm]['plugin_Tp']['sigma_deg'], color=col, ls=':', lw=1.4)
        a.text(4.06, R[arm]['plugin_Tp']['sigma_deg'], ' plug-in $T_p$', va='center',
               fontsize=8.4, color=col)
    a.set_ylim(1.15, 1.75)
    a.set_xlim(1.6, 4.7)
    a.set_xticks([2, 3, 4])
    a.text(1.72, 1.70, 'degree 1 omitted: a readout linear in $h_{pred}$\ncannot be CP-odd '
           '(measured σ > 45°, consistent with none)', fontsize=8.2, color=MUTED, va='top')
    a.set_xlabel('polynomial degree of the fitted readout')
    a.set_ylabel('σ($\\phi_\\tau$) per 10,000 H events [deg]')
    a.legend(fontsize=8.4, loc='lower left')
    a.set_title('(d) no readout of $h_{pred}$ beats its own triple product by much', fontsize=10, color=INK)
    fig.suptitle('Validation of the CP-mixing sensitivity estimates', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'fig5_validation.png')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'figures')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig1(args.data, args.output)
    fig2(args.output)
    fig3(args.data, args.output)
    fig4(args.output)
    fig5(args.data, args.output)
    print('wrote', sorted(p.name for p in args.output.glob('*.png')))
