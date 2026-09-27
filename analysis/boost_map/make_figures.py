import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
R1 = json.load(open(os.path.join(HERE, 'results.json')))
RG = json.load(open(os.path.join(HERE, 'results_grid.json')))
RM = json.load(open(os.path.join(HERE, 'results_mode.json')))

INK, MUTED = '#222222', '#6b6b6b'
STY = {  # fixed identity per arm: colour + marker + line style
    'exact': dict(color='#444444', marker='s', ls='-', label='exact $h$ (ceiling)'),
    'base': dict(color='#2a78d6', marker='o', ls='-', label='reco base (vis + MET)'),
    'ipsv': dict(color='#eb6834', marker='^', ls='--', label='reco + 3 IP + SV'),
    'idealip': dict(color='#1baf7a', marker='D', ls='-.', label='reco + ideal IP (oracle)'),
    'cleo_shuffle': dict(color='#eda100', marker='v', ls=':', label='shuffled geometry (control, CLEO $h$)'),
}
MODE_STY = {'pi': dict(color='#2a78d6', marker='o', label=r'$\pi$'),
            'rho': dict(color='#eb6834', marker='^', label=r'$\rho$'),
            '3pi': dict(color='#1baf7a', marker='D', label=r'$3\pi$')}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6,
                     'lines.markersize': 6, 'legend.frameon': False})
POP = 'H vs Z (91.19 GeV), validation 59,390 ev., overlap weights'


def fig1():
    bins = R1['binnings']['bg_gm']
    x = np.array([b['median'] for b in bins])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    for a in ['exact', 'idealip', 'ipsv', 'base', 'cleo_shuffle']:
        y = [b['auc'][a] for b in bins]
        e = [b['auc_se'][a] for b in bins]
        a1.errorbar(x, y, e, capsize=2, **STY[a])
    a1.set_ylabel('H/Z AUC (fixed cross-fitted readout)')
    a1.set_xscale('log')
    a1.legend(fontsize=8, ncol=2, loc='center right', bbox_to_anchor=(1.0, 0.62))
    a1.set_title(r'H/Z spin separation vs $\tau$ boost (8 equal-population bins)', fontsize=10, color=INK)
    for key, st in [('ipsv-base', STY['ipsv']), ('idealip-base', STY['idealip'])]:
        y = [b['diff'][key] for b in bins]
        e = [b['diff_se'][key] for b in bins]
        s = dict(st)
        s['label'] = {'ipsv-base': '(+3 IP + SV) - base', 'idealip-base': '(ideal IP) - base'}[key]
        a2.errorbar(x, y, e, capsize=2, **s)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel('AUC gain over base\n(same events, paired bootstrap)')
    a2.set_xlabel(r'$\sqrt{(\beta\gamma)_1(\beta\gamma)_2}$ of the truth $\tau$ pair (bin median)')
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.set_xticks([60, 80, 100, 150, 200, 250])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.text(0.01, 0.005, POP + '; error bars: bootstrap 1 s.d.', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIG, 'fig1_auc_vs_boost.png'), dpi=160)


def fig2():
    grid = RG['event_grid']
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    for ax, g in zip(axes, grid):
        x = np.array([b['bg_med'] for b in g['bins']])
        for a in ['exact', 'idealip', 'ipsv', 'base']:
            ax.errorbar(x, [b['auc'][a] for b in g['bins']], [b['auc_se'][a] for b in g['bins']], capsize=2, **STY[a])
        f = g['fits_vs_log_bg']
        ax.set_title(f"mean visible fraction $x$ in [{g['x_lo']:.2f}, {g['x_hi']:.2f}]", fontsize=10, color=INK)
        txt = '\n'.join([f"slope / ln($\\beta\\gamma$):",
                         f"base {f['base']['slope']:+.3f}$\\pm${f['base']['slope_se']:.3f}",
                         f"IP+SV$-$base {f['ipsv-base']['slope']:+.3f}$\\pm${f['ipsv-base']['slope_se']:.3f}"])
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.5, color=INK, va='bottom')
        ax.set_xscale('log')
        ax.set_xticks([60, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r'$\sqrt{(\beta\gamma)_1(\beta\gamma)_2}$ (bin median)')
    axes[0].set_ylabel('H/Z AUC')
    axes[0].set_ylim(0.55, 0.76)
    axes[1].legend(fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.64), ncol=1)
    fig.suptitle(r'At fixed neutrino energy share $x=E_{\rm vis}/E_\tau$, H/Z separation does not depend on the $\tau$ boost; the reco level moves with $x$',
                 fontsize=10, color=INK)
    fig.text(0.01, 0.005, POP + r'; $x$ tercile $\times$ $\beta\gamma$ quartile, ~4,950 ev./cell; bootstrap 1 s.d.; truth $x$ and $\beta\gamma$',
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'fig2_auc_boost_at_fixed_x.png'), dpi=160)


def fig3():
    g = RG['side_grid']
    be, xe = np.array(g['bg_edges']), np.array(g['x_edges'])
    base = np.array(g['hq']['base'])[..., 0]
    gain = np.array(g['hq']['ipsv'])[..., 0] - base
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, M, title, cmap, vmin, vmax in [
            (axes[0], base, r'reco base: corr($\hat h_\perp$, $h_\perp$)', 'Blues', 0.4, 0.8),
            (axes[1], gain, r'gain from 3 IP + SV: $\Delta$corr($\hat h_\perp$, $h_\perp$)', 'Oranges', 0.0, 0.18)]:
        im = ax.imshow(M.T, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=(0, M.shape[0], 0, M.shape[1]))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(i + 0.5, j + 0.5, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if (v - vmin) / (vmax - vmin) > 0.6 else INK)
        ax.set_xticks(np.arange(M.shape[0] + 1))
        ax.set_xticklabels([f'{v:.0f}' for v in be[:-1]] + ['max'], fontsize=8)
        ax.set_yticks(np.arange(M.shape[1] + 1))
        ax.set_yticklabels([f'{v:.2f}' for v in xe], fontsize=8)
        ax.set_xlabel(r'$\beta\gamma$ of this $\tau$ (bin edges, sextiles)')
        ax.set_ylabel(r'visible energy fraction $x=E_{\rm vis}/E_\tau$ (quintile edges)')
        ax.grid(False)
        ax.set_title(title, fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.01, 0.005, 'per tau side (both sides pooled, 118,780 sides), all decay modes; transverse = mean of n and r components; '
             'weighted Pearson correlation with exact (generator-current) h', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, 'fig3_hquality_boost_x_map.png'), dpi=160)


def fig4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, st in MODE_STY.items():
        rows = RM[m][1:] if m == 'pi' else RM[m]   # pi lowest bin: x reweighting has poor support (se 0.04)
        x = [r['bg_med'] for r in rows]
        a1.errorbar(x, [r['base_T_xrw'] for r in rows], [r['base_T_xrw_se'] for r in rows], color=st['color'],
                    marker=st['marker'], ls='-', capsize=2, label=f"{st['label']} base")
        a1.plot(x, [r['ipsv_T_xrw'] for r in rows], color=st['color'], marker=st['marker'], ls='--', mfc='white',
                label=f"{st['label']} + 3 IP + SV")
        a1.plot(x, [r['idealip_T_xrw'] for r in rows], color=st['color'], marker=st['marker'], ls=':', mfc='none', alpha=0.8,
                label=f"{st['label']} + ideal IP")
        a2.plot([r['bg_med'] for r in RM[m]], [r['axis_err_over_theta_med'] for r in RM[m]], color=st['color'],
                marker=st['marker'], ls='-', label=st['label'])
    a1.set_xscale('log')
    a1.set_ylabel(r'corr($\hat h_\perp$, $h_\perp$), $x$ reweighted to mode inclusive')
    a1.set_xlabel(r'$\beta\gamma$ of this $\tau$ (sextile medians)')
    a1.legend(fontsize=7, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    a2.set_xscale('log')
    a2.set_ylabel(r'median  $\delta\theta_{\rm vis\,axis}$ / $\theta(\rm vis,\tau)$')
    a2.set_xlabel(r'$\beta\gamma$ of this $\tau$ (sextile medians)')
    a2.set_title('reco visible-axis error in units of the angle to be resolved', fontsize=10, color=INK)
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.set_xticks([40, 100, 200, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.text(0.01, 0.005, r'per tau side by truth decay mode; left: error bars = bootstrap 1 s.d. of base (IP gain s.e. 0.002-0.008); '
             r'$\pi$ lowest-$\beta\gamma$ bin dropped from left (x reweighting support); right: truth vs reco visible momentum direction',
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, 'fig4_mode_boost_profiles.png'), dpi=160)


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, key, xl in [(axes[0], 'x_mean_truth', r'truth mean $x=E_{\rm vis}/E_\tau$ (sextile medians)'),
                        (axes[1], 'x_coll_mean', r'collinear-approximation mean $x$ from reco vis + MET (valid events only)')]:
        rows = RG[key]
        xs = [r['med'] for r in rows]
        for a in ['exact', 'idealip', 'ipsv', 'base']:
            ax.errorbar(xs, [r['auc'][a] for r in rows], [r['auc_se'][a] for r in rows], capsize=2, **STY[a])
        ax.set_xlabel(xl, fontsize=9)
    axes[0].set_ylabel('H/Z AUC')
    axes[0].legend(fontsize=8)
    axes[1].set_title(f"collinear $x$ in (0,1) for both sides: {100*RG['x_coll_valid_frac']:.0f}% of events", fontsize=9, color=INK)
    fig.text(0.01, 0.005, POP + '; bootstrap 1 s.d.', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, 'fig5_auc_vs_x.png'), dpi=160)


def fig6():
    RI = json.load(open(os.path.join(HERE, 'results_ip.json')))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for m in ('pi', 'rho'):
        st = MODE_STY[m]
        rows = RI[m]
        x = [r['bg_med'] for r in rows]
        a1.plot(x, [r['dphi_abs_med_deg'] for r in rows], color=st['color'], marker=st['marker'], label=st['label'])
        a2.plot(x, [r['ip_um_med'] for r in rows], color=st['color'], marker=st['marker'], label=st['label'])
    a1.set_ylabel(r'median $|\Delta\phi|$ of IP direction around track [deg]')
    a1.set_title('IP azimuth error vs boost', fontsize=10, color=INK)
    a1.text(0.97, 0.05, '\n'.join(f"{MODE_STY[m]['label']}: median track $p_T$ " + ' / '.join(f"{r['track_pt_med']:.0f}" for r in RI[m]) + ' GeV' for m in ('pi', 'rho')), transform=a1.transAxes, ha='right', fontsize=7.5, color=INK)
    a2.set_ylabel(r'median $|{\rm IP}_{\rm PV}|$ [$\mu$m]')
    a2.set_ylim(0, 90)
    a2.set_title(r'IP size stays $\sim c\tau$', fontsize=10, color=INK)
    for ax in (a1, a2):
        ax.set_xscale('log')
        ax.set_xticks([40, 100, 200, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r'$\beta\gamma$ of this $\tau$ (sextile medians)')
        ax.legend(fontsize=8)
    a1.set_ylim(0, 30)
    fig.text(0.01, 0.005, 'reco 1-prong sides (truth mode pi or rho), leading core track; PV-referenced 3-D IP, along-track part removed; '
             'reference = truth tau direction perpendicular to the track', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, 'fig6_ip_azimuth_vs_boost.png'), dpi=160)


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
