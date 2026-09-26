"""Figures for the tau-polarisation run.  Needs the q0-q5 json outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pol_data as PD
import pol_tools as PT
import pol_visible as V

BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 160})
FIGS = PD.FIGS
FIGS.mkdir(exist_ok=True)
Q = {n: json.loads((PD.RESULTS / f'{n}.json').read_text())
     for n in ('q0_structure', 'q1_sensitivity', 'q2_classical',
               'q3_measure', 'q4_transport', 'q5_translate')}
MLAB = {'pi': 'π ν', 'rho': 'ρ ν (π±π⁰)', '3pi': 'a₁ ν (3π)'}

S = PD.load_surface()
lab, h, modes, ow = S['labels'], S['h_gen'], S['modes'], S['weights']
z = ~lab
L = PD.load_ladder()
pt_vis = V.pt(np.array(L['reco_visible_tau_lab4'], float))
P0 = PT.P0_SAMPLE


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
    b = np.linspace(-1, 1, 61)
    for m, c, t in ((lab, ORANGE, 'H → ττ:  B = 0 exactly'),
                    (z, BLUE, 'Z → ττ:  $P_\\tau$ = −0.147')):
        v = np.concatenate([h[m, 0, 2], h[m, 1, 2]])
        ax[0].hist(v, bins=b, density=True, histtype='step', lw=1.8, color=c, label=t)
        ax[0].axvline(v.mean(), color=c, ls='--', lw=1.1)
    ax[0].set_ylim(0, 0.78)
    ax[0].set_xlabel('exact $h_k$  (per τ side)')
    ax[0].set_ylabel('density (unit weight)')
    ax[0].set_title('(a) the measured longitudinal component', loc='left', fontsize=10)
    ax[0].legend(loc='upper left', fontsize=8.5)
    ax[0].text(0.30, 0.10, 'dashed: sample means\n'
               r'$\langle h_k\rangle_H=+0.037$, $\langle h_k\rangle_Z=+0.006$',
               transform=ax[0].transAxes, fontsize=8, color=MUTED)

    # (b) the unpolarised measure, 1/f-unfolded on the H rows, per decay mode
    fH = PT.f_density(h[lab], 0.0, PT.C_H)
    u0 = (1.0 / fH)[:, None] * np.ones((1, 2))
    hk = h[lab][:, :, 2]
    for md, c in ((0, BLUE), (1, ORANGE), (3, AQUA)):
        sel = modes[lab] == md
        w = np.where(sel, u0, 0.0).ravel()
        bb = np.linspace(-1, 1, 33)
        y, _ = np.histogram(hk.ravel(), bins=bb, weights=w, density=True)
        ax[1].step(0.5 * (bb[1:] + bb[:-1]), y, where='mid', lw=1.8, color=c,
                   label=f'{MLAB[PD.MODES[md]]}  $E_0[h_k]$={Q["q0_structure"]["per_mode_p0"][PD.MODES[md]]["H"]["E0_hk"]:+.3f}')
    ax[1].axhline(0.5, color=GRAY, ls=':', lw=1.2)
    ax[1].text(-0.97, 0.13, 'dotted: flat = no acceptance distortion',
               fontsize=8, color=MUTED)
    ax[1].set_ylim(0, 0.85)
    ax[1].set_xlabel('$h_k$'), ax[1].set_ylabel('unpolarised measure $p_0(h_k)$')
    ax[1].set_title('(b) what the visible-$p_T$ selection does to $p_0$',
                    loc='left', fontsize=10)
    ax[1].legend(loc='upper left', fontsize=8.5)

    # (c) profile of <h_k> vs visible pT
    edges = np.percentile(pt_vis, np.linspace(0, 97, 13))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for m, c, mk, t in ((lab, ORANGE, 'o', 'H'), (z, BLUE, 's', 'Z')):
        flat_pt, flat_hk = pt_vis[m].ravel(), h[m][:, :, 2].ravel()
        i = np.clip(np.digitize(flat_pt, edges[1:-1]), 0, len(ctr) - 1)
        mu = np.array([flat_hk[i == k].mean() for k in range(len(ctr))])
        se = np.array([flat_hk[i == k].std(ddof=1) / np.sqrt((i == k).sum())
                       for k in range(len(ctr))])
        ax[2].errorbar(ctr, mu, se, color=c, marker=mk, ms=4, lw=1.4, ls='-', label=t)
    ax[2].axhline(0, color=GRAY, ls=':', lw=1.2)
    ax[2].axhline(P0 / 3, color=VIOLET, ls='--', lw=1.2)
    ax[2].text(edges[1], P0 / 3 + 0.004, r'$P_\tau/3$ = the answer for an isotropic $p_0$',
               fontsize=8, color=VIOLET)
    ax[2].set_xlim(edges[0] - 3, edges[-1] + 3)
    ax[2].set_xlabel('reco visible $p_T$ of the τ side  [GeV]   (lowest 97%)')
    ax[2].set_ylabel(r'$\langle h_k\rangle$')
    ax[2].set_title('(c) the offset is a selection effect', loc='left', fontsize=10)
    ax[2].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig1_first_moment.png')
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2():
    q1, q2, q5 = Q['q1_sensitivity'], Q['q2_classical'], Q['q5_translate']
    items = [('exact h, Cramér–Rao bound', q1['cramer_rao_exact_h']['sigma'], None, VIOLET),
             ('exact h, $h_k$ sum', q1['rows']['exact_hk']['sigma'],
              q1['rows']['exact_hk'].get('sigma_se'), VIOLET),
             ('reco h + ideal-IP oracle', q1['rows']['idealip22_s42_hk']['sigma'],
              q1['rows']['idealip22_s42_hk'].get('sigma_se'), AQUA),
             ('reco h + 3 IP + SV', q1['rows']['full22_s42_hk']['sigma'],
              q1['rows']['full22_s42_hk'].get('sigma_se'), BLUE),
             ('reco h, no geometry', q1['rows']['base_s43_hk']['sigma'],
              q1['rows']['base_s43_hk'].get('sigma_se'), BLUE),
             ('reco h, shuffled geometry', q1['rows']['full22_shuffle_s42_hk']['sigma'],
              q1['rows']['full22_shuffle_s42_hk'].get('sigma_se'), GRAY),
             ('best visible + MET readout', q2['rows']['vismet_best']['sigma'],
              q2['rows']['vismet_best'].get('sigma_se'), ORANGE),
             ('best visible-only readout', q2['rows']['vis_best']['sigma'],
              q2['rows']['vis_best'].get('sigma_se'), ORANGE),
             ('$E_{vis}/E_\\tau$ with truth $E_\\tau$ (LEP-style)',
              q2['rows']['x_truth']['sigma'], q2['rows']['x_truth'].get('sigma_se'), YELLOW)]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(items))[::-1]
    for (name, s, se, c), yy in zip(items, y):
        ax.barh(yy, s, height=0.62, color=c, alpha=0.85)
        if se:
            ax.errorbar(s, yy, xerr=se, color=INK, capsize=2.5, lw=1.0)
        ax.text(s + 0.00012, yy, f'{s:.5f}', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y), ax.set_yticklabels([i[0] for i in items], fontsize=9)
    ax.set_xlabel(r'$\sigma(P_\tau)$ for $10^5$ selected Z → $\tau_h\tau_h$ events'
                  '   (signal only, shape only)')
    lep = Q['q5_translate']['lep_sigma_P']
    ax.axvline(lep, color=INK, ls='--', lw=1.2)
    ax.text(lep, len(items) - 0.4, ' LEP $\\sigma(A_\\tau)=0.0043$', fontsize=8.5,
            color=INK, va='top')
    sec = ax.secondary_xaxis('top', functions=(lambda v: v / 7.870, lambda v: v * 7.870))
    sec.set_xlabel(r'$\sigma(\sin^2\theta^{\rm lept}_{\rm eff})$')
    ax.set_xlim(0, max(i[1] for i in items) * 1.13)
    ax.set_title('All rows use the same 29,740 validation Z events (scaled to $10^5$);\n'
                 'only the observable changes.  Row permutation of $h_{pred}$ gives'
                 ' zero response.', loc='left', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig2_sensitivity_ladder.png')
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3():
    tab = Q['q5_translate']['three_measurements']
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.0))
    names = ['cp_angle', 'entanglement', 'polarisation']
    short = ['CP mixing angle $\\phi_\\tau$\n(direction of $C_\\perp$)\n2026-09-25',
             'entanglement $\\langle W\\rangle$\n(magnitude of $C$)\n2026-09-26',
             'polarisation $P_\\tau$\n(first moment $B$, along $k$)\nthis run']
    x = np.arange(3)
    r = [tab[n]['reco_over_exact'] for n in names]
    g = [tab[n]['ip_sv_gain'] for n in names]
    gi = [tab[n]['ideal_ip_gain'] for n in names]
    ax[0].bar(x - 0.21, r, 0.38, color=BLUE, alpha=0.85, label='reco / exact $h$')
    ax[0].bar(x + 0.21, g, 0.38, color=ORANGE, alpha=0.85,
              label='gain from 3 IP + SV')
    ax[0].plot(x + 0.21, gi, ls='none', marker='D', ms=6, color=VIOLET,
               label='gain from an ideal IP')
    for xx, (a, b) in enumerate(zip(r, g)):
        ax[0].text(xx - 0.21, a + 0.04, f'{a:.2f}', ha='center', fontsize=9)
        ax[0].text(xx + 0.21, b + 0.04 if b > 1.1 else b - 0.11, f'{b:.2f}',
                   ha='center', fontsize=9)
    ax[0].axhline(1.0, color=GRAY, ls=':', lw=1.2)
    ax[0].set_xticks(x), ax[0].set_xticklabels(short, fontsize=8.5)
    ax[0].set_ylabel('factor in the uncertainty')
    ax[0].set_ylim(0.85, 3.1)
    ax[0].legend(fontsize=8.5, loc='upper right')
    ax[0].set_title('(a) the same reconstructed $h$, three measurements',
                    loc='left', fontsize=10)

    # (b) why: per-component regression quality
    comp = ['$h_n$', '$h_r$', '$h_k$']
    arms = [('base_s43', 'reco, no geometry', BLUE, 'o', '-'),
            ('full22_s42', 'reco + 3 IP + SV', ORANGE, 's', '-'),
            ('idealip22_s42', 'reco + ideal-IP', VIOLET, 'D', '--')]
    for arm, d, c, mk, ls in arms:
        hp = PD.load_arm(arm, S)['h_pred']
        cr = [np.corrcoef(h[:, :, i].ravel(), hp[:, :, i].ravel())[0, 1] for i in range(3)]
        ax[1].plot([0, 1, 2], cr, marker=mk, ls=ls, lw=1.6, color=c, ms=6, label=d)
    ax[1].set_xticks([0, 1, 2]), ax[1].set_xticklabels(comp)
    ax[1].set_ylabel('correlation with the exact component')
    ax[1].set_ylim(0.5, 0.88)
    ax[1].legend(fontsize=8.5, loc='upper left')
    ax[1].set_title('(b) the reason: only the transverse components are IP limited',
                    loc='left', fontsize=10)
    ax[1].text(0.04, 0.06, 'all 59,390 events, both τ sides', transform=ax[1].transAxes,
               fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig3_complementarity.png')
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig4():
    q2 = Q['q2_classical']['rows']
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.1),
                           gridspec_kw={'width_ratios': [1.15, 1]})
    L2 = PD.load_ladder()
    tvis, tnu = (np.array(L2['truth_visible_tau_lab4'], float),
                 np.array(L2['truth_neutrino_lab4'], float))
    x = tvis[:, :, V.E] / (tvis[:, :, V.E] + tnu[:, :, V.E])
    for md, c, mk in ((0, BLUE, 'o'), (1, ORANGE, 's'), (3, AQUA, '^')):
        sel = (modes == md) & z[:, None]
        xx, yy = (2 * x - 1)[sel], h[:, :, 2][sel]
        e = np.linspace(-1, 1, 21)
        i = np.clip(np.digitize(xx, e[1:-1]), 0, len(e) - 2)
        ctr = 0.5 * (e[1:] + e[:-1])
        mu = np.array([yy[i == k].mean() if (i == k).sum() > 20 else np.nan
                       for k in range(len(ctr))])
        sd = np.array([yy[i == k].std() if (i == k).sum() > 20 else np.nan
                       for k in range(len(ctr))])
        ax[0].plot(ctr, mu, marker=mk, color=c, lw=1.6, ms=5, label=MLAB[PD.MODES[md]])
        ax[0].fill_between(ctr, mu - sd, mu + sd, color=c, alpha=0.13, lw=0)
    ax[0].plot([-1, 1], [-1, 1], color=GRAY, ls=':', lw=1.3)
    ax[0].text(-0.95, 0.83, 'dotted: $h_k = 2x-1$, i.e. analysing power 1',
               fontsize=8, color=MUTED)
    ax[0].set_xlabel(r'$2E_{\rm vis}/E_\tau - 1$   (truth $E_\tau$)')
    ax[0].set_ylabel(r'exact $h_k$')
    ax[0].set_title('(a) the energy fraction is the whole polarimeter only for π',
                    loc='left', fontsize=10)
    ax[0].legend(fontsize=8.5, loc='lower right')

    keys = [('exact_h', 'exact $h$', VIOLET), ('x_truth', '$E_{vis}/E_\\tau$', YELLOW),
            ('vis_best', 'best visible only', ORANGE),
            ('vismet_best', 'best visible + MET', GRAY),
            ('full22_s42', 'network $h$ + IP + SV', BLUE)]
    xpos = np.arange(3)
    wid = 0.16
    for j, (k, lbl, c) in enumerate(keys):
        vals = [q2[f'{m}_{k}']['sigma'] for m in ('pi', 'rho', '3pi')]
        vals = [min(v, 0.05) for v in vals]
        b = ax[1].bar(xpos + (j - 2) * wid, vals, wid, color=c, alpha=0.85, label=lbl)
        for xx, v, raw in zip(xpos + (j - 2) * wid, vals,
                              [q2[f'{m}_{k}']['sigma'] for m in ('pi', 'rho', '3pi')]):
            if raw > 0.05:
                ax[1].text(xx, 0.0505, f'{raw:.2f}↑', ha='center', fontsize=7.5,
                           color=c, rotation=90)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([MLAB[m] for m in ('pi', 'rho', '3pi')])
    ax[1].set_ylabel(r'$\sigma(P_\tau)$ per $10^5$ Z events'
                     '\n(only this mode contributes)')
    ax[1].set_ylim(0, 0.056)
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_title('(b) per decay mode', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig4_classical.png')
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig5():
    q3, q4, q5 = Q['q3_measure'], Q['q4_transport'], Q['q5_translate']
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))
    inj = q3['injection']
    xs = np.array(sorted(float(k) for k in inj))
    ys = np.array([inj[f'{v:+.6f}']['P_hat'] for v in xs])
    ax[0].plot(xs, ys, marker='o', color=BLUE, lw=1.6, ms=5, label='H-control estimate')
    ax[0].plot(xs, xs, color=GRAY, ls=':', lw=1.3, label='perfect recovery')
    ax[0].axvline(P0, color=VIOLET, ls='--', lw=1.1)
    ax[0].text(P0 + 0.006, -0.30, 'the SM value', fontsize=8, color=VIOLET, rotation=90)
    ax[0].set_xlabel('injected $P_\\tau$ (exact-$h$ reweighting of the Z rows)')
    ax[0].set_ylabel('recovered $P_\\tau$')
    ax[0].set_title('(a) the bias is an offset, not a slope', loc='left', fontsize=10)
    ax[0].legend(fontsize=8.5, loc='upper left')
    ax[0].text(0.04, 0.06, 'offset −0.037 … −0.044 over the whole range',
               transform=ax[0].transAxes, fontsize=8, color=MUTED)

    tr = q4['transport']
    order = ['none', 'overlap weights (existing)', 'per-side pT_vis',
             'per-side pT_vis + mode', 'per-side pT_vis + eta + mode']
    short = ['no reweighting', 'existing overlap\nweights',
             'per-side $p_T^{vis}$', '+ decay mode', '+ η']
    y = np.arange(len(order))[::-1]
    ax[1].barh(y, [tr[k]['bias'] for k in order], 0.6, color=ORANGE, alpha=0.85)
    est = q4['estimator_closure']['H -> H  (truth P = 0)']
    ax[1].axvline(0, color=INK, lw=1.0)
    ax[1].axvspan(-est['sd'], est['sd'], color=GRAY, alpha=0.25, lw=0)
    ax[1].text(0.004, len(order) - 1.2, 'grey band: estimator closure\n(H→H halves, ±1 sd)',
               fontsize=8, color=MUTED, ha='left', va='top')
    ax[1].set_ylim(-0.6, len(order) - 0.2)
    ax[1].set_yticks(y), ax[1].set_yticklabels(short, fontsize=9)
    ax[1].set_xlabel('bias of $\\hat P_\\tau$  (truth $-0.1470$)')
    ax[1].set_title('(b) most of it is H+jet → Z+jet kinematics', loc='left', fontsize=10)

    dP = abs(q4['dP_hat_dE0_hk'])
    N = np.logspace(3.5, 8, 60)
    sig = q5['ladder']['full22_s42_hk']['sigma_P'] * np.sqrt(1e5 / N)
    ax[2].loglog(N, sig / dP, color=BLUE, lw=1.8,
                 label='needed accuracy on $E_0[h_k]$')
    for k, c, t in ((abs(tr['none']['bias']) / dP, GRAY, 'no reweighting'),
                    (abs(tr['per-side pT_vis + eta + mode']['bias']) / dP, ORANGE,
                     'after reweighting')):
        ax[2].axhline(k, color=c, ls='--', lw=1.3)
        ax[2].text(N[1], k * 1.15, f'{t}: {k:.1e}', fontsize=8, color=c)
    ax[2].set_xlabel('selected Z → $\\tau_h\\tau_h$ events')
    ax[2].set_ylabel('accuracy required on $E_0[h_k]$')
    ax[2].set_title('(c) a first-moment measurement is an\n     acceptance measurement',
                    loc='left', fontsize=10)
    ax[2].legend(fontsize=8.5, loc='lower left')
    ax[2].set_ylim(2e-5, 1e-1)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig5_systematic.png')
    plt.close(fig)


if __name__ == '__main__':
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn()
        print('wrote', fn.__name__)
