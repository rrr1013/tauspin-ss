"""Figures for the ditau entanglement run.

All figures use the 59,390-event canonical validation surface; H rows (29,650)
unless stated.  Exact `h` is the generator-current polarimeter (2026-09-24
teacher); reco `h` is `h_pred` of the named point-h arm.  No ATLAS labels: this
sample and this selection are not an ATLAS result.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ent_tools as E
from ent_data import load_arm, load_surface

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
THETA_H = E.pack(np.zeros(3), np.zeros(3), np.diag([1.0, 1.0, -1.0]))
plt.rcParams.update({'figure.dpi': 140, 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.25, 'legend.framealpha': 0.9,
                     'axes.axisbelow': True})


def load():
    S = load_surface()
    R = json.loads((HERE / 'results' / 'reco.json').read_text())
    U = json.loads((HERE / 'results' / 'unfold.json').read_text())
    return S, R, U


# --------------------------------------------------------------- figure 1
def fig1_states(S):
    """The two spin states, through the distributions that carry them."""
    from z_density import density as zd
    h, y = S['h_gen'], S['labels']
    _, Bm, Bp, CZ = zd()
    th = {'H': THETA_H, 'Z': E.pack(-Bm, -Bp, CZ)}
    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.95],
                          height_ratios=[1.0, 0.95],
                          hspace=0.55, wspace=0.35,
                          left=0.06, right=0.985, top=0.855, bottom=0.08)
    axp = fig.add_subplot(gs[0, 0]); axt = fig.add_subplot(gs[0, 1])
    axH = fig.add_subplot(gs[1, 0]); axZ = fig.add_subplot(gs[1, 1])
    axl = fig.add_subplot(gs[:, 2])
    edges = np.linspace(-1, 1, 21)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for m, nm, c, ls, mk in ((y, r'$H\to\tau\tau$', 'C0', '-', 'o'),
                             (~y, r'$Z\to\tau\tau$', 'C3', '--', 's')):
        hm, hp = h[m, 0], h[m, 1]
        idx = np.digitize(hm[:, 2], edges) - 1
        prof = np.array([hp[idx == i, 2].mean() for i in range(len(ctr))])
        err = np.array([hp[idx == i, 2].std() / max(np.sqrt((idx == i).sum()), 1)
                        for i in range(len(ctr))])
        axp.errorbar(ctr, prof, yerr=err, fmt=mk, ls=ls, color=c, ms=4, capsize=2,
                     label=f'{nm}  (n={m.sum():,})')
        dphi = np.mod(np.arctan2(hm[:, 1], hm[:, 0]) - np.arctan2(hp[:, 1], hp[:, 0]),
                      2 * np.pi)
        wgt = np.linalg.norm(hm[:, :2], axis=1) * np.linalg.norm(hp[:, :2], axis=1)
        axt.hist(dphi, bins=36, range=(0, 2 * np.pi), weights=wgt, density=True,
                 histtype='step', color=c, lw=1.7, ls=ls, label=nm)
    axp.plot(ctr, ctr / 3, 'k:', lw=1.2, label=r'$C_{kk}\,h_{-k}/3$  ($C_{kk}=+1$)')
    axp.plot(ctr, -ctr / 3, 'k-.', lw=1.2, label=r'$C_{kk}=-1$')
    axp.set_xlabel(r'$h_-\cdot\hat k$'); axp.set_ylabel(r'$\langle h_+\cdot\hat k\rangle$')
    axp.set_title('longitudinal: H anti-correlated, Z correlated', fontsize=9)
    axp.legend(fontsize=7)
    axt.set_xlabel(r'$\Delta\phi=\phi(h_{-\perp})-\phi(h_{+\perp})$  [rad]')
    axt.set_ylabel(r'normalised, weight $|h_{-\perp}||h_{+\perp}|$')
    axt.set_title('transverse: H aligned, Z flat', fontsize=9)
    axt.legend(fontsize=7.5)
    lab = [r'$\uparrow\uparrow$', r'$\uparrow\downarrow$',
           r'$\downarrow\uparrow$', r'$\downarrow\downarrow$']
    for ax, nm in ((axH, 'H'), (axZ, 'Z')):
        rho = np.real(E.theta_to_density(th[nm]))
        im = ax.imshow(rho, cmap='RdBu_r', vmin=-0.6, vmax=0.6)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{rho[i, j]:+.2f}', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(4), lab); ax.set_yticks(range(4), lab); ax.grid(False)
        o = E.observables(th[nm])
        ax.set_title(rf'generated $\rho_{nm}$ (real part), along $\hat k$'
                     + f'\nconcurrence {o["concurrence"]:.2f},  negativity '
                     f'{max(o["neg_eig_pt"], 0):.2f}', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    rows = [('generated', np.diag(np.array(E.unpack(th['H'])[2])),
             np.diag(np.array(E.unpack(th['Z'])[2])))]
    axl.axis('off')
    txt = (r'$\rho_H$: pure Bell state $(|{\uparrow\downarrow}\rangle+'
           r'|{\downarrow\uparrow}\rangle)/\sqrt{2}$' '\n'
           r'   $C_H=\mathrm{diag}(1,1,-1)$, exact for any $\beta$' '\n'
           r'   concurrence 1, negativity 1/2, $m_{12}=2$' '\n'
           r'   CHSH bound $2\sqrt{2}$ reachable' '\n\n'
           r'$\rho_Z$: separable classical mixture' '\n'
           r'   $0.574\,|{\uparrow\uparrow}\rangle\langle\cdot|'
           r'+0.426\,|{\downarrow\downarrow}\rangle\langle\cdot|$' '\n'
           r'   $C_Z=\mathrm{diag}(0,0,+1)$, $B_\mp=0.147\,\hat k$' '\n'
           r'   concurrence 0, $m_{12}=1$' '\n'
           r'   no CHSH violation possible' '\n\n'
           'Both from the polarised Dirac trace\n'
           '(cp_density / z_density), not fitted.')
    axl.text(0.0, 1.0, txt, va='top', ha='left', fontsize=8.4)
    fig.suptitle('The two ditau spin states of this sample: a maximally entangled Bell '
                 'state (H) and a separable classical mixture (Z)\n'
                 r'generator-current exact $h$; $(n,r,k)$ basis with $\hat k$ the '
                 r'$\tau^+$ direction in the pair rest frame', fontsize=9.5, y=0.975)
    fig.savefig(FIG / 'fig1_two_states.png')
    plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig2_p0(S):
    """Why the textbook estimator fails: p0 is not isotropic."""
    h, y, modes = S['h_gen'], S['labels'], S['modes']
    phi = E.features(h[y])
    w_off = 1.0 / E.f_nominal(phi, THETA_H)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    styles = {0: ('$\\pi$', 'C0', '-'), 1: (r'$\rho$', 'C1', '--'), 3: (r'$3\pi$', 'C2', ':')}
    for md, (nm, c, ls) in styles.items():
        sel = modes[y][:, 0] == md
        ax[0].hist(h[y][sel, 0, 2], bins=40, range=(-1, 1), weights=w_off[sel],
                   density=True, histtype='step', color=c, ls=ls, lw=1.6,
                   label=f'{nm}  (n={sel.sum():,})')
    ax[0].axhline(0.5, color='0.4', lw=1.0, ls='-.', label='isotropic $p_0$')
    ax[0].set_xlabel(r'$h_-\cdot\hat k$   (unpolarised measure $p_0$)')
    ax[0].set_ylabel('normalised')
    ax[0].set_title(r'$p_0$ after selection, by decay mode of the $\tau^-$', fontsize=9)
    ax[0].legend(fontsize=7.5)
    names = ['nn', 'rr', 'kk']
    U = json.loads((HERE / 'results' / 'unfold.json').read_text())
    est = U['H']['estimators']
    x = np.arange(3)
    truth = np.diag(np.array(U['H']['C_generated']))
    ax[1].axhline(0, color='k', lw=1.2, ls='-', label='generated value')
    for k, (lab, c, mk, off) in (('naive', ('textbook plug-in', 'C3', 'o', 0.12)),
                                 ('moment', ('second-moment unfolding', 'C0', 's', -0.12))):
        d = np.diag(np.array(est[k]['C'])) - truth
        e = np.diag(np.array(est[k]['sd_C']))
        ax[1].errorbar(x + off, d, yerr=e, fmt=mk, color=c, ms=6, capsize=3, label=lab)
    ax[1].set_xticks(x, [f'$C_{{{n}}}$' for n in names])
    ax[1].set_ylabel(r'estimate $-$ generated value')
    ax[1].set_title(r'$H$, all modes: the plug-in is biased'
                    '\n'
                    r'($\min\,\mathrm{eig}\,\hat\rho = '
                    f'{est["naive"]["observables"]["min_eig_rho"]:+.3f}$'
                    ', not a density matrix)', fontsize=9)
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_selection_bias.png')
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig3_instrumental(S, R):
    """The two predicted polarimeters are correlated with no spin at all."""
    h, y = S['h_gen'], S['labels']
    A = load_arm('full22_s42', S)
    hp = A['h_pred'][y]
    phi = E.features(h[y])
    w_off = 1.0 / E.f_nominal(phi, THETA_H)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for j, (v, ttl, wgt) in enumerate((
            (h[y], r'exact $h$, spin reweighted off', w_off),
            (hp, r'reco $h_{\rm pred}$ (+IP/SV), spin reweighted off', w_off),
            (hp, r'reco $h_{\rm pred}$ (+IP/SV), as generated', None))):
        H2, _, _ = np.histogram2d(v[:, 0, 0], v[:, 1, 0], bins=36,
                                  range=[[-1, 1], [-1, 1]], weights=wgt, density=True)
        im = ax[j].imshow(H2.T, origin='lower', extent=[-1, 1, -1, 1], aspect='auto',
                          cmap='magma')
        w = np.ones(len(v)) if wgt is None else wgt
        c = np.average(v[:, 0, 0] * v[:, 1, 0], weights=w)
        eg = np.linspace(-1, 1, 13); ct = 0.5 * (eg[1:] + eg[:-1])
        ii = np.digitize(v[:, 0, 0], eg) - 1
        pr = np.array([np.average(v[ii == t, 1, 0], weights=w[ii == t])
                       if (ii == t).sum() > 20 else np.nan for t in range(len(ct))])
        ax[j].plot(ct, pr, 'o-', color='C0', ms=3.5, lw=1.4, mec='w', mew=0.6)
        ax[j].set_xlabel(r'$h_-\cdot\hat n$'); ax[j].set_ylabel(r'$h_+\cdot\hat n$')
        ax[j].set_title(f'{ttl}\n' + r'$\langle h_{-n}h_{+n}\rangle = $' + f'{c:+.4f}',
                        fontsize=8.5)
        plt.colorbar(im, ax=ax[j])
    fig.suptitle('Two polarimeters predicted from the same event are correlated even when '
                 'the spin correlation is removed\n'
                 'H validation events, weights $1/(1+h_-^T C_H h_+)$ for the first two '
                 'panels; the line is the binned profile', fontsize=9, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG / 'fig3_instrumental_correlation.png')
    plt.close(fig)

    arms = list(R['arms'].keys())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(arms))
    for i, (nm, c, mk) in enumerate((('nn', 'C0', 'o'), ('rr', 'C1', 's'), ('kk', 'C2', '^'))):
        fr = [R['arms'][a]['instrumental']['fraction_diag'][i] for a in arms]
        ax.plot(x, fr, marker=mk, color=c, ls=['-', '--', ':'][i], label=f'$C_{{{nm}}}$')
    ax.axhline(0, color='0.4', lw=1.0)
    ax.set_xticks(x, [a.replace('_', '\n') for a in arms], fontsize=7.5)
    ax.set_ylabel(r'instrumental fraction of raw $\langle h_{\rm pred,-}h_{\rm pred,+}\rangle$')
    ax.set_title('More than half of the raw transverse correlation of the predicted\n'
                 'polarimeters is spin-independent; better geometry reduces it', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3b_instrumental_fraction.png')
    plt.close(fig)


# --------------------------------------------------------------- figure 4
def fig4_ladder(R):
    """What reconstruction costs, in events needed to certify entanglement."""
    rows = [('exact $h$', R['exact_h']),
            ('+ ideal-IP oracle', R['arms']['idealip22_s42']),
            ('+ 3 IP + SV (s42)', R['arms']['full22_s42']),
            ('+ 3 IP + SV (s43)', R['arms']['full22_s43']),
            ('reco, no geometry', R['arms']['base_s43'])]
    N_SHOW = 300
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
    yv = np.arange(len(rows))[::-1]
    w = [r[1]['witness'] for r in rows]
    s = [r[1]['witness_sd'] * np.sqrt(r[1]['n'] / N_SHOW) for r in rows]
    ax[0].errorbar(w, yv, xerr=s, fmt='o', color='C0', capsize=3, ms=5)
    ax[0].axvline(0, color='C3', lw=1.4, ls='--')
    ax[0].axvline(-0.5, color='k', lw=1.2, ls=':')
    ax[0].text(0.012, len(rows) - 1.8, 'separable', color='C3', fontsize=8, rotation=90,
               va='center')
    ax[0].text(-0.495, len(rows) - 1.25, r'physical bound $-1/2$', color='k', fontsize=7.5,
               rotation=90, va='center')
    ax[0].set_yticks(yv, [r[0] for r in rows], fontsize=8)
    ax[0].set_xlabel(r'$\langle W\rangle \pm 1\sigma$ of the $\rho_H$ witness, '
                     f'for {N_SHOW} signal events')
    ax[0].set_title('Expected witness, scaled to a common 300 events\n'
                    '(central value is the generated state by construction)', fontsize=9)
    ax[0].set_xlim(-0.72, 0.12)
    n3 = [r[1]['reach_3sigma'] for r in rows]
    n5 = [r[1]['reach_5sigma'] for r in rows]
    ax[1].plot(n3, yv, 'o-', color='C0', ms=5, label=r'$3\sigma$')
    ax[1].plot(n5, yv, 's--', color='C1', ms=5, label=r'$5\sigma$')
    for a, b in zip(n3, yv):
        ax[1].annotate(f'{a:,.0f}', (a, b), textcoords='offset points', xytext=(0, 7),
                       fontsize=7.5, ha='center')
    ax[1].set_yticks(yv, [r[0] for r in rows], fontsize=8)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('signal H events needed to certify entanglement')
    ax[1].set_title('Signal-only, no background, no systematics', fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_reach_ladder.png')
    plt.close(fig)


# --------------------------------------------------------------- figure 5
def fig5_modepairs(R):
    mp = R['mode_pairs_full22_s42']
    names = list(mp.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, c, mk, ls, lab in (('exact', 'C0', 'o', '-', r'exact $h$'),
                                ('reco', 'C1', 's', '--', r'reco $h_{\rm pred}$ + IP/SV')):
        v = [mp[n][key]['reach_3sigma'] for n in names]
        ax[0].plot(x, v, marker=mk, ls=ls, color=c, ms=6, label=lab)
        for xi, vi in zip(x, v):
            ax[0].annotate(f'{vi:.0f}', (xi, vi), textcoords='offset points',
                           xytext=(0, 6), ha='center', fontsize=7)
    ax[0].set_yscale('log')
    ax[0].set_xticks(x, [n.replace(' x ', r'$\times$') for n in names], fontsize=7.5,
                     rotation=20)
    ax[0].set_ylabel(r'signal events for a $3\sigma$ witness')
    ax[0].set_title('Events needed, by decay-mode pair', fontsize=9)
    ax[0].legend(fontsize=8)
    ratio = [mp[n]['reco']['reach_3sigma'] / mp[n]['exact']['reach_3sigma']
             if mp[n]['exact']['reach_3sigma'] and mp[n]['reco']['reach_3sigma'] else np.nan
             for n in names]
    ax[1].bar(x, ratio, color='C4', width=0.6)
    for xi, r, n in zip(x, ratio, names):
        ax[1].annotate(f'{r:.1f}', (xi, r), textcoords='offset points', xytext=(0, 3),
                       ha='center', fontsize=8)
    ax[1].set_xticks(x, [n.replace(' x ', r'$\times$') for n in names], fontsize=7.5,
                     rotation=20)
    ax[1].set_ylabel(r'events needed, reco / exact $h$')
    ax[1].set_title('Reconstruction cost per mode pair', fontsize=9)
    fig.suptitle('H validation events, reco arm full22_s42 (point-h + 3 IP + SV);  '
                 r'$n$ per pair: ' + ',  '.join(f"{n.replace(' x ', '/')} {mp[n]['n']:,}"
                                                for n in names), fontsize=8.2, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / 'fig5_mode_pairs.png')
    plt.close(fig)


def fig6_alignment(S):
    """The instrumental alignment, measured without any reweighting."""
    y = S['labels']
    O = json.loads((HERE / 'results' / 'origin.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    arms = [('exact $h$', S['h_gen'][~y], 'k', '-', 'o'),
            ('reco, no geometry', load_arm('base_s43', S)['h_pred'][~y], 'C3', '--', 's'),
            ('reco + 3 IP + SV', load_arm('full22_s42', S)['h_pred'][~y], 'C1', '-.', '^'),
            ('reco + ideal IP', load_arm('idealip22_s42', S)['h_pred'][~y], 'C0', ':', 'v')]
    vals = []
    for lab, v, c, ls, mk in arms:
        tm, tp = v[:, 0, :2], v[:, 1, :2]
        dphi = np.mod(np.arctan2(tm[:, 1], tm[:, 0]) - np.arctan2(tp[:, 1], tp[:, 0]),
                      2 * np.pi)
        dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
        ax[0].hist(dphi, bins=30, range=(0, np.pi), density=True, histtype='step',
                   color=c, ls=ls, lw=1.7, label=lab)
        um = tm / np.maximum(np.linalg.norm(tm, axis=1, keepdims=True), 1e-12)
        up = tp / np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-12)
        cd = np.sum(um * up, 1)
        vals.append((lab, cd.mean(), cd.std() / np.sqrt(len(cd)), c, mk))
    ax[0].axhline(1 / np.pi, color='0.4', lw=1.1, ls='-')
    ax[0].text(0.05, 1 / np.pi - 0.028, r'no alignment ($1/\pi$)', fontsize=7.5,
               color='0.35')
    ax[0].set_xlabel(r'$|\Delta\phi|$ between $h_{-\perp}$ and $h_{+\perp}$  [rad]')
    ax[0].set_ylabel('normalised')
    ax[0].set_title(r'$Z\to\tau\tau$, where $C_{nn}=C_{rr}=0$ by construction'
                    '\n(29,740 events, no reweighting)', fontsize=9)
    ax[0].legend(fontsize=7.5, loc='upper right')
    x = np.arange(len(vals))
    for i, (lab, m, e, c, mk) in enumerate(vals):
        ax[1].errorbar([i], [m], yerr=[e], fmt=mk, color=c, ms=7, capsize=4)
        ax[1].annotate(f'{m:+.3f}', (i, m), textcoords='offset points', xytext=(0, 10),
                       ha='center', fontsize=8)
    ax[1].axhline(0, color='0.3', lw=1.2)
    ax[1].set_xticks(x, [v[0].replace(' + ', '\n+ ').replace(', ', ',\n') for v in vals],
                     fontsize=7.5)
    ax[1].set_ylim(-0.04, 0.31)
    ax[1].set_ylabel(r'$\langle\cos\Delta\phi\rangle$')
    ax[1].set_title('Spin-independent azimuthal alignment of the two\n'
                    'predicted polarimeters; geometry halves it', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_instrumental_alignment.png')
    plt.close(fig)


def main():
    S, R, U = load()
    fig1_states(S)
    fig2_p0(S)
    fig3_instrumental(S, R)
    fig4_ladder(R)
    fig5_modepairs(R)
    fig6_alignment(S)
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
