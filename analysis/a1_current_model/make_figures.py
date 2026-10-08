"""Figures 2 and 3 from results/analysis.json and results/responses.json."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import reweight as rw

R = rw.HERE / 'results'
A = json.loads((R / 'analysis.json').read_text())
S = json.loads((R / 'responses.json').read_text())
I = json.loads((R / 'interaction.json').read_text())
DC = json.loads((R / 'decomposition.json').read_text())
GEN, CLEO, GREY = '#0072B2', '#D55E00', '0.35'


def crossover(ax, gen_key, cleo_key, title, ikey, pair='any 3pi'):
    auc = A['auc']
    it = I[ikey][pair]
    title = title + f"\nteacher×world interaction {it['interaction']:+.3f} [{it['ci95'][0]:+.3f}, {it['ci95'][1]:+.3f}]"
    for key, col, mk, ls, lab in ((gen_key, GEN, 'o', '-', 'generator-current $h$'),
                                  (cleo_key, CLEO, 's', '--', 'CLEO-current $h$')):
        v = [auc[key][pair]['gen'], auc[key][pair]['cleo']]
        ax.plot([0, 1], v, marker=mk, ls=ls, color=col, ms=8, lw=2, label=lab)
        for x, val in zip((0, 1), v):
            ax.annotate(f'{val:.4f}', (x, val), textcoords='offset points', xytext=(-38 if x == 0 else 8, -3), fontsize=8, color=col)
    ax.set_xticks([0, 1], ['generator world\n(as simulated)', 'CLEO world\n(reweighted)'])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(f'H/Z AUC, events with a 3$\\pi$ side (own y-range)')
    ax.set_title(title, fontsize=9.5, loc='left')


def main():
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.6]})
    crossover(ax[0], 'exact LR h_gen', 'exact LR h_CLEO', '(a) exact $h$ (truth $\\nu$), fixed-$C$ LR', 'exact')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, ['polarimeter / teacher: ' + l[0], 'polarimeter / teacher: ' + l[1]], fontsize=9, frameon=False,
               loc='lower left', ncol=2, bbox_to_anchor=(0.04, 0.0))
    crossover(ax[1], 'gen:full22_s42', 'cleo:full22_s42', '(b) reco point-$h$ + IP/SV (seed 42)', 'full22_s42')
    crossover(ax[2], 'gen:base_s43', 'cleo:base_s43', '(c) reco point-$h$, no geometry (seed 43)', 'base_s43')
    # (d) response ratios
    a = ax[3]
    rows = [('exact h_gen', 'exact $h_{\\rm gen}$', GEN, 'o'),
            ('reco base_s43 / gen teacher', 'reco, no geom. / gen teacher', GEN, 'v'),
            ('reco full22_s42 / gen teacher', 'reco +IP/SV / gen teacher', GEN, 'D'),
            ('reco idealip22_s42 / gen teacher', 'reco ideal IP / gen teacher', GEN, '^'),
            ('exact h_CLEO', 'exact $h_{\\rm CLEO}$', CLEO, 'o'),
            ('reco base_s43 / cleo teacher', 'reco, no geom. / CLEO teacher', CLEO, 'v'),
            ('reco full22_s42 / cleo teacher', 'reco +IP/SV / CLEO teacher', CLEO, 'D'),
            ('reco idealip22_s42 / cleo teacher', 'reco ideal IP / CLEO teacher', CLEO, '^')]
    qs = [('s_k (P_tau)', r'$s_k$ ($P_\tau$)'), ('R_kk', r'$R_{kk}$'), ('R_perp', r'$R_\perp$')]
    for j, (q, ql) in enumerate(qs):
        for i, (key, lab, col, mk) in enumerate(rows):
            v = S['responses']['any 3pi'][key][q]
            y = i + (j - 1) * 0.22
            a.errorbar(v['ratio'], y, xerr=[[v['ratio'] - v['ratio_ci95'][0]], [v['ratio_ci95'][1] - v['ratio']]],
                       fmt=mk, color=col, mfc=[col, 'white', '0.8'][j], ms=6, capsize=2, lw=1,
                       label=ql if i == 0 else None)
    a.axvline(1, color='0.5', lw=0.8)
    a.axvline(A['analyzing_power']['alpha_all'], color=GEN, lw=0.8, ls=':')
    a.axvline(1 / A['analyzing_power']['alpha_all'], color=CLEO, lw=0.8, ls=':')
    a.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel('spin response, CLEO world / generator world (95% CI)')
    a.set_title(r'(d) spin response transfer (3$\pi$ events; dotted: $\alpha$, $1/\alpha$)', fontsize=10, loc='left')
    leg = a.legend(fontsize=8, frameon=False, loc='lower right', title='filled / open / grey', title_fontsize=7)
    for h in leg.legend_handles:
        h.set_color('0.2')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(R / 'fig2_crossover_and_response.png', dpi=140)

    # figure 3: m3pi dependence
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ap = A['analyzing_power']
    c = np.array(ap['centres'])
    al = np.array([np.nan if v is None else v for v in ap['alpha']])
    fr = np.array(ap['weight_fraction'])
    a = ax[0]
    ok = fr > 0.003
    a.plot(c[ok], al[ok], 'o-', color=GEN, lw=1.8, ms=6, label=r'$\langle h_{\rm gen}\cdot h_{\rm CLEO}\rangle$ (CLEO world)')
    for x, yv, f in zip(c[ok], al[ok], fr[ok]):
        a.annotate(f'{100 * f:.0f}%' if f >= 0.01 else '<1%', (x, yv), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color='0.3')
    a.axhline(1, color='0.6', lw=0.7)
    a.set_xlabel(r'$m_{3\pi}$ [GeV]'); a.set_ylabel('effective analyzing power of $h_{\\rm gen}$')
    a.set_title(r'(a) where the two currents disagree (labels: share of 3$\pi$ sides)', fontsize=10, loc='left')
    a.legend(fontsize=8, frameon=False, loc='lower left')
    a = ax[1]
    mb = S['auc_world_effect_vs_m3pi']
    edges = S['m3pi_edges']
    cc = np.arange(len(edges) - 1, dtype=float)
    series = [('exact LR h_gen', 'exact $h_{\\rm gen}$', GEN, 'o', '-'), ('gen:base_s43', 'reco, no geom. / gen teacher', GEN, 'v', '--'),
              ('gen:full22_s42', 'reco +IP/SV / gen teacher', GEN, 'D', ':'), ('exact LR h_CLEO', 'exact $h_{\\rm CLEO}$', CLEO, 'o', '-'),
              ('cleo:full22_s42', 'reco +IP/SV / CLEO teacher', CLEO, 'D', ':'),
              ('cleo:base_s43', 'reco, no geom. / CLEO teacher', CLEO, 'v', '--')]
    for k, (key, lab, col, mk, ls) in enumerate(series):
        v = np.array([mb[b][key]['delta'] for b in mb])
        lo = np.array([mb[b][key]['ci68'][0] for b in mb])
        hi = np.array([mb[b][key]['ci68'][1] for b in mb])
        x = cc + (k - 2.5) * 0.06
        a.errorbar(x, v, [v - lo, hi - v], fmt=mk, ls=ls, color=col, ms=6, lw=1.4, capsize=2,
                   mfc=col if 'exact' in key else 'white', label=lab)
    a.axhline(0, color='0.5', lw=0.8)
    a.set_xticks(cc, [f'{edges[i]:.2f}–{edges[i + 1]:.2f}\n({mb[b]["events"]})' for i, b in enumerate(mb)], fontsize=8)
    a.set_xlabel(r'$m_{3\pi}$ [GeV] (events with exactly one 3$\pi$ side)')
    a.set_ylabel('AUC(CLEO world) − AUC(generator world)')
    a.set_title('(b) world effect on H/Z AUC vs $m_{3\\pi}$ (68% CI)', fontsize=10, loc='left')
    a.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(R / 'fig3_m3pi_dependence.png', dpi=140)

    # figure 5: decomposition of the world effect (review follow-up)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    rows = [('exact LR h_gen', 'exact $h_{\\rm gen}$', GEN), ('gen:base_s43', 'reco, no geom. / gen teacher', GEN),
            ('gen:full22_s42', 'reco +IP/SV / gen teacher', GEN), ('cleo:base_s43', 'reco, no geom. / CLEO teacher', CLEO),
            ('cleo:full22_s42', 'reco +IP/SV / CLEO teacher', CLEO)]
    comps = [('u_only', 'Dalitz part only ($u$)', 'o', '0.8'), ('spin_only', 'spin part only ($f_{\\rm CLEO}/f_{\\rm gen}$)', 's', 'white'),
             ('full', 'full CLEO world', 'D', None)]
    for a, pair in zip(ax, ('any 3pi', '3pi x 3pi')):
        for i, (key, lab, col) in enumerate(rows):
            for j, (cn, cl, mk, fc) in enumerate(comps):
                v = DC['auc_delta_vs_gen'][key][pair][cn]
                yv = i + (j - 1) * 0.22
                a.errorbar(v['delta'], yv, xerr=[[v['delta'] - v['ci95'][0]], [v['ci95'][1] - v['delta']]], fmt=mk,
                           color=col, mfc=fc if fc else col, ms=6, capsize=2, lw=1, label=cl if i == 0 else None)
        a.axvline(0, color='0.5', lw=0.8)
        a.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
        a.invert_yaxis()
        a.set_xlabel('AUC(world) − AUC(generator world), 95% CI')
        a.set_title(('(a) events with a 3$\\pi$ side' if pair == 'any 3pi' else '(b) 3$\\pi$ × 3$\\pi$ events'), fontsize=10, loc='left')
        leg = a.legend(fontsize=8, frameon=False, loc='lower left')
        for h in leg.legend_handles:
            h.set_color('0.2')
    fig.tight_layout()
    fig.savefig(R / 'fig5_decomposition.png', dpi=140)


if __name__ == '__main__':
    main()
