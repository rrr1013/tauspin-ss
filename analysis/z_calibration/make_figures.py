"""Figures of the Z-calibration study (results/*.json -> results/fig*.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parent / 'results'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})


def fig_current():
    d = json.load(open(R / 'transfer_a1.json'))['scopes']
    est = [('exact h_gen', 'exact $h$ (generator current)'),
           ('reco base_s43 / gen teacher', 'reco, no IP/SV'),
           ('reco full22_s42 / gen teacher', 'reco +IP/SV (seed 42)'),
           ('reco full22_s43 / gen teacher', 'reco +IP/SV (seed 43)'),
           ('reco idealip22_s42 / gen teacher', 'reco, ideal IP')]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for ax, sc in zip(axs, ('any 3pi', '3pi x 3pi')):
        x = np.arange(len(est))
        for j, (key, lab, mk, col, off) in enumerate((
                ('single_T', 'uncalibrated: $r_T$(CLEO) / $r_T$(gen)', 'o', C[1], -0.12),
                ('double_T_over_Q', 'after Z calibration: $\\rho_T / \\rho_Q$', 's', C[0], 0.0),
                ('double_A_over_Q', 'CP-odd after Z calibration: $\\rho_A / \\rho_Q$', 'D', C[2], 0.12))):
            v = [d[sc][e]['single_ratio']['T'] if key == 'single_T' else d[sc][e][key] for e, _ in est]
            ci = np.array([d[sc][e]['ci95'][key] for e, _ in est])
            ax.errorbar(x + off, v, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], fmt=mk, ms=6, color=col,
                        mfc=col if j != 0 else 'white', capsize=2, lw=1.2, label=lab)
        ax.axhline(1, color='#555', lw=0.8, ls='--')
        ax.set_xticks(x, [l for _, l in est], rotation=25, ha='right')
        ax.set_title(f"{sc.replace('3pi', '3π').replace(' x ', '×')} events ({d[sc]['events']:,})", fontsize=10)
    axs[0].set_ylabel('response ratio\nCLEO-current world / generator world')
    axs[0].legend(loc='lower left', fontsize=8.5, frameon=False)
    fig.suptitle('3π hadronic-current change: transverse spin response before and after a Z (m = 2) calibration\n'
                 'generator-teacher networks, ATLAS full-reco validation, flat-spin weights, paired bootstrap 95% CI', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(R / 'fig1_current_transfer.png', dpi=150)


def fig_requirement():
    k = json.load(open(R / 'kt_requirement.json'))['fits']
    z = json.load(open(R / 'z_state.json'))
    s = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    xs = [0.02, 0.1, 0.3, 0.5, 1.0, 2.0]  # 0 drawn at 0.02 on the log axis
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for lev, lab, col in (('tauspin', 'tauspin $\\hat h$', C[0]), ('phicp', 'LHC CP observables ($\\varphi^*_{CP}$ set)', C[1])):
        for sfx, ls, mk, bl in (('', '-', 'o', 'side-bands only'), ('_tau1', '--', 's', 'aux. control τ = 1')):
            v = [k[lev][f'sT{x:g}{sfx}']['Z'] for x in s]
            ax.plot(xs, v, ls=ls, marker=mk, color=col, lw=1.6, ms=6, label=f'{lab}, {bl}')
            ax.plot([4.0], [k[lev][f'bound_T{sfx}']['Z']], marker=mk, color=col, mfc='white', ms=7, ls='none')
            ax.plot([8.0], [k[lev][f'kT_free{sfx}']['Z']], marker=mk, color=col, mfc='white', ms=7, ls='none')
    N = 1090861 * 0.718
    for key, lab, xx in (('reco_recoframe', 'Z calib. stat. 3 ab$^{-1}$', None),):
        sig = z['per_event'][key]['all']['sigma_kappa_sqrtN']
        for L, a in ((3000, 0.25), (140, 0.12)):
            v = sig / np.sqrt(N * L / 3000)
            ax.axvspan(0.015, v, color=C[2], alpha=a, lw=0)
            ax.text(v * 1.06, 0.65 if L == 140 else 0.3, f'Z calib. (stat.) {L:g} fb$^{{-1}}$: {100 * v:.1f}%',
                    fontsize=8, color='#1b6e4f', va='center')
    ax.set_xscale('log')
    ax.set_xticks([0.02, 0.1, 0.3, 0.5, 1, 2, 4, 8], ['0', '0.1', '0.3', '0.5', '1', '2', 'phys.\nbound', 'free'])
    ax.axvline(3, color='#999', lw=0.6)
    ax.set_xlabel('width of the prior on the transverse response $k_T$ (relative)')
    ax.set_ylabel('expected entanglement significance [σ]')
    ax.set_ylim(0, 3.4)
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title('H→τ$_h$τ$_h$ entanglement vs knowledge of $k_T$ (3 ab$^{-1}$, 14 TeV, one experiment, Asimov)\n'
                 '$k_L$ prior 10%; open markers: only the Cauchy–Schwarz bound $k_T\\leq4.1$ (tauspin), or $k_T$ free',
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(R / 'fig2_kt_requirement.png', dpi=150)


def fig_kin():
    d = json.load(open(R / 'transfer_kin.json'))
    rows = [k for k in d if k.startswith('mode') or k.startswith('E_min') or k == 'all']
    lab = {'mode 0x0': 'π×π', 'mode 0x1': 'π×ρ', 'mode 1x1': 'ρ×ρ', 'mode 0x3': 'π×3π', 'mode 1x3': 'ρ×3π',
           'mode 3x3': '3π×3π', 'all': 'all'}
    names = [lab[r] if r in lab else 'softer-τ E ' + r.split()[1].upper() + '\n' + r.split('[')[1].split(']')[0].replace(',', '–') + ' GeV'
             for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    x = np.arange(len(rows))
    for j, (n, nl, col, off) in enumerate((('reco full22_s42', 'reco +IP/SV', C[0], -0.1), ('reco base_s43', 'reco, no IP/SV', C[1], 0.1))):
        for key, mk, mf in (('T_over_Q', 'o', None), ('A_over_T', 'D', 'white')):
            v = np.array([d[r][n][key] for r in rows])
            ci = np.array([d[r][n]['ci68'][key] for r in rows])
            ax.errorbar(x + off + (0.04 if key == 'A_over_T' else 0), v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt=mk, color=col,
                        mfc=mf or col, ms=6, capsize=2, label=f"{nl}: {'$r_T/r_Q$ (H needs / Z gives)' if key == 'T_over_Q' else '$r_A/r_T$ (CP-odd / CP-even)'}")
    ax.axhline(1, color='#555', lw=0.8, ls='--')
    ax.set_xticks(x, [f"{nm}\n{d[r]['events']:,}" for nm, r in zip(names, rows)], fontsize=8)
    ax.set_ylabel('ratio of transverse responses (MC)')
    ax.set_ylim(0.8, 1.2)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper left')
    ax.set_title('Structural transfer m = 2 (Z) → m = 0 (H) in the nominal MC, by truth decay-mode pair and softer-τ energy\n'
                 'ATLAS full-reco validation, flat-spin weights, 68% bootstrap CI', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(R / 'fig3_mc_transfer.png', dpi=150)


def fig_calib():
    f = R / 'transfer_calib.json'
    if not f.exists():
        return
    d = json.load(open(f))
    arms = [('local', 'reco +IP/SV (local)'), ('none', 'reco, no IP/SV')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (arm, al) in zip(axs, arms):
        vs = list(d[arm])
        x = np.arange(len(vs))
        for key, lab, mk, col, off, mf in (('single_T', 'uncalibrated $r_T$(shift)/$r_T$(nominal)', 'o', C[1], -0.12, 'white'),
                                           ('double_T_over_Q', 'after Z calibration $\\rho_T/\\rho_Q$', 's', C[0], 0.0, None),
                                           ('single_kk', 'longitudinal $r_{kk}$ (Z calibrates directly)', '^', C[3], 0.12, 'white')):
            v = np.array([d[arm][k]['all'][key]['value'] for k in vs])
            ci = np.array([d[arm][k]['all'][key]['ci95'] for k in vs])
            ax.errorbar(x + off, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt=mk, color=col, mfc=mf or col, ms=5.5, capsize=2, label=lab)
        ax.axhline(1, color='#555', lw=0.8, ls='--')
        ax.set_xticks(x, [k.replace(' (consistent)', '').replace(' (fixed)', ' MET-fixed').replace('trk', 'track') for k in vs],
                      rotation=40, ha='right', fontsize=8)
        ax.set_title(al)
    axs[0].set_ylabel('response ratio, shifted / nominal input')
    axs[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Detector-calibration shifts of the fixed point-h network: transverse response before / after Z calibration\n'
                 'ATLAS full-reco validation (59,390 events), flat-spin weights, paired bootstrap 95% CI', fontsize=10)
    fig.tight_layout()
    fig.savefig(R / 'fig4_calib_transfer.png', dpi=150)


if __name__ == '__main__':
    fig_current()
    fig_requirement()
    fig_kin()
    fig_calib()
