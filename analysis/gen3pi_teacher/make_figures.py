"""Figures: CLEO vs generator 3pi teacher for reco point-h readouts and the null-space ceiling."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
PAIRS = ['pi x pi', 'rho x rho', '3pi x 3pi', 'pi x rho', 'pi x 3pi', 'rho x 3pi']
PLAB = {'pi x pi': 'π×π', 'rho x rho': 'ρ×ρ', '3pi x 3pi': '3π×3π',
        'pi x rho': 'π×ρ', 'pi x 3pi': 'π×3π', 'rho x 3pi': 'ρ×3π'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})

RECO = [('base_s43', 'current reco\n(no IP/SV), s43'), ('full22_s42', '+IP/SV\ns42'),
        ('full22_s43', '+IP/SV\ns43'), ('ens_full22', '+IP/SV\n2-seed mean'),
        ('idealip22_s42', 'ideal IP\ndirection, s42')]
CEIL = [('flat', 'vis+MET\n+mass shells'), ('ip3', '+3 IPs'), ('sv', '+SV'),
        ('ip3_sv', '+3 IPs\n+SV'), ('ip3_sv_mass', '+3 IPs+SV\n+parent mass'), ('truth_h', 'exact h')]


def fig_ladder(P, N, out):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [6.5, 6]})
    ax = axs[0]
    x = np.arange(len(RECO))
    for tag, col, mk, dx, name in (('old', ORANGE, 'o', -0.12, 'CLEO 3π teacher (previous)'),
                                   ('new', BLUE, 's', 0.12, 'generator 3π teacher')):
        v = [P['auc'][f'{tag}:{a}']['overall'] for a, _ in RECO]
        ax.plot(x + dx, v, mk, color=col, ms=8, label=name)
    for i, (a, _) in enumerate(RECO):
        d = P['delta_new_minus_old'][a]['overall']
        ax.text(i, max(P['auc'][f'new:{a}']['overall'], P['auc'][f'old:{a}']['overall']) + 0.003,
                f'{d["delta"]:+.4f}\n[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]', ha='center', fontsize=7, color=INK)
    ax.set_xticks(x, [l for _, l in RECO], fontsize=8.5)
    ax.set_ylabel('H vs Z AUC (fixed readout, parent-overlap weighted)')
    ax.set_ylim(0.61, 0.665)
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    ax.legend(loc='upper left', fontsize=8.5)
    ax.set_title('(a) reco point-h regression → fixed readout', fontsize=10, loc='left')
    ax = axs[1]
    x = np.arange(len(CEIL))
    for tag, col, mk, dx, name in (('cleo', ORANGE, 'o', -0.12, 'CLEO 3π current'),
                                   ('gen', BLUE, 's', 0.12, 'generator 3π current')):
        v = [N['auc'][f'{tag}:{m}']['overall'] for m, _ in CEIL]
        ax.plot(x + dx, v, mk, color=col, ms=8, label=name)
    for i, (m, _) in enumerate(CEIL):
        d = N['delta_gen_minus_cleo'][m]['overall']
        ax.text(i, N['auc'][f'gen:{m}']['overall'] + 0.004, f'{d["delta"]:+.4f}', ha='center', fontsize=7.5, color=INK)
    ax.set_xticks(x, [l for _, l in CEIL], fontsize=8.5)
    ax.set_ylabel('H vs Z AUC (solution-set likelihood ratio, unweighted)')
    ax.set_ylim(0.6, 0.745)
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    ax.legend(loc='upper left', fontsize=8.5)
    ax.set_title('(b) ceiling on the solution set (truth visible + exact MET)', fontsize=10, loc='left')
    fig.suptitle('Changing only the 3π polarimeter (teacher / likelihood) from the CLEO to the generator current; '
                 'validation 59,390 events; numbers: new − old [paired 95% CI]', fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'figure_1_old_vs_new_teacher.png', dpi=170)
    plt.close(fig)


def fig_pairs(P, N, out):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(PAIRS))
    series = [('exact h (likelihood ratio)', P['delta_new_minus_old']['exact_h_LR'], GRAY, 'D', -0.24),
              ('ceiling: +3 IPs+SV+parent mass', N['delta_gen_minus_cleo']['ip3_sv_mass'], VIOLET, '^', -0.08),
              ('reco +IP/SV, 2-seed mean', P['delta_new_minus_old']['ens_full22'], BLUE, 's', 0.08),
              ('reco current (no IP/SV), s43', P['delta_new_minus_old']['base_s43'], AQUA, 'o', 0.24)]
    for name, d, col, mk, dx in series:
        v = np.array([d[p]['delta'] for p in PAIRS])
        lo = v - np.array([d[p]['ci95'][0] for p in PAIRS])
        hi = np.array([d[p]['ci95'][1] for p in PAIRS]) - v
        ax.errorbar(x + dx, v, yerr=[lo, hi], fmt=mk, color=col, ms=7, elinewidth=1.2, capsize=0, label=name)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [PLAB[p] for p in PAIRS])
    ax.set_xlabel('truth decay-mode pair')
    ax.set_ylabel('AUC(generator 3π) − AUC(CLEO 3π)')
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('Where the change acts: per mode pair, paired event bootstrap 95% CI\n'
                 '(reco and exact h: parent-overlap weighted; ceiling: unweighted)', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'figure_2_delta_by_mode_pair.png', dpi=170)
    plt.close(fig)


def fig_hquality(P, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    comps = ['h_n', 'h_r', 'h_k']
    for ax, arm, title in ((axs[0], 'base_s43', '(a) current reco (no IP/SV), s43'),
                           (axs[1], 'full22_s42', '(b) +IP/SV, s42')):
        for j, (mode, col) in enumerate((('pi', ORANGE), ('rho', BLUE), ('3pi', VIOLET))):
            for tag, fill, dx in (('old', False, -0.08), ('new', True, 0.08)):
                q = P['h_quality'].get(f'{tag}:{arm}', {}).get(mode)
                if q is None:
                    continue
                xs = np.arange(3) + (j - 1) * 0.28 + dx
                ax.plot(xs, q['corr_vs_gen_nrk'], 's' if tag == 'new' else 'o', color=col, ms=7,
                        mfc=col if fill else 'white', mew=1.5,
                        label=f'{mode}, {"generator" if tag == "new" else "CLEO"} teacher')
        ax.set_xticks(range(3), comps)
        ax.set_title(title, fontsize=10, loc='left')
        ax.grid(axis='y', color='#e4e3df', lw=0.8)
    axs[0].set_ylabel('corr(predicted h, generator h)  per τ side')
    axs[1].legend(fontsize=7.5, loc='lower left', ncol=3)
    fig.suptitle('How well the reco regression recovers the sample\'s true polarimeter; validation, both τ sides, unit weight',
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'figure_3_h_quality.png', dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--point-h', type=Path, required=True)
    ap.add_argument('--nullspace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    P = json.loads(a.point_h.read_text())
    N = json.loads(a.nullspace.read_text())
    fig_ladder(P, N, a.output)
    fig_pairs(P, N, a.output)
    fig_hquality(P, a.output)


if __name__ == '__main__':
    main()
