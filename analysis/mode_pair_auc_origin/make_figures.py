"""Figures for the mode-pair exact-h AUC study (reads run_mode_pair_auc.py outputs)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7'
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
PAIRS = ['pi x pi', 'rho x rho', '3pi x 3pi', 'pi x rho', 'pi x 3pi', 'rho x 3pi']
LABEL = {'pi x pi': 'π×π', 'rho x rho': 'ρ×ρ', '3pi x 3pi': '3π×3π',
         'pi x rho': 'π×ρ', 'pi x 3pi': 'π×3π', 'rho x 3pi': 'ρ×3π'}
MODES = {0: 'π', 1: 'ρ', 3: '3π'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def mdot(a, b):
    return a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], -1)


def fig_ladder(r, out):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(PAIRS))
    ideal = r['theory']['ideal_auc']
    ax.axhspan(ideal - 0.0015, ideal + 0.0015, color=GRAY, alpha=0.18, lw=0)
    ax.axhline(ideal, color=GRAY, lw=1, ls='--')
    ax.text(len(PAIRS) - 0.45, ideal + 0.002, f'ideal unit polarimeters, no selection: {ideal:.3f}',
            ha='right', va='bottom', color=MUTED, fontsize=8.5)
    series = [
        ('data, canonical h (3π: CLEO current)', r['observed']['CLEO_hRef'], ORANGE, 'o', -0.24, True),
        ('data, h from the generator current', r['observed']['generator_current'], BLUE, 's', -0.08, True),
        ('re-decay toy, no selection', {k: r['toy'][k]['no_selection'] for k in PAIRS}, GRAY, 'D', 0.08, False),
        ('re-decay toy, vis. pT>20 GeV, |η|<2.5', {k: r['toy'][k]['with_selection'] for k in PAIRS}, AQUA, '^', 0.24, True),
    ]
    for name, rows, col, mk, dx, fill in series:
        yv = [rows[k]['auc'] for k in PAIRS]
        ev = [rows[k]['se'] for k in PAIRS]
        ax.errorbar(x + dx, yv, yerr=ev, fmt=mk, ms=7, color=col, mfc=col if fill else 'white',
                    mew=1.5, elinewidth=1.2, capsize=0, label=name, zorder=3)
    ax.set_xticks(x, [LABEL[k] for k in PAIRS])
    ax.set_ylabel('H vs Z AUC  (fixed-C likelihood ratio on h)')
    ax.set_xlabel('truth decay-mode pair')
    ax.set_ylim(0.665, 0.755)
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    ax.legend(loc='lower left', fontsize=8.5, ncol=2)
    ax.set_title('Exact-h AUC by mode pair: data vs ideal-correlation re-decay toy\n'
                 'canonical validation cohort, 59,390 events, unit weight, ±1 s.e. (bootstrap)',
                 fontsize=10, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(out / 'figure_1_auc_by_mode_pair.png', dpi=170)
    plt.close(fig)


def fig_threepi(D, A, r, out):
    h_ref = D['h_ref'].astype(float)
    h_gen = A['h_gen'].astype(float)
    m = D['modes']
    cos, m3 = [], []
    for s in (0, 1):
        k = m[:, s] == 3
        cos.append(np.sum(h_ref[k, s] * h_gen[k, s], -1))
        v = D['pions'][k, s].sum(1)
        m3.append(np.sqrt(np.maximum(mdot(v, v), 0)))
    cos, m3 = np.concatenate(cos), np.concatenate(m3)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    ax = axs[0]
    ax.hist(cos, bins=np.linspace(-1, 1, 81), color=VIOLET, histtype='stepfilled', alpha=0.85)
    ax.set_yscale('log')
    ax.axvline(cos.mean(), color=INK, lw=1, ls='--')
    ax.text(cos.mean() - 0.03, ax.get_ylim()[1] * 0.3, f'mean {cos.mean():.3f}\nmedian {np.median(cos):.3f}',
            ha='right', va='top', fontsize=8.5, color=INK)
    ax.set_xlabel('cos∠(h_CLEO, h_generator)   per 3π side')
    ax.set_ylabel('τ sides / 0.025')
    ax.set_title('(a) the two 3π polarimeters disagree', fontsize=10, loc='left')
    ax = axs[1]
    edges = np.linspace(0.45, 1.75, 14)
    c = 0.5 * (edges[1:] + edges[:-1])
    cnt = np.histogram(m3, edges)[0]
    mean = [cos[(m3 >= a) & (m3 < b)].mean() if n >= 20 else np.nan
            for a, b, n in zip(edges[:-1], edges[1:], cnt)]
    ax.plot(c, mean, 'o-', color=VIOLET, ms=6, lw=2)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(1, color=GRAY, lw=0.8, ls=':')
    ax.set_xlabel('m(3π) [GeV]')
    ax.set_ylabel('⟨cos∠(h_CLEO, h_generator)⟩')
    ax2 = ax.twinx()
    ax2.bar(c, cnt, width=np.diff(edges) * 0.9, color=GRAY, alpha=0.18, zorder=0)
    ax2.set_yticks([])
    ax2.spines['right'].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title('(b) agreement vs 3π mass (grey: 3π sides; bins ≥20)', fontsize=10, loc='left')
    ax = axs[2]
    comps = ['C_nn', 'C_rr', 'C_kk']
    ideal_H, ideal_Z = [1, 1, -1], [0, 0, 1]
    for j, (tag, col, mk, lab) in enumerate((('CLEO_hRef', ORANGE, 'o', 'CLEO h'),
                                            ('generator_current', BLUE, 's', 'generator h'))):
        row = r['observed'][tag]['3pi x 3pi']
        for cls, off, fill in (('C_H', -0.12, True), ('C_Z', 0.12, False)):
            vals = np.diag(np.array(row[cls]))
            ax.plot(np.arange(3) + off + (j - 0.5) * 0.08, vals, mk, color=col, ms=7,
                    mfc=col if fill else 'white', mew=1.5,
                    label=f'{lab}, {"H" if cls == "C_H" else "Z"}')
    for i in range(3):
        ax.plot([i - 0.25, i + 0.01], [ideal_H[i]] * 2, color=INK, lw=1.2)
        ax.plot([i - 0.01, i + 0.25], [ideal_Z[i]] * 2, color=INK, lw=1.2, ls='--')
    ax.set_xticks(range(3), comps)
    ax.set_ylabel('9⟨h⁻_i h⁺_i⟩  (3π×3π)')
    ax.set_title('(c) 3π×3π: black = ideal H (—), Z (--)', fontsize=9.5, loc='left')
    ax.set_ylim(-1.2, 1.6)
    ax.legend(fontsize=7.5, loc='upper center', ncol=2)
    fig.suptitle('3π: the canonical "exact" h uses a different hadronic current from the generator (TauDecay UFO)',
                 fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'figure_2_threepi_current_mismatch.png', dpi=170)
    plt.close(fig)


def fig_acceptance(D, A, out):
    h = A['h_gen'].astype(float)
    x = A['x'].astype(float)
    m, y = D['modes'], D['labels']
    bins = np.linspace(-1, 1, 21)
    c = 0.5 * (bins[1:] + bins[:-1])
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    ax = axs[0]
    for md, col, mk in ((0, ORANGE, 'o'), (1, BLUE, 's'), (3, VIOLET, '^')):
        hk = np.concatenate([h[(m[:, s] == md) & (y == 1), s, 2] for s in (0, 1)])
        d, _ = np.histogram(hk, bins, density=True)
        ax.plot(c, d, mk + '-', color=col, ms=5, lw=1.8, label=f'{MODES[md]}  (3⟨h_k⟩ = {3 * hk.mean():+.2f})')
    ax.axhline(0.5, color=INK, lw=1, ls='--')
    ax.text(0.98, 0.44, 'unpolarised expectation', ha='right', va='top', fontsize=8, color=MUTED)
    ax.set_xlabel('h_k (one τ, H events, generator h)')
    ax.set_ylabel('density')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('(a) selection sculpts h_k only for π', fontsize=10, loc='left')
    ax = axs[1]
    hk = A['pi_hk'].astype(float)
    lab, acc, cut = A['pi_label'], A['pi_acc'], A['pi_cut']
    for k, col, ls, name in ((acc & (lab == 1), GRAY, '--', 'toy, no selection'),
                             (acc & cut & (lab == 1), AQUA, '-', 'toy, vis. pT>20, |η|<2.5')):
        d, _ = np.histogram(hk[k][:, 0], bins, density=True)
        ax.step(c, d, where='mid', color=col, lw=2, ls=ls, label=name)
    k = (m[:, 0] == 0) & (y == 1)
    d, _ = np.histogram(h[k, 0, 2], bins, density=True)
    e = np.sqrt(np.histogram(h[k, 0, 2], bins)[0]) / (k.sum() * np.diff(bins))
    ax.errorbar(c, d, yerr=e, fmt='o', color=ORANGE, ms=5, label='data (τ⁻ = π)')
    ax.set_xlabel('h_k of τ⁻→πν (H events)')
    ax.set_ylabel('density')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('(b) π: a visible-pT threshold reproduces it', fontsize=10, loc='left')
    ax = axs[2]
    for md, col, mk in ((0, ORANGE, 'o'), (1, BLUE, 's'), (3, VIOLET, '^')):
        hk = np.concatenate([h[m[:, s] == md, s, 2] for s in (0, 1)])
        xs = np.concatenate([x[m[:, s] == md, s] for s in (0, 1)])
        idx = np.digitize(hk, bins) - 1
        med = [np.median(xs[idx == i]) for i in range(len(c))]
        lo = [np.quantile(xs[idx == i], 0.16) for i in range(len(c))]
        hi = [np.quantile(xs[idx == i], 0.84) for i in range(len(c))]
        ax.fill_between(c, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(c, med, mk + '-', color=col, ms=5, lw=1.8,
                label=f'{MODES[md]}  corr = {np.corrcoef(hk, xs)[0, 1]:.2f}')
    ax.set_xlabel('h_k')
    ax.set_ylabel('visible energy fraction x = E_vis/E_τ (lab)')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('(c) why: x tracks h_k for π only (median, 16–84%)', fontsize=10, loc='left')
    fig.suptitle('Visible-energy selection vs the longitudinal polarimeter component; canonical validation cohort, unit weight',
                 fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'figure_3_selection_sculpts_pi.png', dpi=170)
    plt.close(fig)


def fig_joint(D, A, out):
    h = A['h_gen'].astype(float)
    m, y = D['modes'], D['labels']
    hk, lab, acc, cut = A['pi_hk'].astype(float), A['pi_label'], A['pi_acc'], A['pi_cut']
    bins = np.linspace(-1, 1, 11)
    pp = (m[:, 0] == 0) & (m[:, 1] == 0)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True, sharey=True)
    for i, (cls, name) in enumerate(((1, 'H'), (0, 'Z'))):
        panels = [(hk[acc & (lab == cls)], 'toy, no selection'),
                  (hk[acc & cut & (lab == cls)], 'toy, vis. pT>20, |η|<2.5'),
                  (h[pp & (y == cls)][:, :, 2], f'data π×π (N={np.sum(pp & (y == cls))})')]
        for j, (v, title) in enumerate(panels):
            H, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=[bins, bins], density=True)
            im = axs[i, j].imshow(H.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Blues', vmin=0, vmax=0.55)
            axs[i, j].set_title(f'{name}: {title}', fontsize=9.5, loc='left')
            if i == 1:
                axs[i, j].set_xlabel('h_k of τ⁻')
            if j == 0:
                axs[i, j].set_ylabel('h_k of τ⁺')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label('density')
    fig.suptitle('π×π: the threshold removes the low-energy corners where the H (C_kk=−1) and Z (C_kk=+1) patterns differ',
                 fontsize=10.5, x=0.01, ha='left')
    fig.savefig(out / 'figure_4_pipi_longitudinal_joint.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    r = json.loads((args.results / 'results.json').read_text())
    A = np.load(args.results / 'arrays.npz')
    D = np.load(args.surface)
    fig_ladder(r, args.output)
    fig_threepi(D, A, r, args.output)
    fig_acceptance(D, A, args.output)
    fig_joint(D, A, args.output)


if __name__ == '__main__':
    main()
