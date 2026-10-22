"""Figures of the tau-kinematics run (results/ in this directory).

fig1_auc_pairs.png        weighted H/Z AUC by truth mode pair: classical arms vs the learner
fig2_side_quality.png     per-side correlation with the exact h (n, r, k) on correctly reconstructed pi / rho / 3pi sides
fig3_pi_ip_bins.png       pi sides: transverse quality vs the measured |IP| of the track (6 quantile bins)
fig4_vis_mismatch.png     reco pi sides: ATLAS tau four-vector vs track energy, and the pi-side h_k error before / after
fig5_ip_azimuth_bias.png  TRAIN (1-prong leading track): measured vs truth tau azimuth around the track, and the mean signed
                          azimuth difference vs the measured azimuth, by |IP|
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN  # noqa: E402
from hquality import wcorr  # noqa: E402
import posterior as CP  # noqa: E402

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
ROWS = [  # (label, key in eval scores / side_quality, color, marker, filled)
    ('classical, 10/22 primary (τ four-vector as visible)', 'base:ip3_sv_mass', C[1], 'o', False),
    ('classical, visible = track (polarimeter input)', 'track:ip3_sv_mass', C[1], 'o', True),
    ('  + decay length (|IP|, SV length)', 'track_dl:ip3_sv_mass', C[3], 's', True),
    ('  + IP response by track pT, |η|', 'track_dl_fine:ip3_sv_mass', C[4], 'v', True),
    ('  + joint IP azimuth × |IP| response (1-prong)', 'track_dl_joint:ip3_sv_mass', C[5], 'P', True),
    ('  + TRAIN parent-mass prior', 'track_dl_mt:ip3_sv_mass', C[2], 'D', True),
    ('  + IP response by signed IP orientation', 'track_dl_orient:ip3_sv_mass', C[2], '*', True),
    ('learner, no IP/SV (base)', 'learner base s43', C[0], '^', False),
    ('learner, main line (full22)', 'learner', C[0], '^', True),
]


def fig1(res, out):
    pairs = ['overall', 'pi x pi', 'pi x rho', 'rho x rho', 'rho x 3pi', '3pi x 3pi']
    names = {'overall': 'all events', 'pi x pi': 'π × π', 'pi x rho': 'π × ρ', 'rho x rho': 'ρ × ρ',
             'rho x 3pi': 'ρ × 3π', '3pi x 3pi': '3π × 3π'}
    rows = [r for r in ROWS if r[1] in res['auc']]
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for j, (lab, key, col, mk, fill) in enumerate(rows):
        yy = np.arange(len(pairs)) + (j - (len(rows) - 1) / 2) * 0.11
        ax.plot([res['auc'][key][p] for p in pairs], yy, ls='none', marker=mk, ms=7.5, mec=col,
                mfc=col if fill else 'white', mew=1.5, label=lab.strip())
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels([f"{names[p]}\n({res['events_truth_pairs'][p]:,} events)" for p in pairs])
    ax.invert_yaxis()
    ax.set_xlabel('weighted H/Z AUC (validation, parent-overlap weights)')
    ax.grid(axis='x', color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.42, -0.09), ncol=2, frameon=False)
    ax.set_title('ATLAS full-reco H/Z (both 91.19 GeV), by truth decay-mode pair\n'
                 'classical: IP + SV + MET + parent mass, π⁰ marginalised on reco ρ', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'fig1_auc_pairs.png', dpi=150)


def fig2(res, out):
    groups = [('pi', 'π side (truth π, reco π)'), ('rho', 'ρ side (truth ρ, reco ρ)'), ('3pi', '3π side (truth 3π, reco 3π)')]
    key = lambda k: 'learner full22 s42' if k == 'learner' else k
    rows = [r for r in ROWS if key(r[1]) in res['side_quality']]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for ax, (g, title) in zip(axs, groups):
        for j, (lab, k, col, mk, fill) in enumerate(rows):
            q = res['side_quality'][key(k)][g]
            xx = np.arange(3) + (j - (len(rows) - 1) / 2) * 0.1
            ax.plot(xx, [q[c] for c in 'nrk'], ls='none', marker=mk, ms=7, mec=col, mfc=col if fill else 'white',
                    mew=1.4, label=lab.strip())
        ax.set_xticks(range(3))
        ax.set_xticklabels(['h_n', 'h_r', 'h_k'])
        ax.set_title(f"{title}\n{res['side_quality'][key(rows[0][1])][g]['sides']:,} sides", fontsize=9)
        ax.grid(axis='y', color='#dddddd', lw=0.6)
        ax.set_axisbelow(True)
    axs[0].set_ylabel('weighted correlation with the exact h (validation)')
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7.5, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out / 'fig2_side_quality.png', dpi=150)


def pi_sides(posts):
    ref = np.load(posts[next(iter(posts))])
    ids, w = ref['global_indices'], ref['weights'].astype(float)
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    assert np.array_equal(t['global_indices'][v], ids)
    hx = t['h'][v].astype(float)
    msk = ref['reco_h_available_side'] & (ref['reco_h_mode'] == 0) & (ref['modes'] == 0)
    return ref, ids, w, hx, msk


def fig3(posts, out):
    ref, ids, w, hx, msk = pi_sides(posts)
    tracks = dict(np.load(CP.TRACKS))
    m = CP.P12.ip_measurements(tracks, ids, 0)
    msk = msk & m['available']
    W = np.broadcast_to(w[:, None], msk.shape)
    x = np.log10(np.maximum(m['mag_um'], 1e-3))
    e = np.quantile(x[msk], np.linspace(0, 1, 7))
    cen = [np.median(x[msk & (x >= e[i]) & (x <= e[i + 1])]) for i in range(6)]
    est = {k: np.load(p)['ip3_sv_mass_Eh'] for k, p in posts.items()}
    est['learner full22 s42'] = np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')['h_pred']
    est['learner base s43'] = np.load(GEN / 'p11' / 'base_s43' / 'validation_predictions.npz')['h_pred']
    style = {r[1]: r for r in ROWS}
    style['learner full22 s42'] = style['learner']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for c, ax in enumerate(axs):
        for k, h in est.items():
            if k + ':ip3_sv_mass' in style:
                lab, _, col, mk, fill = style[k + ':ip3_sv_mass']
            elif k in style:
                lab, _, col, mk, fill = style[k]
            else:
                continue
            yy = [wcorr(h[..., c][s], hx[..., c][s], W[s]) for s in
                  (msk & (x >= e[i]) & (x <= e[i + 1]) for i in range(6))]
            ax.plot(10 ** np.array(cen), yy, marker=mk, ms=6.5, color=col, mfc=col if fill else 'white', mew=1.3,
                    lw=1.2, ls='-' if fill else '--', label=lab.strip())
        ax.set_xscale('log')
        ax.set_xlabel('measured |IP| of the π track, PV-referenced [μm]\n(bin median; 1/6 of the sides per bin)')
        ax.set_title(f"π side (truth π, reco π): {'h_n' if c == 0 else 'h_r'}", fontsize=9.5)
        ax.grid(color='#dddddd', lw=0.6)
        ax.set_axisbelow(True)
    axs[0].set_ylabel('weighted correlation with the exact h')
    axs[0].legend(fontsize=7.2, loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fig3_pi_ip_bins.png', dpi=150)


def fig4(posts, out):
    ref, ids, w, hx, msk = pi_sides(posts)
    with np.load(CP.LADDER / 'validation.npz') as d:
        assert np.array_equal(d['global_indices'], ids)
        ev = d['reco_visible_tau_lab4'][..., 3]
        et = d['reco_h_pions_lab4'][:, :, 0, 3]
        etrue = d['truth_visible_tau_lab4'][..., 3]
    r = ev / et - 1
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    bins = np.linspace(-0.3, 0.6, 61)
    rm = ref['reco_h_mode'] == 0
    av = ref['reco_h_available_side']
    for lab, s, col, ls in (('truth π, reco π', msk, C[0], '-'),
                            ('truth ρ, reco π (π⁰ lost)', av & rm & (ref['modes'] == 1), C[1], '--')):
        axs[0].hist(np.clip(r[s], bins[0], bins[-1]), bins, weights=np.broadcast_to(w[:, None], s.shape)[s],
                    density=True, histtype='step', lw=1.6, color=col, ls=ls, label=lab)
    axs[0].set_xlabel('E(ATLAS τ four-vector) / E(track) − 1\n(under/overflow in the end bins)')
    axs[0].set_ylabel('weighted density')
    axs[0].set_yscale('log')
    axs[0].legend(fontsize=8, frameon=False)
    axs[0].set_title('reco π sides, validation', fontsize=9.5)
    # pi side h_k quality vs the mismatch (exactly 0 on most sides, so explicit bins)
    x = r[msk]
    edges = [(-1e-9, 1e-9, '= 0'), (1e-9, 0.05, '0-0.05'), (0.05, 0.15, '0.05-0.15'), (0.15, 0.3, '0.15-0.3'), (0.3, np.inf, '> 0.3')]
    W = np.broadcast_to(w[:, None], msk.shape)
    est = {'base': np.load(posts['base'])['ip3_sv_mass_Eh'], 'track': np.load(posts['track'])['ip3_sv_mass_Eh'],
           'learner': np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')['h_pred']}
    style = {r_[1].split(':')[0]: r_ for r_ in ROWS}
    frac = []
    for k, h in est.items():
        lab, _, col, mk, fill = style[k]
        yy = []
        for lo, hi, _ in edges:
            s_ = np.zeros_like(msk)
            s_[msk] = (x > lo) & (x <= hi)
            yy.append(wcorr(h[..., 2][s_], hx[..., 2][s_], W[s_]))
            if k == 'base':
                frac.append(np.mean((x > lo) & (x <= hi)))
        axs[1].plot(range(len(edges)), yy, marker=mk, ms=6.5, color=col, mfc=col if fill else 'white', mew=1.3, lw=1.2,
                    ls='-' if fill else '--', label=lab.strip())
    axs[1].set_xticks(range(len(edges)))
    axs[1].set_xticklabels([f'{e_[2]}\n({f:.0%})' for e_, f in zip(edges, frac)], fontsize=8)
    axs[1].set_xlabel('E(ATLAS τ four-vector) / E(track) − 1   (share of truth-π sides)')
    axs[1].set_ylabel('weighted correlation of h_k with the exact h_k')
    axs[1].set_title('π side (truth π, reco π): h_k vs the visible-energy mismatch', fontsize=9.5)
    axs[1].legend(fontsize=7.5, frameon=False, loc='lower left')
    for ax in axs:
        ax.grid(color='#dddddd', lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / 'fig4_vis_mismatch.png', dpi=150)


def fig5(out):
    P12 = CP.P12
    with np.load(CP.LADDER / 'train.npz') as d:
        ids, ev = d['global_indices'], d['event_numbers']
        tt = d['truth_visible_tau_lab4'] + d['truth_neutrino_lab4']
    keep = ev % 5 != 0
    tracks = dict(np.load(CP.TRACKS))
    m = P12.ip_measurements(tracks, ids, 0)
    psi_t = P12.azimuth_in_frame(CP.unit(tt[..., :3]), m['e1'], m['e2'])
    dp = np.angle(np.exp(1j * (psi_t - m['psi'])))
    ok = m['available'] & (m['n_core'] < 2) & keep[:, None]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    b = np.linspace(-180, 180, 37)
    axs[0].hist(np.degrees(psi_t[ok]), b, density=True, histtype='step', lw=1.6, color=C[0], label='truth τ direction')
    axs[0].hist(np.degrees(m['psi'][ok]), b, density=True, histtype='step', lw=1.6, color=C[1], ls='--', label='measured IP')
    axs[0].set_xlabel('azimuth around the π track [deg]  (0, ±180: along d0;  ±90: along z0 sinθ)')
    axs[0].set_ylabel('density [1/deg]')
    axs[0].set_xticks(np.arange(-180, 181, 90))
    axs[0].legend(fontsize=8, frameon=False, loc='lower center')
    axs[0].set_title(f'TRAIN, 1-prong leading track ({ok.sum():,} sides)', fontsize=9.5)
    cen = np.linspace(-180, 180, 17)
    cc = 0.5 * (cen[1:] + cen[:-1])
    for (lo, hi), col, mk, ls in (((10, 25), C[1], 'o', '-'), ((25, 60), C[3], 's', '--'), ((60, 125), C[2], 'D', '-.'),
                                  ((125, 1e9), C[0], '^', ':')):
        s = ok & (m['mag_um'] >= lo) & (m['mag_um'] < hi)
        k = np.clip(np.digitize(np.degrees(m['psi']), cen) - 1, 0, 15)
        yy = [np.degrees(np.angle(np.mean(np.exp(1j * dp[s & (k == i)])))) for i in range(16)]
        axs[1].plot(cc, yy, marker=mk, ms=5, color=col, ls=ls, lw=1.2,
                    label=f'|IP| {lo}–{hi:.0f} μm' if hi < 1e8 else f'|IP| > {lo} μm')
    axs[1].axhline(0, color='#999999', lw=0.8)
    axs[1].set_xticks(np.arange(-180, 181, 90))
    axs[1].set_xlabel('measured IP azimuth around the track [deg]')
    axs[1].set_ylabel('circular mean of (truth − measured) azimuth [deg]')
    axs[1].set_title('the measured azimuth is pulled toward the z0 axis (±90°)', fontsize=9.5)
    axs[1].legend(fontsize=8, frameon=False, ncol=2)
    for ax in axs:
        ax.grid(color='#dddddd', lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / 'fig5_ip_azimuth_bias.png', dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', type=Path, required=True)
    ap.add_argument('--post', nargs='+', required=True, help='NAME=posterior.npz')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = json.loads(a.eval.read_text())
    posts = dict(s.split('=', 1) for s in a.post)
    fig1(res, a.out)
    fig2(res, a.out)
    fig3(posts, a.out)
    fig4(posts, a.out)
    fig5(a.out)


if __name__ == '__main__':
    main()
