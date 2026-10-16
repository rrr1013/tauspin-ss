"""Figures of the ATLAS classical-posterior run (all numbers recomputed from the npz files).

fig1_auc_pairs.png   weighted H/Z AUC by truth decay-mode pair: classical posterior variants vs
                     the geometry-free learner (P4 baseline, CLEO 3pi teacher), the main-line
                     learner (gen3pi full22 ensemble) and the reco-visible + truth-nu reference
fig2_side_quality.png per-side weighted correlation of each h component with the exact h, for
                     correctly reconstructed pi, rho (1p1n) and 3pi sides
fig3_geometry_gain.png AUC gain from IP/SV by mode pair, classical vs learner
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import wauc, pair_masks  # noqa: E402

GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
P4 = Path('/home/rbaba/hz-beyond-ceiling-20260923/artifacts')
LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz')
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#7a7a74'
PAIRS = ['pi x pi', 'pi x rho', 'rho x rho', 'pi x 3pi', 'rho x 3pi', '3pi x 3pi', 'overall']


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    R = a.root
    M = np.load(R / 'main' / 'posterior.npz')
    ids, y, w = M['global_indices'], M['labels'].astype(int), M['weights'].astype(float)
    sel = pair_masks(M['modes'])
    load = lambda tag: np.load(R / tag / 'posterior.npz')
    E, X = load('exactmet'), load('xprior_aniso')
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    assert np.array_equal(t['global_indices'][v], ids)
    hx = t['h'][v].astype(float)

    def score(path):
        d = np.load(path)
        assert np.array_equal(d['global_indices'], ids)
        return d['scores']
    lp4b = score(P4 / 'p4_readout' / 'readout_scores_baseline.npz')
    lp4i = score(P4 / 'p4_readout' / 'readout_scores_ip_pv.npz')
    lbase = score(GEN / 'p11_readout' / 'readout_scores_base_s43.npz')
    lfull = score(GEN / 'p11_readout' / 'readout_scores_ens_full22.npz')
    auc = lambda s: [wauc(s[sel[p]], y[sel[p]], w[sel[p]]) for p in PAIRS]

    # fig1
    rows = [
        ('classical: MET only', auc(M['met_lr']), ORANGE, 'o', 'none'),
        ('classical: IP+SV+mass', auc(M['ip3_sv_mass_lr']), ORANGE, 'o', ORANGE),
        ('classical: IP+SV+mass, exact MET (oracle)', auc(E['ip3_sv_mass_lr']), ORANGE, 'D', 'none'),
        ('classical: IP+SV+mass, x prior, aniso MET', auc(X['ip3_sv_mass_lr']), ORANGE, 's', ORANGE),
        ('learner: no geometry (P4 baseline, CLEO 3pi)', auc(lp4b), BLUE, 'o', 'none'),
        ('learner: main line (gen3pi full22 ens)', auc(lfull), BLUE, 'o', BLUE),
        ('reco visible + truth nu (exact formula)', auc(M['truthnu_lr']), GRAY, '|', GRAY),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    n = len(rows)
    for k, (lab, vals, c, mk, fc) in enumerate(rows):
        yy = np.arange(len(PAIRS)) + (k - (n - 1) / 2) * 0.1
        ax.plot(vals, yy, ls='none', marker=mk, mfc=fc, mec=c, color=c, ms=7 if mk != '|' else 12,
                mew=1.6, label=lab)
    for i in range(len(PAIRS) - 1):
        ax.axhline(i + 0.5, color='#dddddb', lw=0.8, zorder=0)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([f"{p}\n({int(sel[p].sum())})" for p in PAIRS])
    ax.invert_yaxis()
    ax.set_xlabel('weighted H/Z AUC (validation, parent-overlap weights)')
    ax.set_title('ATLAS full reco: classical posterior polarimeter vs learned point h', fontsize=11)
    ax.grid(axis='x', color='#eeeeec')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    ax.set_xlim(0.54, 0.73)
    fig.tight_layout()
    fig.savefig(a.output / 'fig1_auc_pairs.png', dpi=160)
    plt.close(fig)

    # fig2
    L = np.load(LADDER)
    rm, raw, tm, av = L['reco_h_mode'], L['reco_mode_raw'], L['modes'], L['reco_h_available_side']
    W = np.broadcast_to(w[:, None], rm.shape)
    cats = [('pi (truth pi)', (rm == 0) & (tm == 0)), ('rho 1p1n (truth rho)', (rm == 1) & (tm == 1) & (raw == 1)),
            ('3pi (truth 3pi)', (rm == 3) & (tm == 3))]
    d4 = np.load(P4 / 'p4' / 'baseline' / 'validation_predictions.npz')
    assert np.array_equal(d4['global_indices'], ids)
    hp4 = d4['h_pred'].astype(float)
    hfull = np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')['h_pred'].astype(float)
    est = [('classical: MET only', M['met_Eh'], ORANGE, 'o', 'none'),
           ('classical: IP+SV+mass', M['ip3_sv_mass_Eh'], ORANGE, 'o', ORANGE),
           ('classical: IP+SV+mass, exact MET', E['ip3_sv_mass_Eh'], ORANGE, 'D', 'none'),
           ('classical: IP+SV+mass, x prior, aniso MET', X['ip3_sv_mass_Eh'], ORANGE, 's', ORANGE),
           ('learner: no geometry (P4 baseline)', hp4, BLUE, 'o', 'none'),
           ('learner: main line (full22 s42)', hfull, BLUE, 'o', BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    for ax, (cname, cm) in zip(axes, cats):
        for k, (lab, h, c, mk, fc) in enumerate(est):
            m = cm & av & np.isfinite(h).all(-1)
            vals = [wcorr(h[..., i][m], hx[..., i][m], W[m]) for i in range(3)]
            ax.plot(np.arange(3) + (k - 2.5) * 0.09, vals, ls='none', marker=mk, mfc=fc, mec=c, ms=6,
                    mew=1.5, label=lab)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['h_n', 'h_r', 'h_k'])
        ax.set_title(f"{cname}  [{int((cm & av).sum())} sides]", fontsize=9.5)
        ax.grid(axis='y', color='#eeeeec')
        ax.set_ylim(-0.05, 1.0)
    axes[0].set_ylabel('weighted corr(estimate, exact h)')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Per-side h quality on correctly reconstructed sides (P4 learner: CLEO 3pi teacher)', fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(a.output / 'fig2_side_quality.png', dpi=160)
    plt.close(fig)

    # fig3
    g_cl = np.array(auc(M['ip3_sv_lr'])) - np.array(auc(M['met_lr']))
    g_p4 = np.array(auc(lp4i)) - np.array(auc(lp4b))
    g_gen = np.array(auc(lfull)) - np.array(auc(lbase))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xx = np.arange(len(PAIRS))
    for off, g, c, mk, fc, lab in ((-0.15, g_cl, ORANGE, 'o', ORANGE, 'classical: MET only -> +IP3+SV'),
                                   (0.0, g_p4, BLUE, 'o', 'none', 'learner P4: no geometry -> +lead-track PV IP'),
                                   (0.15, g_gen, BLUE, 's', BLUE, 'learner gen3pi: base (has scalar d0) -> full22')):
        ax.vlines(xx + off, 0, g, color=c, lw=2)
        ax.plot(xx + off, g, ls='none', marker=mk, mfc=fc, mec=c, ms=7, mew=1.5, label=lab)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(PAIRS, fontsize=8.5)
    ax.set_ylabel('AUC gain from IP/SV')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', color='#eeeeec')
    fig.tight_layout()
    fig.savefig(a.output / 'fig3_geometry_gain.png', dpi=160)
    plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    main()
