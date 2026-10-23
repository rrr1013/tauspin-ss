"""Evaluate full validation arms for the 2x2 oracle matrix + pi0 decomposition:
Computes weighted AUC by pair, side qualities, and paired bootstrap (400 replicates).
Saves structured JSON for plotting and table generation.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN, lr_of, pair_masks, wauc  # noqa: E402
from hquality import wcorr  # noqa: E402


def bootstrap_diff(sA, sB, y, w, pair_mask=None, n_boot=400, seed=20261024):
    rng = np.random.default_rng(seed)
    N = len(y)
    if pair_mask is not None:
        idx_pool = np.flatnonzero(pair_mask)
    else:
        idx_pool = np.arange(N)
    n_sel = len(idx_pool)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        sample = idx_pool[rng.integers(0, n_sel, size=n_sel)]
        aucA = wauc(sA[sample], y[sample], w[sample])
        aucB = wauc(sB[sample], y[sample], w[sample])
        diffs[b] = aucA - aucB
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    return {'mean': mean_diff, 'ci95': [ci_low, ci_high]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-dir', default=Path('/home/rbaba/rho-gap-20261024/out'), type=Path)
    parser.add_argument('--n-boot', default=400, type=int)
    args = parser.parse_args()

    arm_names = [
        'val_reco_pi0_reco_psi',
        'val_reco_pi0_truth_psi',
        'val_truth_pi0_reco_psi',
        'val_truth_pi0_truth_psi',
        'val_truthdir_pi0_reco_psi',
        'val_truthe_pi0_reco_psi'
    ]

    P = {}
    for name in arm_names:
        p_path = args.base_dir / name / 'posterior.npz'
        if p_path.exists():
            P[name] = np.load(p_path)
            print(f"Loaded {name} ({len(P[name]['global_indices'])} events)")
        else:
            print(f"Warning: {p_path} not found, skipping {name}")

    if not P:
        print("No posterior files found!")
        return

    ref_arm = next(iter(P))
    ref = P[ref_arm]
    ids = ref['global_indices']
    y = ref['labels'].astype(int)
    w = ref['weights'].astype(float)
    modes = ref['modes']
    rmode = ref['reco_h_mode'].astype(int)
    av = ref['reco_h_available_side']
    W = np.broadcast_to(w[:, None], av.shape)

    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    look = {int(g): i for i, g in enumerate(t['global_indices'][v])}
    pos = np.array([look[int(g)] for g in ids])
    assert np.array_equal(t['global_indices'][v][pos], ids)
    hx = t['h'][v][pos].astype(float)

    dd = np.load(GEN / 'p11_readout' / 'readout_scores_ens_full22.npz')
    assert np.array_equal(dd['global_indices'][pos], ids)
    s_ens = dd['scores'][pos]

    pairs = pair_masks(modes)
    exact_lr = lr_of(hx)

    results = {
        'events': int(len(ids)),
        'auc': {},
        'side_quality': {},
        'bootstrap': {}
    }

    # Reference exact h and learner
    results['auc']['exact_h'] = {
        'overall': float(wauc(exact_lr, y, w)),
        'pairs': {p: float(wauc(exact_lr[m], y[m], w[m])) for p, m in pairs.items()}
    }
    results['auc']['learner_ens'] = {
        'overall': float(wauc(s_ens, y, w)),
        'pairs': {p: float(wauc(s_ens[m], y[m], w[m])) for p, m in pairs.items()}
    }

    # Learner s42 side quality
    ref_l = np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')
    hl = ref_l['h_pred'].astype(float)
    results['side_quality']['learner_s42'] = {}
    for g, (tm, rmode_val) in [('pi', (0, 0)), ('rho', (1, 1)), ('3pi', (3, 3))]:
        msk = av & (rmode == rmode_val) & (modes == tm)
        c = [float(wcorr(hl[..., i][msk], hx[..., i][msk], W[msk])) for i in range(3)]
        results['side_quality']['learner_s42'][g] = {'hn': c[0], 'hr': c[1], 'hk': c[2], 'sides': int(msk.sum())}

    # Evaluate each arm
    for name, d in P.items():
        lr = d['ip3_sv_mass_lr']
        results['auc'][name] = {
            'overall': float(wauc(lr, y, w)),
            'pairs': {p: float(wauc(lr[m], y[m], w[m])) for p, m in pairs.items()}
        }
        h_side = d['ip3_sv_mass_Eh']
        results['side_quality'][name] = {}
        for g, (tm, rmode_val) in [('pi', (0, 0)), ('rho', (1, 1)), ('3pi', (3, 3))]:
            msk = av & (rmode == rmode_val) & (modes == tm)
            c = [float(wcorr(h_side[..., i][msk], hx[..., i][msk], W[msk])) for i in range(3)]
            results['side_quality'][name][g] = {'hn': c[0], 'hr': c[1], 'hk': c[2], 'sides': int(msk.sum())}

    # Paired Bootstrap
    print(f"Running paired bootstrap ({args.n_boot} replicates)...")
    base_name = 'val_reco_pi0_reco_psi'
    if base_name in P:
        s_base = P[base_name]['ip3_sv_mass_lr']
        for target_name in P:
            if target_name == base_name:
                continue
            s_target = P[target_name]['ip3_sv_mass_lr']
            diff_all = bootstrap_diff(s_target, s_base, y, w, n_boot=args.n_boot)
            diff_rr = bootstrap_diff(s_target, s_base, y, w, pair_mask=pairs['rho x rho'], n_boot=args.n_boot)
            diff_pr = bootstrap_diff(s_target, s_base, y, w, pair_mask=pairs['pi x rho'], n_boot=args.n_boot)
            results['bootstrap'][f'{target_name}_minus_base'] = {
                'overall': diff_all,
                'rho_x_rho': diff_rr,
                'pi_x_rho': diff_pr
            }

        # Compare against learner
        diff_base_learner = bootstrap_diff(s_base, s_ens, y, w, n_boot=args.n_boot)
        diff_base_learner_rr = bootstrap_diff(s_base, s_ens, y, w, pair_mask=pairs['rho x rho'], n_boot=args.n_boot)
        results['bootstrap']['base_minus_learner'] = {
            'overall': diff_base_learner,
            'rho_x_rho': diff_base_learner_rr
        }

        if 'val_truth_pi0_reco_psi' in P:
            s_tp0 = P['val_truth_pi0_reco_psi']['ip3_sv_mass_lr']
            results['bootstrap']['truth_pi0_minus_learner'] = {
                'overall': bootstrap_diff(s_tp0, s_ens, y, w, n_boot=args.n_boot),
                'rho_x_rho': bootstrap_diff(s_tp0, s_ens, y, w, pair_mask=pairs['rho x rho'], n_boot=args.n_boot)
            }
        if 'val_truthdir_pi0_reco_psi' in P:
            s_tdir = P['val_truthdir_pi0_reco_psi']['ip3_sv_mass_lr']
            results['bootstrap']['truthdir_minus_learner'] = {
                'overall': bootstrap_diff(s_tdir, s_ens, y, w, n_boot=args.n_boot),
                'rho_x_rho': bootstrap_diff(s_tdir, s_ens, y, w, pair_mask=pairs['rho x rho'], n_boot=args.n_boot)
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
