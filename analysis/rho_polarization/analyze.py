"""Analysis script for rho decay internal polarization and energy sharing asymmetry y.

Investigates:
1. Exact and predicted polarimeter vectors vs y and cos(theta*).
2. H/Z separation AUC across (y1, y2) phase space in rho-rho and mixed modes.
3. Detector charge/neutral reconstruction asymmetry (hard pi0 vs hard pi+/-).
4. Robustness in the unphysical visible mass region (m_vis > m_tau).
5. Effective spin correlation matrix tomography across y bins.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def boost(v, beta):
    """Passive boost of (..., px, py, pz, E) into frame moving with beta."""
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1e-12, 1.0 - b2))
    bv = np.sum(v[..., :3] * beta, axis=-1)
    factor = gamma * gamma / (gamma + 1.0) * bv - gamma * v[..., 3]
    return np.concatenate((v[..., :3] + factor[..., None] * beta,
                           (gamma * (v[..., 3] - bv))[..., None]), axis=-1)


def compute_cos_theta_star(pi_ch, pi_0, nu4):
    """Compute rest-frame decay angle cos(theta*) of charged pion in rho rest frame.
    
    pi_ch, pi_0, nu4: (N, 4) in (px, py, pz, E).
    """
    tau = pi_ch + pi_0 + nu4
    beta_tau = tau[:, :3] / tau[:, 3, None]
    
    # Boost into tau rest frame
    q = pi_ch + pi_0
    q_tau = boost(q, beta_tau)
    pi_ch_tau = boost(pi_ch, beta_tau)
    
    # Boost from tau rest frame into rho rest frame
    beta_rho = q_tau[:, :3] / q_tau[:, 3, None]
    pi_ch_rho = boost(pi_ch_tau, beta_rho)
    
    # In rho rest frame, angle of pi_ch relative to rho direction in tau rest frame
    q_dir = q_tau[:, :3] / np.linalg.norm(q_tau[:, :3], axis=-1, keepdims=True)
    pi_dir = pi_ch_rho[:, :3] / np.linalg.norm(pi_ch_rho[:, :3], axis=-1, keepdims=True)
    cos_theta_star = np.sum(pi_dir * q_dir, axis=-1)
    return cos_theta_star


def compute_bilinear_score(h0, h1):
    """Compute bilinear log-likelihood ratio score for H vs Z."""
    # C_H = diag(1, 1, -1), C_Z = diag(0, 0, 1) in (n, r, k)
    term_H = 1.0 + (h0[:, 0] * h1[:, 0] + h0[:, 1] * h1[:, 1] - h0[:, 2] * h1[:, 2])
    term_Z = 1.0 + (h0[:, 2] * h1[:, 2])
    return np.log(np.maximum(1e-6, term_H)) - np.log(np.maximum(1e-6, term_Z))


def bootstrap_auc_ci(labels, score, weights, n_boot=500, rng_seed=42):
    """Bootstrap 95% CI for weighted AUC."""
    rng = np.random.RandomState(rng_seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        # Check if both classes are present
        if len(np.unique(labels[idx])) < 2:
            continue
        auc = roc_auc_score(labels[idx], score[idx], sample_weight=weights[idx])
        aucs.append(auc)
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    t0 = time.time()
    data_dir = Path('analysis/cp_mixing/data')
    out_dir = Path('analysis/rho_polarization/results')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ladder = np.load(data_dir / 'ladder_validation.npz')
    surface = np.load(data_dir / 'truth_surface_cleo.npz')
    f_full22 = np.load(data_dir / 'gen3pi_full22_s42.npz')
    f_base = np.load(data_dir / 'gen3pi_base_s43.npz')
    f_ideal = np.load(data_dir / 'gen3pi_idealip22_s42.npz')

    modes = ladder['modes']  # (59390, 2): 0: pi, 1: rho, 3: 3pi
    labels = f_full22['labels']
    weights = f_full22['overlap_weights']
    
    # Truth 4-vectors (px, py, pz, E)
    pi_ch_truth = surface['pions'][:, :, 0, :]  # (59390, 2, 4)
    pi0_truth = surface['pi0']  # (59390, 2, 4)
    nu_truth = surface['nu4']  # (59390, 2, 4)
    
    # Reco 4-vectors
    pi_ch_reco = ladder['reco_h_pions_lab4'][:, :, 0, :]  # (59390, 2, 4)
    pi0_reco = ladder['reco_h_neutral_lab4']  # (59390, 2, 4)
    vis_reco = ladder['reco_visible_tau_lab4']  # (59390, 2, 4)

    # Polarimeter vectors (59390, 2, 3) in (n, r, k) basis
    h_exact = f_full22['h']
    h_full22 = f_full22['h_pred']
    h_base = f_base['h_pred']
    h_ideal = f_ideal['h_pred']

    # Compute energy asymmetry y_lab for each tau
    y_truth = (pi_ch_truth[..., 3] - pi0_truth[..., 3]) / (pi_ch_truth[..., 3] + pi0_truth[..., 3])
    y_reco = (pi_ch_reco[..., 3] - pi0_reco[..., 3]) / np.maximum(1e-6, pi_ch_reco[..., 3] + pi0_reco[..., 3])

    # Compute cos(theta*) in rho rest frame
    cos_theta_star = np.zeros_like(y_truth)
    for side in [0, 1]:
        mask_side = (modes[:, side] == 1)
        cos_theta_star[mask_side, side] = compute_cos_theta_star(
            pi_ch_truth[mask_side, side],
            pi0_truth[mask_side, side],
            nu_truth[mask_side, side]
        )

    # Visible mass
    m_vis_reco = np.sqrt(np.maximum(0, vis_reco[..., 3]**2 - np.sum(vis_reco[..., :3]**2, axis=-1)))
    q_truth = pi_ch_truth + pi0_truth
    m_pipi_truth = np.sqrt(np.maximum(0, q_truth[..., 3]**2 - np.sum(q_truth[..., :3]**2, axis=-1)))
    q_reco = pi_ch_reco + pi0_reco
    m_pipi_reco = np.sqrt(np.maximum(0, q_reco[..., 3]**2 - np.sum(q_reco[..., :3]**2, axis=-1)))

    print(f"Data loaded in {time.time() - t0:.2f} s")

    # -------------------------------------------------------------
    # 1. Per-side correlation and response across y bins
    # -------------------------------------------------------------
    print("Evaluating per-side polarimeter quality vs y...")
    side_results = []
    # Combine side 0 and side 1 for all rho sides
    rho_sides_mask = (modes == 1)
    y_all_truth = y_truth[rho_sides_mask]
    y_all_reco = y_reco[rho_sides_mask]
    cts_all = cos_theta_star[rho_sides_mask]
    h_ex_all = h_exact[rho_sides_mask]
    h_pred_all = h_full22[rho_sides_mask]
    h_base_all = h_base[rho_sides_mask]
    h_ideal_all = h_ideal[rho_sides_mask]
    m_vis_all = m_vis_reco[rho_sides_mask]

    # Bins of y_truth: 10 equal bins from -1 to 1
    y_bins = np.linspace(-1.0, 1.0, 11)
    bin_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

    for i in range(len(y_bins) - 1):
        low, high = y_bins[i], y_bins[i + 1]
        in_bin = (y_all_truth >= low) & (y_all_truth < high)
        n_ev = int(in_bin.sum())
        if n_ev == 0:
            continue
        
        # Pearson correlations
        def get_corrs(hp):
            c_k = float(np.corrcoef(hp[in_bin, 2], h_ex_all[in_bin, 2])[0, 1])
            c_n = float(np.corrcoef(hp[in_bin, 0], h_ex_all[in_bin, 0])[0, 1])
            c_r = float(np.corrcoef(hp[in_bin, 1], h_ex_all[in_bin, 1])[0, 1])
            c_perp = 0.5 * (c_n + c_r)
            norm_ratio = float(np.mean(np.linalg.norm(hp[in_bin], axis=-1)) /
                                np.mean(np.linalg.norm(h_ex_all[in_bin], axis=-1)))
            return {
                'corr_longitudinal': c_k,
                'corr_transverse': c_perp,
                'norm_ratio': norm_ratio
            }

        res_bin = {
            'y_low': float(low),
            'y_high': float(high),
            'y_center': float(bin_centers[i]),
            'count': n_ev,
            'mean_cos_theta_star': float(np.mean(cts_all[in_bin])),
            'full22': get_corrs(h_pred_all),
            'base': get_corrs(h_base_all),
            'ideal': get_corrs(h_ideal_all),
            'mean_m_vis_reco': float(np.mean(m_vis_all[in_bin])),
            'unphysical_mass_fraction': float(np.mean(m_vis_all[in_bin] > 1.777))
        }
        side_results.append(res_bin)

    # -------------------------------------------------------------
    # 2. H/Z Separation in rho x rho events
    # -------------------------------------------------------------
    print("Evaluating rho-rho H/Z AUC across (y1, y2)...")
    m_rhorho = (modes[:, 0] == 1) & (modes[:, 1] == 1)
    n_rhorho = int(m_rhorho.sum())
    print(f"Total rho-rho events: {n_rhorho}")

    labels_rr = labels[m_rhorho]
    weights_rr = weights[m_rhorho]
    y0_rr = y_truth[m_rhorho, 0]
    y1_rr = y_truth[m_rhorho, 1]
    y0_reco_rr = y_reco[m_rhorho, 0]
    y1_reco_rr = y_reco[m_rhorho, 1]
    m_vis0_rr = m_vis_reco[m_rhorho, 0]
    m_vis1_rr = m_vis_reco[m_rhorho, 1]

    # Bilinear scores
    score_exact_rr = compute_bilinear_score(h_exact[m_rhorho, 0], h_exact[m_rhorho, 1])
    score_full22_rr = compute_bilinear_score(h_full22[m_rhorho, 0], h_full22[m_rhorho, 1])
    score_base_rr = compute_bilinear_score(h_base[m_rhorho, 0], h_base[m_rhorho, 1])
    score_ideal_rr = compute_bilinear_score(h_ideal[m_rhorho, 0], h_ideal[m_rhorho, 1])

    # Global rho-rho AUCs
    auc_global = {
        'exact': float(roc_auc_score(labels_rr, score_exact_rr, sample_weight=weights_rr)),
        'full22': float(roc_auc_score(labels_rr, score_full22_rr, sample_weight=weights_rr)),
        'base': float(roc_auc_score(labels_rr, score_base_rr, sample_weight=weights_rr)),
        'ideal': float(roc_auc_score(labels_rr, score_ideal_rr, sample_weight=weights_rr)),
    }
    auc_global['full22_ci'] = bootstrap_auc_ci(labels_rr, score_full22_rr, weights_rr)
    auc_global['exact_ci'] = bootstrap_auc_ci(labels_rr, score_exact_rr, weights_rr)
    print(f"Global rho-rho AUC: exact={auc_global['exact']:.4f}, full22={auc_global['full22']:.4f}, base={auc_global['base']:.4f}")

    # Specific physical categories in (y0, y1)
    ay0, ay1 = np.abs(y0_rr), np.abs(y1_rr)
    ay0_rec, ay1_rec = np.abs(y0_reco_rr), np.abs(y1_reco_rr)

    categories = [
        ('both_high_asym_truth', (ay0 >= 0.7) & (ay1 >= 0.7)),
        ('both_high_asym_reco', (ay0_rec >= 0.7) & (ay1_rec >= 0.7)),
        ('both_mid_asym', (ay0 >= 0.35) & (ay0 < 0.7) & (ay1 >= 0.35) & (ay1 < 0.7)),
        ('both_low_asym_symmetric', (ay0 < 0.35) & (ay1 < 0.35)),
        ('mixed_one_low_one_high', ((ay0 < 0.35) & (ay1 >= 0.7)) | ((ay0 >= 0.7) & (ay1 < 0.35))),
        ('both_hard_pi0', (y0_rr < -0.5) & (y1_rr < -0.5)),
        ('both_hard_pich', (y0_rr > 0.5) & (y1_rr > 0.5)),
        ('opposite_sign_hard', ((y0_rr > 0.5) & (y1_rr < -0.5)) | ((y0_rr < -0.5) & (y1_rr > 0.5))),
        ('physical_m_vis', (m_vis0_rr <= 1.777) & (m_vis1_rr <= 1.777)),
        ('unphysical_m_vis_at_least_one', (m_vis0_rr > 1.777) | (m_vis1_rr > 1.777)),
        ('unphysical_m_vis_both', (m_vis0_rr > 1.777) & (m_vis1_rr > 1.777)),
    ]

    cat_results = {}
    for name, cond in categories:
        sub = cond
        cnt = int(sub.sum())
        if cnt < 50:
            continue
        auc_e = float(roc_auc_score(labels_rr[sub], score_exact_rr[sub], sample_weight=weights_rr[sub]))
        auc_f = float(roc_auc_score(labels_rr[sub], score_full22_rr[sub], sample_weight=weights_rr[sub]))
        auc_b = float(roc_auc_score(labels_rr[sub], score_base_rr[sub], sample_weight=weights_rr[sub]))
        auc_i = float(roc_auc_score(labels_rr[sub], score_ideal_rr[sub], sample_weight=weights_rr[sub]))
        ci_f = bootstrap_auc_ci(labels_rr[sub], score_full22_rr[sub], weights_rr[sub], n_boot=300)
        ci_e = bootstrap_auc_ci(labels_rr[sub], score_exact_rr[sub], weights_rr[sub], n_boot=300)

        cat_results[name] = {
            'count': cnt,
            'fraction': float(cnt / n_rhorho),
            'auc_exact': auc_e,
            'auc_exact_ci': ci_e,
            'auc_full22': auc_f,
            'auc_full22_ci': ci_f,
            'auc_base': auc_b,
            'auc_ideal': auc_i,
            'delta_full22_base': float(auc_f - auc_b)
        }
        print(f"Cat {name}: N={cnt}, exact={auc_e:.4f}, full22={auc_f:.4f} {ci_f}")

    # -------------------------------------------------------------
    # 3. 2D grid AUC over (|y0|, |y1|)
    # -------------------------------------------------------------
    print("Evaluating 2D grid AUC...")
    grid_bins = np.array([0.0, 0.35, 0.65, 1.0])
    grid_2d = []
    for i in range(len(grid_bins) - 1):
        for j in range(len(grid_bins) - 1):
            cond = ((ay0 >= grid_bins[i]) & (ay0 < grid_bins[i + 1]) &
                    (ay1 >= grid_bins[j]) & (ay1 < grid_bins[j + 1]))
            cnt = int(cond.sum())
            if cnt < 30:
                continue
            auc_e = float(roc_auc_score(labels_rr[cond], score_exact_rr[cond], sample_weight=weights_rr[cond]))
            auc_f = float(roc_auc_score(labels_rr[cond], score_full22_rr[cond], sample_weight=weights_rr[cond]))
            auc_b = float(roc_auc_score(labels_rr[cond], score_base_rr[cond], sample_weight=weights_rr[cond]))
            grid_2d.append({
                'bin0': [float(grid_bins[i]), float(grid_bins[i + 1])],
                'bin1': [float(grid_bins[j]), float(grid_bins[j + 1])],
                'count': cnt,
                'auc_exact': auc_e,
                'auc_full22': auc_f,
                'auc_base': auc_b
            })

    # -------------------------------------------------------------
    # 4. Mixed Modes: pi x rho and rho x 3pi
    # -------------------------------------------------------------
    print("Evaluating mixed modes vs rho |y|...")
    mixed_results = {}
    for mode_pair, (m0, m1), rho_side in [
        ('pi_rho', (0, 1), 1),
        ('rho_pi', (1, 0), 0),
        ('rho_3pi', (1, 3), 0),
        ('3pi_rho', (3, 1), 1),
    ]:
        m_pair = (modes[:, 0] == m0) & (modes[:, 1] == m1)
        cnt_pair = int(m_pair.sum())
        if cnt_pair == 0:
            continue
        y_rho_side = y_truth[m_pair, rho_side]
        ay_rho = np.abs(y_rho_side)
        
        lbl = labels[m_pair]
        wgt = weights[m_pair]
        sc_ex = compute_bilinear_score(h_exact[m_pair, 0], h_exact[m_pair, 1])
        sc_f = compute_bilinear_score(h_full22[m_pair, 0], h_full22[m_pair, 1])
        sc_b = compute_bilinear_score(h_base[m_pair, 0], h_base[m_pair, 1])

        # Overall
        mixed_results[mode_pair] = {
            'count': cnt_pair,
            'overall_auc_exact': float(roc_auc_score(lbl, sc_ex, sample_weight=wgt)),
            'overall_auc_full22': float(roc_auc_score(lbl, sc_f, sample_weight=wgt)),
            'overall_auc_base': float(roc_auc_score(lbl, sc_b, sample_weight=wgt)),
            'low_y_auc_full22': float(roc_auc_score(lbl[ay_rho < 0.4], sc_f[ay_rho < 0.4], sample_weight=wgt[ay_rho < 0.4])),
            'high_y_auc_full22': float(roc_auc_score(lbl[ay_rho >= 0.7], sc_f[ay_rho >= 0.7], sample_weight=wgt[ay_rho >= 0.7])),
            'low_y_auc_exact': float(roc_auc_score(lbl[ay_rho < 0.4], sc_ex[ay_rho < 0.4], sample_weight=wgt[ay_rho < 0.4])),
            'high_y_auc_exact': float(roc_auc_score(lbl[ay_rho >= 0.7], sc_ex[ay_rho >= 0.7], sample_weight=wgt[ay_rho >= 0.7])),
        }
        print(f"Mixed mode {mode_pair}: N={cnt_pair}, full22 overall={mixed_results[mode_pair]['overall_auc_full22']:.4f}, "
              f"low_y={mixed_results[mode_pair]['low_y_auc_full22']:.4f}, high_y={mixed_results[mode_pair]['high_y_auc_full22']:.4f}")

    # -------------------------------------------------------------
    # 5. Spin correlation tomography C_ij in rho-rho vs y
    # -------------------------------------------------------------
    print("Evaluating spin correlation matrix tomography...")
    def get_correlation_matrix(h0, h1, w):
        w_norm = w / np.sum(w)
        c_mat = np.zeros((3, 3))
        for ii in range(3):
            for jj in range(3):
                c_mat[ii, jj] = 9.0 * np.sum(w_norm * h0[:, ii] * h1[:, jj])
        return c_mat

    # H events and Z events in rho-rho
    is_H_rr = (labels_rr == 1)
    is_Z_rr = (labels_rr == 0)

    tomography = {}
    for tag, cond in [
        ('all_rho_rho', np.ones(n_rhorho, dtype=bool)),
        ('both_high_y', (ay0 >= 0.7) & (ay1 >= 0.7)),
        ('both_low_y', (ay0 < 0.35) & (ay1 < 0.35)),
    ]:
        c_exact_H = get_correlation_matrix(h_exact[m_rhorho, 0][cond & is_H_rr],
                                           h_exact[m_rhorho, 1][cond & is_H_rr],
                                           weights_rr[cond & is_H_rr])
        c_exact_Z = get_correlation_matrix(h_exact[m_rhorho, 0][cond & is_Z_rr],
                                           h_exact[m_rhorho, 1][cond & is_Z_rr],
                                           weights_rr[cond & is_Z_rr])
        c_full_H = get_correlation_matrix(h_full22[m_rhorho, 0][cond & is_H_rr],
                                          h_full22[m_rhorho, 1][cond & is_H_rr],
                                          weights_rr[cond & is_H_rr])
        c_full_Z = get_correlation_matrix(h_full22[m_rhorho, 0][cond & is_Z_rr],
                                          h_full22[m_rhorho, 1][cond & is_Z_rr],
                                          weights_rr[cond & is_Z_rr])
        tomography[tag] = {
            'count': int(cond.sum()),
            'exact_diag_H': [float(c_exact_H[k, k]) for k in range(3)],
            'exact_diag_Z': [float(c_exact_Z[k, k]) for k in range(3)],
            'full22_diag_H': [float(c_full_H[k, k]) for k in range(3)],
            'full22_diag_Z': [float(c_full_Z[k, k]) for k in range(3)],
        }

    # Assemble summary dictionary
    summary = {
        'total_events': int(len(labels)),
        'total_rho_rho_events': n_rhorho,
        'auc_global_rho_rho': auc_global,
        'per_side_y_bins': side_results,
        'category_results': cat_results,
        'grid_2d': grid_2d,
        'mixed_modes': mixed_results,
        'tomography': tomography,
        'runtime_seconds': time.time() - t0
    }

    out_json = out_dir / 'summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {out_json} in {time.time() - t0:.2f} s")

    # Also save npz arrays for plotting
    np.savez_compressed(
        out_dir / 'arrays.npz',
        m_rhorho=m_rhorho,
        y0_rr=y0_rr,
        y1_rr=y1_rr,
        y0_reco_rr=y0_reco_rr,
        y1_reco_rr=y1_reco_rr,
        labels_rr=labels_rr,
        weights_rr=weights_rr,
        score_exact_rr=score_exact_rr,
        score_full22_rr=score_full22_rr,
        score_base_rr=score_base_rr,
        score_ideal_rr=score_ideal_rr,
        m_vis0_rr=m_vis0_rr,
        m_vis1_rr=m_vis1_rr,
        y_all_truth=y_all_truth,
        y_all_reco=y_all_reco,
        cts_all=cts_all
    )
    print("Arrays saved successfully.")


if __name__ == '__main__':
    main()
