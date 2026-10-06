"""Analysis script for 1-prong pion IP kinematics, angular resolution, and spin headroom.

Investigates:
1. Physical lever arm d0 vs visible energy fraction x = E_pi / E_tau and theta*.
2. Angular resolution of reconstructed IP (|Delta phi_IP|, cos Delta phi) vs d0, x, and pT.
3. Neural network polarimeter vector response (h_perp, h_k correlation) for Base, Full22, IdealIP.
4. H/Z separation AUC in pi-pi across (x1, x2) and (d0_1, d0_2) phase space.
5. Propagation to mixed modes (pi x rho, pi x 3pi) and experimental quality selection.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def score_lr(h):
    """Bilinear log-likelihood ratio score for H vs Z in (n, r, k) basis."""
    h0 = h[:, 0]
    h1 = h[:, 1]
    term_H = 1.0 + (h0[:, 0] * h1[:, 0] + h0[:, 1] * h1[:, 1] - h0[:, 2] * h1[:, 2])
    term_Z = 1.0 + (h0[:, 2] * h1[:, 2])
    return np.log(np.maximum(1e-6, term_H)) - np.log(np.maximum(1e-6, term_Z))


def corr(a, b):
    """Pearson correlation coefficient."""
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    denom = np.sqrt(np.sum(a_c * a_c)) * np.sqrt(np.sum(b_c * b_c))
    if denom <= 0:
        return 0.0
    return float(np.sum(a_c * b_c) / denom)


def bootstrap_auc_ci(labels, score, weights, n_boot=500, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    n = len(labels)
    if n < 20 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(labels[idx], score[idx], sample_weight=weights[idx]))
        except Exception:
            continue
    if len(boot_aucs) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(boot_aucs)), float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def main():
    t0 = time.time()
    data_dir = Path("analysis/cp_mixing/data")
    out_dir = Path("analysis/pion_ip_spin/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    val = np.load(data_dir / "ladder_validation.npz")
    ids = val["global_indices"]
    modes = val["modes"]  # (59390, 2)
    labels = val["labels"]  # (59390,)
    weights = val["weights"]  # (59390,)
    truth_vis = val["truth_visible_tau_lab4"]  # (59390, 2, 4)
    truth_nu = val["truth_neutrino_lab4"]  # (59390, 2, 4)
    reco_basis = val["reco_basis"]  # (59390, 2, 3, 3)

    trk = np.load(data_dir / "ip_tracks.npz")
    pv = trk["pv"][ids]
    t = trk["trk"][ids, :, 0]  # leading core track: pt, theta, phi, q, d0, z0, nhits

    theta = t[..., 1]
    phi = t[..., 2]
    d0 = t[..., 4]
    z0 = t[..., 5]
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)

    # Beam reference reconstruction
    bxy = np.median(trk["pv"][:, :2], axis=0)
    ok_lead = np.isfinite(trk["trk"][:, :, 0, 0])
    dz = (trk["trk"][:, :, 0, 5] - trk["pv"][:, None, 2])[ok_lead]
    bz = -float(np.median(dz))
    beam = np.array([bxy[0], bxy[1], bz])

    # Reconstruct PV-referenced 3D IP
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip_pv = pca - pv[:, None, :]
    ip_pv = ip_pv - np.sum(ip_pv * tdir, -1, keepdims=True) * tdir
    ip_norm_um = np.linalg.norm(ip_pv, axis=-1) * 1000.0  # in micrometers

    # Truth kinematics and angles
    tau_p4 = truth_vis + truth_nu
    tau_p3 = tau_p4[..., :3]
    tau_e = tau_p4[..., 3]
    tau_p = np.linalg.norm(tau_p3, axis=-1, keepdims=True)
    tau_dir = tau_p3 / np.maximum(tau_p, 1e-12)

    # Opening angle theta_tau_pi
    pi_p3 = truth_vis[..., :3]
    pi_p = np.linalg.norm(pi_p3, axis=-1, keepdims=True)
    pi_dir = pi_p3 / np.maximum(pi_p, 1e-12)
    cos_theta_tau_pi = np.sum(tau_dir * pi_dir, axis=-1)
    theta_tau_pi_mrad = np.arccos(np.clip(cos_theta_tau_pi, -1.0, 1.0)) * 1000.0

    # True transverse component of tau wrt pion track direction
    tau_perp = tau_dir - np.sum(tau_dir * tdir, -1, keepdims=True) * tdir
    tau_perp_u = unit(tau_perp)
    ip_u = unit(ip_pv)
    cos_dphi = np.sum(tau_perp_u * ip_u, axis=-1)
    dphi_deg = np.arccos(np.clip(cos_dphi, -1.0, 1.0)) * 180.0 / np.pi

    # Visible energy fraction x
    x = truth_vis[..., 3] / np.maximum(tau_e, 1e-12)
    pt_pi = t[..., 0]

    # Models
    print("Loading polarimeter predictions...", flush=True)
    m_base = np.load(data_dir / "cleo_base_s43.npz")
    m_full = np.load(data_dir / "cleo_full22_s42.npz")
    m_ideal = np.load(data_dir / "cleo_idealip22_s42.npz")
    cleo = np.load(data_dir / "truth_surface_cleo.npz")

    h_base = m_base["h_pred"]
    h_full = m_full["h_pred"]
    h_ideal = m_ideal["h_pred"]
    h_exact = cleo["h_ref"]

    sc_base = score_lr(h_base)
    sc_full = score_lr(h_full)
    sc_ideal = score_lr(h_ideal)
    sc_exact = score_lr(h_exact)

    # Valid track mask for pi
    valid_trk = np.isfinite(d0) & np.isfinite(z0) & np.isfinite(theta) & np.isfinite(phi) & (t[..., 0] > 0)
    mask_pi_single = (modes == 0) & valid_trk  # (N, 2)

    # 1. Kinematic lever arm and resolution table vs x
    print("\n--- 1. Single Pion IP Resolution vs x ---")
    x_bins = [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    res_vs_x = []
    for i in range(len(x_bins) - 1):
        x_min, x_max = x_bins[i], x_bins[i + 1]
        sub = mask_pi_single & (x >= x_min) & (x < x_max)
        cos_sub = cos_dphi[sub]
        dphi_sub = dphi_deg[sub]
        ip_sub = ip_norm_um[sub]
        pt_sub = pt_pi[sub]
        theta_sub = theta_tau_pi_mrad[sub]
        res_vs_x.append({
            "x_min": x_min,
            "x_max": x_max,
            "count": int(sub.sum()),
            "mean_cos_dphi": float(np.mean(cos_sub)),
            "median_dphi_deg": float(np.median(dphi_sub)),
            "p68_dphi_deg": float(np.percentile(dphi_sub, 68)),
            "median_ip_um": float(np.median(ip_sub)),
            "median_pt_GeV": float(np.median(pt_sub)),
            "median_theta_tau_pi_mrad": float(np.median(theta_sub)),
        })
        print(f"x in [{x_min:.2f}, {x_max:.2f}): N={res_vs_x[-1]['count']:5d} | "
              f"mean cos={res_vs_x[-1]['mean_cos_dphi']:.4f} | "
              f"med |dphi|={res_vs_x[-1]['median_dphi_deg']:5.1f} deg | "
              f"med IP={res_vs_x[-1]['median_ip_um']:5.1f} um | "
              f"med opening={res_vs_x[-1]['median_theta_tau_pi_mrad']:4.1f} mrad")

    # 2. Polarimeter vector correlation hierarchy vs x
    print("\n--- 2. Polarimeter Vector Correlation Hierarchy vs x ---")
    corr_vs_x = []
    for i in range(len(x_bins) - 1):
        x_min, x_max = x_bins[i], x_bins[i + 1]
        sub = mask_pi_single & (x >= x_min) & (x < x_max)
        ex_n = h_exact[..., 0][sub]
        ex_r = h_exact[..., 1][sub]
        ex_k = h_exact[..., 2][sub]

        entry = {"x_min": x_min, "x_max": x_max, "count": int(sub.sum())}
        for name, h_arr in [("Base", h_base), ("Full22", h_full), ("IdealIP", h_ideal)]:
            vn = h_arr[..., 0][sub]
            vr = h_arr[..., 1][sub]
            vk = h_arr[..., 2][sub]
            c_perp = 0.5 * (corr(vn, ex_n) + corr(vr, ex_r))
            c_k = corr(vk, ex_k)
            entry[f"corr_perp_{name}"] = float(c_perp)
            entry[f"corr_k_{name}"] = float(c_k)
        corr_vs_x.append(entry)
        print(f"x in [{x_min:.2f}, {x_max:.2f}): "
              f"Base perp={entry['corr_perp_Base']:.4f} | "
              f"Full22 perp={entry['corr_perp_Full22']:.4f} | "
              f"IdealIP perp={entry['corr_perp_IdealIP']:.4f} | "
              f"k(Full)={entry['corr_k_Full22']:.4f}")

    # 3. pi x pi Phase Space H/Z Separation AUC
    print("\n--- 3. pi x pi H/Z Separation across Phase Space ---")
    mask_pipi = (modes[:, 0] == 0) & (modes[:, 1] == 0) & valid_trk[:, 0] & valid_trk[:, 1]
    ip_min = np.minimum(ip_norm_um[:, 0], ip_norm_um[:, 1])
    ip_max = np.maximum(ip_norm_um[:, 0], ip_norm_um[:, 1])
    x_max = np.maximum(x[:, 0], x[:, 1])
    x_min = np.minimum(x[:, 0], x[:, 1])

    # Overall pipi AUC
    auc_pipi_base, ci_b_l, ci_b_h = bootstrap_auc_ci(labels[mask_pipi], sc_base[mask_pipi], weights[mask_pipi])
    auc_pipi_full, ci_f_l, ci_f_h = bootstrap_auc_ci(labels[mask_pipi], sc_full[mask_pipi], weights[mask_pipi])
    auc_pipi_ideal, ci_i_l, ci_i_h = bootstrap_auc_ci(labels[mask_pipi], sc_ideal[mask_pipi], weights[mask_pipi])
    auc_pipi_exact, ci_e_l, ci_e_h = bootstrap_auc_ci(labels[mask_pipi], sc_exact[mask_pipi], weights[mask_pipi])

    pipi_overall = {
        "count": int(mask_pipi.sum()),
        "auc_base": auc_pipi_base,
        "ci_base": [ci_b_l, ci_b_h],
        "auc_full": auc_pipi_full,
        "ci_full": [ci_f_l, ci_f_h],
        "auc_ideal": auc_pipi_ideal,
        "ci_ideal": [ci_i_l, ci_i_h],
        "auc_exact": auc_pipi_exact,
        "ci_exact": [ci_e_l, ci_e_h],
        "delta_full_base": auc_pipi_full - auc_pipi_base,
        "headroom_ideal_full": auc_pipi_ideal - auc_pipi_full,
    }
    print(f"Overall pipi (N={pipi_overall['count']}): "
          f"Base={auc_pipi_base:.4f} | Full22={auc_pipi_full:.4f} | Ideal={auc_pipi_ideal:.4f} | Exact={auc_pipi_exact:.4f}")

    # Binned by min IP
    ip_cut_bins = [0, 30, 50, 70, 100, 500]
    pipi_by_min_ip = []
    for i in range(len(ip_cut_bins) - 1):
        low, high = ip_cut_bins[i], ip_cut_bins[i + 1]
        sub = mask_pipi & (ip_min >= low) & (ip_min < high)
        n = int(sub.sum())
        ab, _, _ = bootstrap_auc_ci(labels[sub], sc_base[sub], weights[sub], n_boot=200)
        af, _, _ = bootstrap_auc_ci(labels[sub], sc_full[sub], weights[sub], n_boot=200)
        ai, _, _ = bootstrap_auc_ci(labels[sub], sc_ideal[sub], weights[sub], n_boot=200)
        ae, _, _ = bootstrap_auc_ci(labels[sub], sc_exact[sub], weights[sub], n_boot=200)
        pipi_by_min_ip.append({
            "ip_low": low,
            "ip_high": high,
            "count": n,
            "auc_base": ab,
            "auc_full": af,
            "auc_ideal": ai,
            "auc_exact": ae,
            "delta_full_base": af - ab,
        })
        print(f"min IP in [{low:3d}, {high:3d}) um (N={n:4d}): "
              f"Base={ab:.4f} | Full22={af:.4f} | Ideal={ai:.4f} | Exact={ae:.4f} | Delta={af - ab:+.4f}")

    # 2D (x1, x2) categorization
    pipi_2d_x = []
    x_thresholds = [
        ("Both low (x<0.45)", (x[:, 0] < 0.45) & (x[:, 1] < 0.45)),
        ("One low, one mid", ((x[:, 0] < 0.45) & (x[:, 1] >= 0.45) & (x[:, 1] < 0.75)) |
                             ((x[:, 1] < 0.45) & (x[:, 0] >= 0.45) & (x[:, 0] < 0.75))),
        ("Both mid (0.45<=x<0.75)", (x[:, 0] >= 0.45) & (x[:, 0] < 0.75) & (x[:, 1] >= 0.45) & (x[:, 1] < 0.75)),
        ("One mid, one high", ((x[:, 0] >= 0.45) & (x[:, 0] < 0.75) & (x[:, 1] >= 0.75)) |
                              ((x[:, 1] >= 0.45) & (x[:, 1] < 0.75) & (x[:, 0] >= 0.75))),
        ("One low, one high", ((x[:, 0] < 0.45) & (x[:, 1] >= 0.75)) |
                              ((x[:, 1] < 0.45) & (x[:, 0] >= 0.75))),
        ("Both high (x>=0.75)", (x[:, 0] >= 0.75) & (x[:, 1] >= 0.75)),
    ]
    for r_name, r_cond in x_thresholds:
        sub = mask_pipi & r_cond
        n = int(sub.sum())
        ab, _, _ = bootstrap_auc_ci(labels[sub], sc_base[sub], weights[sub], n_boot=200)
        af, _, _ = bootstrap_auc_ci(labels[sub], sc_full[sub], weights[sub], n_boot=200)
        ai, _, _ = bootstrap_auc_ci(labels[sub], sc_ideal[sub], weights[sub], n_boot=200)
        ae, _, _ = bootstrap_auc_ci(labels[sub], sc_exact[sub], weights[sub], n_boot=200)
        pipi_2d_x.append({
            "region": r_name,
            "count": n,
            "auc_base": ab,
            "auc_full": af,
            "auc_ideal": ai,
            "auc_exact": ae,
            "delta_full_base": af - ab,
        })
        print(f"{r_name:26s} (N={n:4d}): Base={ab:.4f} | Full22={af:.4f} | Ideal={ai:.4f} | Exact={ae:.4f} | Delta={af - ab:+.4f}")

    # 4. Mixed modes (pi x rho, pi x 3pi)
    print("\n--- 4. Mixed Modes: pi x rho and pi x 3pi ---")
    mask_pirho_01 = (modes[:, 0] == 0) & (modes[:, 1] == 1) & valid_trk[:, 0]
    mask_pirho_10 = (modes[:, 0] == 1) & (modes[:, 1] == 0) & valid_trk[:, 1]
    mask_pirho = mask_pirho_01 | mask_pirho_10
    ip_pi_rho = np.where(mask_pirho_01, ip_norm_um[:, 0], ip_norm_um[:, 1])

    mask_pi3pi_02 = (modes[:, 0] == 0) & (modes[:, 1] == 3) & valid_trk[:, 0]
    mask_pi3pi_20 = (modes[:, 0] == 3) & (modes[:, 1] == 0) & valid_trk[:, 1]
    mask_pi3pi = mask_pi3pi_02 | mask_pi3pi_20
    ip_pi_3pi = np.where(mask_pi3pi_02, ip_norm_um[:, 0], ip_norm_um[:, 1])

    mixed_results = {}
    for mname, mmask, ip_arr in [("pi_x_rho", mask_pirho, ip_pi_rho), ("pi_x_3pi", mask_pi3pi, ip_pi_3pi)]:
        ab, _, _ = bootstrap_auc_ci(labels[mmask], sc_base[mmask], weights[mmask], n_boot=200)
        af, _, _ = bootstrap_auc_ci(labels[mmask], sc_full[mmask], weights[mmask], n_boot=200)
        ai, _, _ = bootstrap_auc_ci(labels[mmask], sc_ideal[mmask], weights[mmask], n_boot=200)
        ae, _, _ = bootstrap_auc_ci(labels[mmask], sc_exact[mmask], weights[mmask], n_boot=200)

        # Cuts
        cuts_res = []
        for c in [0, 30, 60, 90, 120]:
            sub = mmask & (ip_arr >= c)
            n = int(sub.sum())
            if n < 20 or len(np.unique(labels[sub])) < 2:
                continue
            cab = roc_auc_score(labels[sub], sc_base[sub], sample_weight=weights[sub])
            caf = roc_auc_score(labels[sub], sc_full[sub], sample_weight=weights[sub])
            cai = roc_auc_score(labels[sub], sc_ideal[sub], sample_weight=weights[sub])
            cuts_res.append({"cut_um": c, "count": n, "auc_base": float(cab), "auc_full": float(caf), "auc_ideal": float(cai)})

        mixed_results[mname] = {
            "total_count": int(mmask.sum()),
            "auc_base": ab, "auc_full": af, "auc_ideal": ai, "auc_exact": ae,
            "cuts": cuts_res
        }
        print(f"{mname} (N={mmask.sum()}): Base={ab:.4f} | Full22={af:.4f} | Ideal={ai:.4f} | Exact={ae:.4f}")
        for cr in cuts_res:
            print(f"  IP_pi >= {cr['cut_um']:3d} um: N={cr['count']:5d} | Base={cr['auc_base']:.4f} | Full22={cr['auc_full']:.4f} | Ideal={cr['auc_ideal']:.4f} | Gain={cr['auc_full'] - cr['auc_base']:+.4f}")

    # 5. Experimental Selection Recipes (Pure Reco)
    print("\n--- 5. Pure Reconstructed Event Selection Recipes ---")
    # For pipi: threshold on min IP
    pipi_cuts_eval = []
    for c in [0, 20, 40, 60, 80, 100]:
        sub = mask_pipi & (ip_min >= c)
        n = int(sub.sum())
        frac = n / mask_pipi.sum()
        ab = roc_auc_score(labels[sub], sc_base[sub], sample_weight=weights[sub])
        af = roc_auc_score(labels[sub], sc_full[sub], sample_weight=weights[sub])
        ai = roc_auc_score(labels[sub], sc_ideal[sub], sample_weight=weights[sub])
        pipi_cuts_eval.append({
            "min_ip_cut_um": c,
            "retained_count": n,
            "efficiency": float(frac),
            "auc_base": float(ab),
            "auc_full22": float(af),
            "auc_idealip22": float(ai),
            "delta_auc": float(af - ab),
        })

    # Save summary json
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_validation_events": len(labels),
        "pipi_overall": pipi_overall,
        "single_pi_resolution_vs_x": res_vs_x,
        "single_pi_polarimeter_corr_vs_x": corr_vs_x,
        "pipi_by_min_ip": pipi_by_min_ip,
        "pipi_2d_x_phase_space": pipi_2d_x,
        "mixed_modes": mixed_results,
        "pipi_experimental_cuts": pipi_cuts_eval,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save npz arrays for plotting
    np.savez_compressed(
        out_dir / "analysis_arrays.npz",
        mask_pi_single=mask_pi_single,
        x=x,
        pt_pi=pt_pi,
        ip_norm_um=ip_norm_um,
        cos_dphi=cos_dphi,
        dphi_deg=dphi_deg,
        theta_tau_pi_mrad=theta_tau_pi_mrad,
        mask_pipi=mask_pipi,
        sc_base=sc_base,
        sc_full=sc_full,
        sc_ideal=sc_ideal,
        sc_exact=sc_exact,
        labels=labels,
        weights=weights,
        modes=modes,
    )
    print(f"\nAnalysis completed in {time.time() - t0:.2f} s. Saved to {out_dir}")


if __name__ == "__main__":
    main()
