"""Comprehensive empirical stress study of correlated systematic errors in Z -> tau tau calibration.

Evaluates the ATLAS validation cohort (59,390 events) under:
1. Frame azimuthal jitter sigma_psi (0 to 200 mrad)
2. Coherent frame misalignment psi_0 (-100 to +100 mrad)
3. Kinematic MET resolution sigma_MET (0 to 50 GeV) inducing frame jitter delta_psi(MET)
4. Shared Primary Vertex transverse jitter sigma_PV (0 to 50 um)
5. Mode-pair stratification (pi x pi, pi x rho, rho x rho, 3pi x 3pi, any 3pi, overall)
6. 2D combined phase-space map (sigma_psi, sigma_PV) -> tolerance envelope (|R - 1| <= 1%, 3%)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'a1_current_model'))
sys.path.insert(0, str(HERE))

import analyze as an  # noqa: E402
import reweight as rw  # noqa: E402
from correlated_theory import comps, rotate_transverse  # noqa: E402

BOOT = 200


def wcov_mat(X, S, w):
    """Weighted covariance matrix Cov_w(X_a, S_j)."""
    w = w / w.sum()
    Xc = X - w @ X
    Sc = S - w @ S
    return (Xc * w[:, None]).T @ Sc


def get_transfer_metrics(X, S, wf, idx=None):
    """Compute (r_T, r_A, r_Q1, r_Q2, r_kk, r_Q) from polarimeter representations."""
    if idx is not None:
        X, S, wf = X[idx], S[idx], wf[idx]
    d = np.diag(wcov_mat(X, S, wf))
    rT, rA, rQ1, rQ2, rkk = d[0], d[1], d[2], d[3], d[4]
    rQ = 0.5 * (rQ1 + rQ2)
    return {
        'rT': float(rT), 'rA': float(rA),
        'rQ1': float(rQ1), 'rQ2': float(rQ2),
        'rkk': float(rkk), 'rQ': float(rQ),
        'rT_over_rQ': float(rT / rQ) if rQ != 0 else np.nan,
    }


def main():
    t0 = time.time()
    print("Loading validation cohort and networks...")
    d, nets = an.load_all()
    n_events = len(d['y'])
    modes = d['modes']

    # Spin-flat weight
    y, w0, h_gen = d['y'], d['w0'], d['h_gen']
    piH = w0[y == 1].sum() / w0.sum()
    fH = rw.spin_factor(h_gen, np.ones_like(y))
    fZ = rw.spin_factor(h_gen, np.zeros_like(y))
    wf = w0 / (piH * fH + (1 - piH) * fZ)

    # Truth spin operator
    S = comps(h_gen)

    # Estimators to benchmark
    est = {
        'exact_h': h_gen,
        'reco_full22': nets[('gen', 'full22_s42')]['h_pred'],
        'reco_base': nets[('gen', 'base_s43')]['h_pred'],
        'reco_idealip22': nets[('gen', 'idealip22_s42')]['h_pred'],
    }

    # Load kinematics for MET and PV studies
    data_dir = rw.COHORT
    z_lad = np.load(data_dir / 'ladder_validation.npz')
    ids = z_lad['global_indices']
    assert np.array_equal(ids, d['ids'])
    met = z_lad['reco_met_xy']
    vis = z_lad['reco_visible_tau_lab4']
    rb = z_lad['reco_basis']  # (N, 2, 3, 3)

    vis_px = vis[:, 0, 0] + vis[:, 1, 0]
    vis_py = vis[:, 0, 1] + vis[:, 1, 1]
    qT_x = vis_px + met[:, 0]
    qT_y = vis_py + met[:, 1]
    phi_ditau = np.arctan2(qT_y, qT_x)
    qT = np.sqrt(qT_x**2 + qT_y**2)

    # Load IP track info for PV study
    trk_data = np.load(data_dir / 'ip_tracks.npz')
    pv = trk_data['pv'][ids]
    t = trk_data['trk'][ids]
    lead = t[:, :, 0]

    theta = lead[..., 1]
    phi_trk = lead[..., 2]
    d0 = lead[..., 4]
    z0 = lead[..., 5]

    tdir = np.stack([np.sin(theta) * np.cos(phi_trk), np.sin(theta) * np.sin(phi_trk), np.cos(theta)], -1)
    beam = np.array([*np.median(pv[:, :2], axis=0), -float(np.median((lead[..., 5] - pv[:, None, 2])[np.isfinite(lead[..., 0])]))])
    pca = np.stack([-d0 * np.sin(phi_trk), d0 * np.cos(phi_trk), z0], -1) + beam
    ip_vec = pca - pv[:, None, :]
    ip_vec = ip_vec - np.sum(ip_vec * tdir, -1, keepdims=True) * tdir

    ip_n = np.sum(ip_vec * rb[:, :, 0, :], -1)
    ip_r = np.sum(ip_vec * rb[:, :, 1, :], -1)
    phi_ip = np.arctan2(ip_r, ip_n)

    # Mode scopes
    scopes = {
        'overall': np.ones(n_events, bool),
        'pipi': (modes[:, 0] == 0) & (modes[:, 1] == 0),
        'pirho': ((modes[:, 0] == 0) & (modes[:, 1] == 1)) | ((modes[:, 0] == 1) & (modes[:, 1] == 0)),
        'rhorho': (modes[:, 0] == 1) & (modes[:, 1] == 1),
        '3pi3pi': (modes[:, 0] == 3) & (modes[:, 1] == 3),
        'any3pi': (modes == 3).any(1),
    }

    results = {
        'meta': {
            'n_events': int(n_events),
            'median_qT_GeV': float(np.median(qT)),
            'mean_qT_GeV': float(np.mean(qT)),
            'bootstraps': BOOT,
        },
        'nominal': {},
        'study1_frame_jitter': {},
        'study2_frame_bias': {},
        'study3_met_resolution': {},
        'study4_pv_resolution': {},
        'study5_tolerance_grid': {},
    }

    # 0. Nominal responses
    print("\n--- 0. Nominal Responses ---")
    for name, h_arr in est.items():
        results['nominal'][name] = {}
        for sc_name, sc_mask in scopes.items():
            idx = np.flatnonzero(sc_mask)
            X = comps(h_arr)
            m = get_transfer_metrics(X, S, wf, idx)
            results['nominal'][name][sc_name] = m
            if sc_name == 'overall':
                print(f"{name:15s} nominal rT={m['rT']:.4f}, rQ={m['rQ']:.4f}, rT/rQ={m['rT_over_rQ']:.4f}")

    rng = np.random.default_rng(20261013)

    # 1. Study 1: Frame azimuthal jitter sigma_psi
    print("\n--- 1. Study 1: Frame Azimuthal Jitter (sigma_psi) ---")
    sigma_psi_grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    for sp in sigma_psi_grid:
        key = f"sigma_{int(sp*1000):03d}mrad"
        results['study1_frame_jitter'][key] = {'sigma_psi_rad': float(sp), 'models': {}}
        psi = rng.normal(0, sp, size=n_events) if sp > 0 else np.zeros(n_events)
        
        for name, h_arr in est.items():
            h_rot = rotate_transverse(h_arr, psi)
            X = comps(h_rot)
            nom = results['nominal'][name]['overall']
            m = get_transfer_metrics(X, S, wf)
            rho_T = m['rT'] / nom['rT']
            rho_Q = m['rQ'] / nom['rQ']
            double_R = rho_T / rho_Q
            theory_R = float(np.exp(2.0 * sp**2))
            
            # Bootstrap for reco_full22
            ci95 = [double_R, double_R]
            if name == 'reco_full22' and sp > 0:
                bb = []
                for _ in range(BOOT):
                    b_idx = rng.choice(n_events, n_events)
                    b_nom = get_transfer_metrics(comps(h_arr[b_idx]), S[b_idx], wf[b_idx])
                    b_pert = get_transfer_metrics(X[b_idx], S[b_idx], wf[b_idx])
                    bb.append((b_pert['rT'] / b_nom['rT']) / (b_pert['rQ'] / b_nom['rQ']))
                ci95 = np.quantile(bb, [0.025, 0.975]).tolist()
            
            results['study1_frame_jitter'][key]['models'][name] = {
                'rho_T': float(rho_T), 'rho_Q': float(rho_Q),
                'double_R': float(double_R), 'theory_R': theory_R,
                'ci95': ci95,
            }
            if name == 'reco_full22':
                print(f"sigma_psi={sp*1000:3.0f} mrad: rho_T={rho_T:.4f}, rho_Q={rho_Q:.4f}, R_T/Q={double_R:.4f} [{ci95[0]:.4f}, {ci95[1]:.4f}] (theory: {theory_R:.4f})")

    # 2. Study 2: Coherent Frame Misalignment psi_0
    print("\n--- 2. Study 2: Coherent Frame Misalignment (psi_0) ---")
    psi0_grid = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    for p0 in psi0_grid:
        key = f"bias_{int(p0*1000):+04d}mrad"
        results['study2_frame_bias'][key] = {'psi0_rad': float(p0), 'models': {}}
        for name, h_arr in est.items():
            h_rot = rotate_transverse(h_arr, p0)
            X = comps(h_rot)
            nom = results['nominal'][name]['overall']
            m = get_transfer_metrics(X, S, wf)
            rho_T = m['rT'] / nom['rT']
            rho_Q = m['rQ'] / nom['rQ']
            double_R = rho_T / rho_Q
            results['study2_frame_bias'][key]['models'][name] = {
                'rho_T': float(rho_T), 'rho_Q': float(rho_Q),
                'double_R': float(double_R),
            }
            if name == 'reco_full22':
                print(f"psi_0={p0*1000:+4.0f} mrad: rho_T={rho_T:.4f}, rho_Q={rho_Q:.4f}, R_T/Q={double_R:.4f}")

    # 3. Study 3: Kinematic MET Resolution
    print("\n--- 3. Study 3: Kinematic MET Resolution (sigma_MET) ---")
    sigma_met_grid = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    for sm in sigma_met_grid:
        key = f"sigma_met_{sm:02d}GeV"
        if sm > 0:
            dmet_x = rng.normal(0, sm, size=n_events)
            dmet_y = rng.normal(0, sm, size=n_events)
            qTx_new = qT_x + dmet_x
            qTy_new = qT_y + dmet_y
            phi_new = np.arctan2(qTy_new, qTx_new)
            dpsi = np.arctan2(np.sin(phi_new - phi_ditau), np.cos(phi_new - phi_ditau))
        else:
            dpsi = np.zeros(n_events)
        
        rms_psi = float(np.sqrt(np.average(dpsi**2, weights=wf)))
        results['study3_met_resolution'][key] = {
            'sigma_met_GeV': float(sm),
            'rms_psi_mrad': float(rms_psi * 1000),
            'models': {}
        }
        
        for name, h_arr in est.items():
            h_rot = rotate_transverse(h_arr, dpsi)
            X = comps(h_rot)
            nom = results['nominal'][name]['overall']
            m = get_transfer_metrics(X, S, wf)
            rho_T = m['rT'] / nom['rT']
            rho_Q = m['rQ'] / nom['rQ']
            double_R = rho_T / rho_Q
            theory_R = float(np.exp(2.0 * rms_psi**2))
            
            ci95 = [double_R, double_R]
            if name == 'reco_full22' and sm > 0:
                bb = []
                for _ in range(BOOT):
                    b_idx = rng.choice(n_events, n_events)
                    b_nom = get_transfer_metrics(comps(h_arr[b_idx]), S[b_idx], wf[b_idx])
                    b_pert = get_transfer_metrics(X[b_idx], S[b_idx], wf[b_idx])
                    bb.append((b_pert['rT'] / b_nom['rT']) / (b_pert['rQ'] / b_nom['rQ']))
                ci95 = np.quantile(bb, [0.025, 0.975]).tolist()

            results['study3_met_resolution'][key]['models'][name] = {
                'rho_T': float(rho_T), 'rho_Q': float(rho_Q),
                'double_R': float(double_R), 'theory_R': theory_R,
                'ci95': ci95,
            }
            if name == 'reco_full22':
                print(f"sigma_MET={sm:2d} GeV: rms_psi={rms_psi*1000:5.1f} mrad, rho_T={rho_T:.4f}, rho_Q={rho_Q:.4f}, R_T/Q={double_R:.4f} [{ci95[0]:.4f}, {ci95[1]:.4f}]")

    # 4. Study 4: Shared Primary Vertex Jitter
    print("\n--- 4. Study 4: Shared Primary Vertex Jitter (sigma_PV) ---")
    sigma_pv_grid = [0, 5, 10, 15, 20, 25, 30, 40, 50]  # um
    for spv in sigma_pv_grid:
        key = f"sigma_pv_{spv:02d}um"
        if spv > 0:
            spv_mm = spv * 0.001
            dx = rng.normal(0, spv_mm, size=(n_events, 1))
            dy = rng.normal(0, spv_mm, size=(n_events, 1))
            dz = rng.normal(0, spv_mm * 1.5, size=(n_events, 1))
            dpv = np.stack([dx, dy, dz], -1)

            ip_vec_pert = ip_vec - dpv + np.sum(dpv * tdir, -1, keepdims=True) * tdir
            ip_n_pert = np.sum(ip_vec_pert * rb[:, :, 0, :], -1)
            ip_r_pert = np.sum(ip_vec_pert * rb[:, :, 1, :], -1)
            phi_ip_pert = np.arctan2(ip_r_pert, ip_n_pert)
            dphi_ip = np.arctan2(np.sin(phi_ip_pert - phi_ip), np.cos(phi_ip_pert - phi_ip))
        else:
            dphi_ip = np.zeros((n_events, 2))

        results['study4_pv_resolution'][key] = {'sigma_pv_um': float(spv), 'scopes': {}}
        
        # Apply to reco_full22 for IP-reliant sides (mode 0: pion)
        h_pert = np.copy(est['reco_full22'])
        for side in (0, 1):
            is_pion = (modes[:, side] == 0) & np.isfinite(dphi_ip[:, side])
            c = np.cos(dphi_ip[:, side])
            s = np.sin(dphi_ip[:, side])
            hn = np.where(is_pion, c * h_pert[:, side, 0] + s * h_pert[:, side, 1], h_pert[:, side, 0])
            hr = np.where(is_pion, -s * h_pert[:, side, 0] + c * h_pert[:, side, 1], h_pert[:, side, 1])
            h_pert[:, side, 0], h_pert[:, side, 1] = hn, hr
        
        X = comps(h_pert)
        for sc_name, sc_mask in scopes.items():
            idx = np.flatnonzero(sc_mask)
            nom = results['nominal']['reco_full22'][sc_name]
            m = get_transfer_metrics(X, S, wf, idx)
            rho_T = m['rT'] / nom['rT']
            rho_Q = m['rQ'] / nom['rQ']
            double_R = rho_T / rho_Q
            results['study4_pv_resolution'][key]['scopes'][sc_name] = {
                'rho_T': float(rho_T), 'rho_Q': float(rho_Q),
                'double_R': float(double_R),
            }
        
        ov = results['study4_pv_resolution'][key]['scopes']['overall']
        pi = results['study4_pv_resolution'][key]['scopes']['pipi']
        print(f"sigma_PV={spv:2d} um: overall rho_T={ov['rho_T']:.4f}, rho_Q={ov['rho_Q']:.4f}, R={ov['double_R']:.4f} | pipi rho_T={pi['rho_T']:.4f}, rho_Q={pi['rho_Q']:.4f}, R={pi['double_R']:.4f}")

    # 5. Study 5: 2D Combined Tolerance Grid (sigma_psi, sigma_PV)
    print("\n--- 5. Study 5: 2D Combined Tolerance Grid ---")
    grid_psi = [0, 20, 40, 60, 80, 100, 120, 150]  # mrad
    grid_pv = [0, 10, 20, 30, 40, 50]  # um
    
    grid_res = []
    for sp_mrad in grid_psi:
        sp_rad = sp_mrad * 0.001
        psi = rng.normal(0, sp_rad, size=n_events) if sp_rad > 0 else np.zeros(n_events)
        row = []
        for spv in grid_pv:
            if spv > 0:
                spv_mm = spv * 0.001
                dx = rng.normal(0, spv_mm, size=(n_events, 1))
                dy = rng.normal(0, spv_mm, size=(n_events, 1))
                dz = rng.normal(0, spv_mm * 1.5, size=(n_events, 1))
                dpv = np.stack([dx, dy, dz], -1)

                ip_vec_pert = ip_vec - dpv + np.sum(dpv * tdir, -1, keepdims=True) * tdir
                ip_n_pert = np.sum(ip_vec_pert * rb[:, :, 0, :], -1)
                ip_r_pert = np.sum(ip_vec_pert * rb[:, :, 1, :], -1)
                phi_ip_pert = np.arctan2(ip_r_pert, ip_n_pert)
                dphi_ip = np.arctan2(np.sin(phi_ip_pert - phi_ip), np.cos(phi_ip_pert - phi_ip))
            else:
                dphi_ip = np.zeros((n_events, 2))

            h_pert = np.copy(est['reco_full22'])
            # Apply PV shift to pion sides
            for side in (0, 1):
                is_pion = (modes[:, side] == 0) & np.isfinite(dphi_ip[:, side])
                c = np.cos(dphi_ip[:, side])
                s = np.sin(dphi_ip[:, side])
                hn = np.where(is_pion, c * h_pert[:, side, 0] + s * h_pert[:, side, 1], h_pert[:, side, 0])
                hr = np.where(is_pion, -s * h_pert[:, side, 0] + c * h_pert[:, side, 1], h_pert[:, side, 1])
                h_pert[:, side, 0], h_pert[:, side, 1] = hn, hr
            
            # Apply frame jitter
            h_comb = rotate_transverse(h_pert, psi)
            X = comps(h_comb)
            nom = results['nominal']['reco_full22']['overall']
            m = get_transfer_metrics(X, S, wf)
            double_R = (m['rT'] / nom['rT']) / (m['rQ'] / nom['rQ'])
            row.append(float(double_R))
        grid_res.append(row)
        print(f"psi={sp_mrad:3d} mrad | R values across PV [0..50 um]: {[round(x, 4) for x in row]}")

    results['study5_tolerance_grid'] = {
        'grid_psi_mrad': grid_psi,
        'grid_pv_um': grid_pv,
        'double_R_matrix': grid_res,
    }

    out_json = HERE / 'results' / 'correlated_stress_study.json'
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nAll studies completed in {time.time() - t0:.1f}s. Results written to {out_json}")


if __name__ == '__main__':
    main()
