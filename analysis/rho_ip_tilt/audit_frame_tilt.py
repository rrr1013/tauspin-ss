#!/usr/bin/env python3
"""Audit of track-frame vs visible-frame tilt in tau impact parameters.

Dissects why the PV-referenced impact parameter improved 1p0n x 3p0n (+0.017)
and 1p0n x 1p0n (+0.007) in the 2026-09-23 run, but yielded only +0.003 in
rho x rho (1p1n x 1p1n), despite rho representing 56% of all taus.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - np.sum(v * axis, axis=-1, keepdims=True) * axis


def signed_azimuth(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Angle from a to b around axis (both already perpendicular to axis)."""
    return np.arctan2(np.sum(np.cross(a, b) * axis, axis=-1), np.sum(a * b, axis=-1))


def stats(x: np.ndarray) -> dict:
    a = np.abs(x)
    return {
        "n": int(x.size),
        "median_abs_rad": float(np.median(a)) if x.size else 0.0,
        "p68_abs_rad": float(np.percentile(a, 68)) if x.size else 0.0,
        "frac_within_30deg": float(np.mean(a < np.pi / 6)) if x.size else 0.0,
        "frac_within_45deg": float(np.mean(a < np.pi / 4)) if x.size else 0.0,
        "frac_within_90deg": float(np.mean(a < np.pi / 2)) if x.size else 0.0,
        "mean_cos": float(np.mean(np.cos(x))) if x.size else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ladder-dir", type=Path, default=Path("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1"))
    p.add_argument("--ip-tracks", type=Path, default=Path("/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis/rho_ip_tilt/results"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    t0 = time.time()
    val = np.load(args.ladder_dir / "validation.npz")
    ids = val["global_indices"]
    trk = np.load(args.ip_tracks)
    pv = trk["pv"][ids]
    t = trk["trk"][ids, :, 0]  # leading core track
    theta, phi, d0, z0 = (t[..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)

    # Beam reference
    ok = np.isfinite(trk["trk"][:, :, 0, 0])
    bxy = np.median(trk["pv"][:, :2], axis=0)
    dz = (trk["trk"][:, :, 0, 5] - trk["pv"][:, None, 2])[ok]
    bz = -float(np.median(dz))
    beam = np.array([bxy[0], bxy[1], bz])

    # Reconstruct PV-referenced IP vector
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip_pv = perp(pca - pv[:, None, :], tdir)
    u_ip = unit(ip_pv)

    # Kinematics and basis
    vis = val["truth_visible_tau_lab4"][..., :3]
    nu = val["truth_neutrino_lab4"][..., :3]
    tau_hat = unit(vis + nu)
    basis = val["reco_basis"]  # (N, 2, (n, r, k), xyz)
    n_vec = basis[..., 0, :]
    r_vec = basis[..., 1, :]
    k_vec = basis[..., 2, :]

    modes = val["reco_h_mode"]
    truth_modes = val["modes"]
    labels = val["labels"]
    weights = val["weights"]

    e_ch = val["reco_h_pions_lab4"][..., 0, 3]
    e_ne = val["reco_h_neutral_lab4"][..., 3]
    tot_e = e_ch + e_ne
    upsilon = np.where(tot_e > 0, (e_ch - e_ne) / tot_e, 0.0)

    # Opening angles
    cos_tk = np.sum(tdir * k_vec, -1)
    theta_tk_mrad = np.arccos(np.clip(cos_tk, -1, 1)) * 1e3
    cos_tau_t = np.sum(tau_hat * tdir, -1)
    theta_tau_t_mrad = np.arccos(np.clip(cos_tau_t, -1, 1)) * 1e3
    sin_tau_t = np.sqrt(np.maximum(1 - cos_tau_t**2, 0.0))

    # Azimuth around track axis
    tau_perp_t = perp(tau_hat, tdir)
    ip_perp_t = perp(ip_pv, tdir)
    dphi_track = signed_azimuth(unit(tau_perp_t), unit(ip_perp_t), tdir)

    # Azimuth around visible reco axis (k)
    tau_perp_k = perp(tau_hat, k_vec)
    ip_perp_k = perp(ip_pv, k_vec)
    dphi_vis = signed_azimuth(unit(tau_perp_k), unit(ip_perp_k), k_vec)

    # Parallax corrected: tau_rec = cos(theta_tau_t) * t + sin(theta_tau_t) * u_ip
    tau_rec = cos_tau_t[..., None] * tdir + sin_tau_t[..., None] * u_ip
    tau_rec_perp_k = perp(tau_rec, k_vec)
    dphi_corr = signed_azimuth(unit(tau_perp_k), unit(tau_rec_perp_k), k_vec)

    # Approximate parallax correction using only reconstructed quantities:
    # tau_rec_approx = t + theta_expected * u_ip, where theta_expected ~ m_tau / E_vis
    e_vis = val["reco_visible_tau_lab4"][..., 3]
    theta_approx = np.clip(1.77686 / np.maximum(e_vis, 10.0), 0.005, 0.05)
    tau_rec_approx = tdir + theta_approx[..., None] * u_ip
    tau_rec_approx_perp_k = perp(tau_rec_approx, k_vec)
    dphi_approx = signed_azimuth(unit(tau_perp_k), unit(tau_rec_approx_perp_k), k_vec)

    print(f"Data processed in {time.time()-t0:.1f}s", flush=True)

    results = {
        "n_events": int(len(ids)),
        "beam_reference_mm": beam.tolist(),
        "by_mode": {},
        "rho_by_upsilon": [],
        "rho_by_theta_tk": [],
        "rho_by_ip_um": [],
    }

    mode_names = {0: "pi (1p0n)", 1: "rho (1p1n)", 3: "3pi (3p0n)"}
    for m, name in mode_names.items():
        # Evaluate for side 0 (side 1 is symmetric)
        mask = (modes[:, 0] == m) & np.isfinite(d0[:, 0]) & np.isfinite(tau_hat[:, 0, 0])
        th_tk = theta_tk_mrad[:, 0][mask]
        th_tau_t = theta_tau_t_mrad[:, 0][mask]
        dp_t = dphi_track[:, 0][mask]
        dp_k = dphi_vis[:, 0][mask]
        dp_c = dphi_corr[:, 0][mask]
        dp_a = dphi_approx[:, 0][mask]

        results["by_mode"][name] = {
            "n_sides": int(mask.sum()),
            "fraction_of_cohort": float(mask.sum() / len(ids)),
            "theta_tk_mrad": {
                "median": float(np.median(th_tk)),
                "p68": float(np.percentile(th_tk, 68)),
                "p90": float(np.percentile(th_tk, 90)),
                "p95": float(np.percentile(th_tk, 95)),
            },
            "theta_tau_t_mrad": {
                "median": float(np.median(th_tau_t)),
                "p68": float(np.percentile(th_tau_t, 68)),
                "p90": float(np.percentile(th_tau_t, 90)),
            },
            "track_frame": stats(dp_t),
            "naive_vis_frame": stats(dp_k),
            "parallax_corrected": stats(dp_c),
            "reco_approx_corrected": stats(dp_a),
        }
        print(f"=== {name} (N={mask.sum()}) ===")
        print(f"  theta(track, k): median = {np.median(th_tk):.2f} mrad, 90% = {np.percentile(th_tk, 90):.2f} mrad")
        print(f"  Track frame:      mean cos = {np.mean(np.cos(dp_t)):.4f}, med |dphi| = {np.median(np.abs(dp_t)):.4f} rad, <30deg = {np.mean(np.abs(dp_t)<np.pi/6)*100:.1f}%")
        print(f"  Naive vis frame:  mean cos = {np.mean(np.cos(dp_k)):.4f}, med |dphi| = {np.median(np.abs(dp_k)):.4f} rad, <30deg = {np.mean(np.abs(dp_k)<np.pi/6)*100:.1f}%")
        print(f"  Parallax corr:    mean cos = {np.mean(np.cos(dp_c)):.4f}, med |dphi| = {np.median(np.abs(dp_c)):.4f} rad, <30deg = {np.mean(np.abs(dp_c)<np.pi/6)*100:.1f}%")
        print(f"  Reco approx corr: mean cos = {np.mean(np.cos(dp_a)):.4f}, med |dphi| = {np.median(np.abs(dp_a)):.4f} rad, <30deg = {np.mean(np.abs(dp_a)<np.pi/6)*100:.1f}%")

    # Upsilon breakdown for rho
    mask_rho = (modes[:, 0] == 1) & (tot_e[:, 0] > 0) & np.isfinite(d0[:, 0]) & np.isfinite(tau_hat[:, 0, 0])
    u_bins = np.linspace(-1.0, 1.0, 11)
    for lo, hi in zip(u_bins[:-1], u_bins[1:]):
        bmask = mask_rho & (upsilon[:, 0] >= lo) & (upsilon[:, 0] < hi)
        if bmask.sum() == 0:
            continue
        entry = {
            "upsilon_range": [float(lo), float(hi)],
            "n": int(bmask.sum()),
            "theta_tk_mrad_median": float(np.median(theta_tk_mrad[:, 0][bmask])),
            "track_frame": stats(dphi_track[:, 0][bmask]),
            "naive_vis_frame": stats(dphi_vis[:, 0][bmask]),
            "parallax_corrected": stats(dphi_corr[:, 0][bmask]),
            "reco_approx_corrected": stats(dphi_approx[:, 0][bmask]),
        }
        results["rho_by_upsilon"].append(entry)

    # Opening angle theta_tk breakdown for rho
    tk_edges = [0, 2, 4, 6, 8, 12, 16, 25, 100]
    for lo, hi in zip(tk_edges[:-1], tk_edges[1:]):
        bmask = mask_rho & (theta_tk_mrad[:, 0] >= lo) & (theta_tk_mrad[:, 0] < hi)
        if bmask.sum() == 0:
            continue
        entry = {
            "theta_tk_mrad_range": [float(lo), float(hi)],
            "n": int(bmask.sum()),
            "track_frame": stats(dphi_track[:, 0][bmask]),
            "naive_vis_frame": stats(dphi_vis[:, 0][bmask]),
            "parallax_corrected": stats(dphi_corr[:, 0][bmask]),
        }
        results["rho_by_theta_tk"].append(entry)

    # IP magnitude breakdown for rho
    ip_mag_um = np.linalg.norm(ip_pv[:, 0], axis=-1) * 1e3
    ip_edges = [0, 20, 40, 80, 160, 320, 1e6]
    for lo, hi in zip(ip_edges[:-1], ip_edges[1:]):
        bmask = mask_rho & (ip_mag_um >= lo) & (ip_mag_um < hi)
        if bmask.sum() == 0:
            continue
        entry = {
            "ip_um_range": [float(lo), float(hi)],
            "n": int(bmask.sum()),
            "track_frame": stats(dphi_track[:, 0][bmask]),
            "naive_vis_frame": stats(dphi_vis[:, 0][bmask]),
            "parallax_corrected": stats(dphi_corr[:, 0][bmask]),
        }
        results["rho_by_ip_um"].append(entry)

    out_file = args.output_dir / "audit_summary.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_file}", flush=True)

    # Also save arrays for plotting
    np.savez_compressed(
        args.output_dir / "audit_arrays.npz",
        modes=modes[:, 0],
        upsilon=upsilon[:, 0],
        theta_tk_mrad=theta_tk_mrad[:, 0],
        theta_tau_t_mrad=theta_tau_t_mrad[:, 0],
        ip_mag_um=ip_mag_um,
        dphi_track=dphi_track[:, 0],
        dphi_vis=dphi_vis[:, 0],
        dphi_corr=dphi_corr[:, 0],
        dphi_approx=dphi_approx[:, 0],
        labels=labels,
        weights=weights,
    )
    print("Audit arrays saved successfully.", flush=True)


if __name__ == "__main__":
    main()
