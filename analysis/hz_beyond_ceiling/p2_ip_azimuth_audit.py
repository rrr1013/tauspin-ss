"""Audit how well the impact parameter measures the tau azimuth around its track.

For a one-prong decay the tau flies PV -> DV and the charged track leaves DV
along t.  The point of closest approach of the track line to the PV is
PV + L * (tau_hat - (tau_hat.t) t), i.e. the 3-D impact-parameter vector points
along the component of the tau direction perpendicular to the track.  Its
direction is the tau azimuth around the track -- exactly the degree of freedom
that visible momenta + MET + mass shells leave undetermined.

Two IP definitions are compared for the same tracks:
  * ``legacy``: (-d0 sin phi, d0 cos phi, z0 sin theta) with z0 relative to the
    beam spot, as used by the 2026-09-20 geometry studies;
  * ``pv``: perigee point rebuilt in global coordinates and referenced to the
    reconstructed primary vertex, with the along-track component removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--geometry-audit", type=Path, default=Path(
        "/home/rbaba/tauspin-ip-sv-analysis-corrected-20260920/geometry_audit.npz"))
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - np.sum(v * axis, axis=-1, keepdims=True) * axis


def signed_azimuth(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Angle from a to b around axis (both already perpendicular to axis)."""
    return np.arctan2(np.sum(np.cross(a, b) * axis, axis=-1), np.sum(a * b, axis=-1))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trk = np.load(args.tracks)
    with np.load(args.geometry_audit) as g:
        truth_dir = np.asarray(g["truth_tau_direction"])
        labels = np.asarray(g["labels"])
    pv = trk["pv"]
    t = trk["trk"]  # rows, side, core, (pt, theta, phi, q, d0, z0, npix)
    n_core = trk["n_core"]

    # Beam-spot reference: MC uses a fixed luminous-region centre.
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    bxy = np.median(pv[:, :2], axis=0)
    dz = (lead[..., 5] - pv[:, None, 2])[ok]
    bz = -float(np.median(dz))  # z0 = z_pca - bz and z_pca ~ PVz on average
    beam = np.array([bxy[0], bxy[1], bz])

    theta, phi, d0, z0 = t[..., 1], t[..., 2], t[..., 4], t[..., 5]
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip_pv = perp(pca - pv[:, None, None, :], tdir)
    ip_legacy = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0 * np.sin(theta)], -1)
    ip_legacy_perp = perp(ip_legacy, tdir)

    tau_perp = perp(np.broadcast_to(truth_dir[:, :, None, :], tdir.shape), tdir)
    cone = np.arccos(np.clip(np.sum(truth_dir[:, :, None, :] * tdir, -1), -1, 1))
    valid = np.isfinite(truth_dir).all(-1)[:, :, None] & np.isfinite(d0)
    one_prong = (n_core == 1)[:, :, None] & valid
    one_prong[:, :, 1:] = False
    three_prong = (n_core == 3)[:, :, None] & valid

    dphi_pv = signed_azimuth(unit(tau_perp), unit(ip_pv), tdir)
    dphi_legacy = signed_azimuth(unit(tau_perp), unit(ip_legacy_perp), tdir)
    ipmag = np.linalg.norm(ip_pv, axis=-1) * 1e3  # um

    def stats(x: np.ndarray) -> dict:
        a = np.abs(x)
        return {"n": int(x.size), "median_abs_rad": float(np.median(a)),
                "frac_within_30deg": float(np.mean(a < np.pi / 6)),
                "frac_within_90deg": float(np.mean(a < np.pi / 2)),
                "mean_cos": float(np.mean(np.cos(x)))}

    report = {"beam_reference_mm": beam.tolist(),
              "one_prong": {"pv": stats(dphi_pv[one_prong]), "legacy": stats(dphi_legacy[one_prong])},
              "three_prong_tracks": {"pv": stats(dphi_pv[three_prong]), "legacy": stats(dphi_legacy[three_prong])}}
    # dependence on IP magnitude (one prong)
    edges = np.array([0, 10, 20, 40, 80, 160, 320, 1e9])
    bins = []
    m1 = ipmag[one_prong]
    d1 = dphi_pv[one_prong]
    c1 = cone[one_prong]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (m1 >= lo) & (m1 < hi)
        bins.append({"ip_um": [float(lo), float(hi)], "fraction": float(s.mean()),
                     **stats(d1[s]), "median_cone_mrad": float(np.median(c1[s]) * 1e3)})
    report["one_prong_by_ip_um"] = bins
    # implied transverse tau-direction error if the cone angle were known: cone*|dphi| (small angle)
    implied = c1 * 2 * np.sin(np.abs(d1) / 2) * 1e3
    report["one_prong_implied_direction_error_mrad"] = {
        "median": float(np.median(implied)), "p68": float(np.percentile(implied, 68)),
        "median_cone_mrad": float(np.median(c1) * 1e3)}
    (args.output_dir / "p2_ip_azimuth.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    b = np.linspace(-np.pi, np.pi, 73)
    ax[0].hist(dphi_legacy[one_prong], b, histtype="step", lw=1.6, ls="--", color="0.4",
               density=True, label="legacy IP (z0 w.r.t. beam spot)")
    ax[0].hist(dphi_pv[one_prong], b, histtype="step", lw=1.8, color="C0", density=True,
               label="PV-referenced 3-D IP")
    ax[0].axhline(1 / (2 * np.pi), color="k", lw=0.8, ls=":", label="uniform (no information)")
    ax[0].set_xlabel(r"$\Delta\phi$ = azimuth(IP) $-$ azimuth(truth $\hat\tau_\perp$) around track [rad]")
    ax[0].set_ylabel("density (unit area)")
    ax[0].set_title("1-prong reco taus, leading core track")
    ax[0].legend(fontsize=8, loc="upper left")
    centers = [np.sqrt(max(x["ip_um"][0], 5) * min(x["ip_um"][1], 640)) for x in bins]
    ax[1].plot(centers, [x["median_abs_rad"] for x in bins], "o-", color="C0", label=r"median $|\Delta\phi|$")
    ax[1].plot(centers, [x["frac_within_30deg"] for x in bins], "s--", color="C1", label=r"frac $|\Delta\phi|<30^\circ$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$|$IP$_{PV}|$ [$\mu$m]")
    ax[1].set_title("PV-referenced IP: azimuth quality vs IP size")
    ax[1].legend(fontsize=8)
    ax2 = ax[1].twinx()
    ax2.bar(centers, [x["fraction"] for x in bins], width=np.array(centers) * 0.35, alpha=0.2, color="0.3")
    ax2.set_ylabel("fraction of 1-prong sides (bars)")
    ax[2].hist(np.clip(implied, 0, 20), np.linspace(0, 20, 81), histtype="step", color="C0", lw=1.8,
               density=True, label=r"cone$\times|\Delta\phi|$ (IP azimuth)")
    ax[2].axvline(4.55, color="C3", ls="--", label="3p SV core width 4.55 mrad (09-22)")
    ax[2].set_xlabel("implied transverse tau-direction error [mrad] (overflow in last bin)")
    ax[2].set_ylabel("density")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "p2_ip_azimuth.png", dpi=140)


if __name__ == "__main__":
    main()
