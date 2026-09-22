"""Measure the event-specific sheet information carried by reconstructed SVs.

The kinematic surface is deliberately kept at truth-visible + exact tau-MET in
this work packet.  That isolates the detector geometry channel from the much
larger reconstructed-surface failure measured by WP-A.  The SV response is fit
only on a calibration partition of the training cohort.  A disjoint training
partition is then used for the information test; validation and test are not
read.

Each decay-mode response is an isotropic von-Mises--Fisher core plus a uniform
outlier component.  Both mixture fraction and concentration are fitted by EM,
so no angular quality cut or hand-picked Gaussian width enters the selector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_NAMES = {0: "1p0n", 1: "rho", 3: "3p0n"}
MODE_INDEX = {0: 0, 1: 1, 3: 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--posterior-calibration-events", type=int, default=10_000)
    parser.add_argument("--proposal-lattice", type=int, default=64)
    parser.add_argument("--keep", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def log_vmf_ratio(cosine: np.ndarray, kappa: float) -> np.ndarray:
    """log[vMF(direction|truth)/Uniform(S2)] with stable large-kappa algebra."""
    if kappa <= 1.0e-10:
        return np.zeros_like(cosine, dtype=np.float64)
    if kappa < 40.0:
        log_norm = math.log(kappa) - math.log(math.sinh(kappa))
        return log_norm + kappa * cosine
    return math.log(2.0 * kappa) + kappa * (cosine - 1.0)


def solve_kappa(resultant: float) -> float:
    """Solve coth(kappa)-1/kappa=resultant for a 3-D vMF distribution."""
    r = float(np.clip(resultant, 0.0, 1.0 - 1.0e-12))
    if r < 1.0e-8:
        return 0.0
    low, high = 0.0, max(10.0, 2.0 / max(1.0 - r, 1.0e-12))
    for _ in range(100):
        mid = 0.5 * (low + high)
        if mid < 20.0:
            value = 1.0 / math.tanh(mid) - 1.0 / mid
        else:
            value = 1.0 - 1.0 / mid
        if value < r:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def fit_vmf_uniform(cosine: np.ndarray) -> dict[str, float | int]:
    values = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    if len(values) < 50:
        return {"events": int(len(values)), "pi_core": 0.0, "kappa": 0.0,
                "effective_sigma_mrad": None, "log_likelihood": 0.0}
    pi = 0.9
    kappa = solve_kappa(max(float(values.mean()), 0.0))
    for _ in range(100):
        log_ratio = log_vmf_ratio(values, kappa)
        logit = math.log(max(pi, 1.0e-12)) - math.log(max(1.0 - pi, 1.0e-12)) + log_ratio
        responsibility = np.empty_like(logit)
        positive = logit >= 0.0
        responsibility[positive] = 1.0 / (1.0 + np.exp(-np.minimum(logit[positive], 700.0)))
        exp_logit = np.exp(np.maximum(logit[~positive], -700.0))
        responsibility[~positive] = exp_logit / (1.0 + exp_logit)
        new_pi = float(np.clip(responsibility.mean(), 1.0e-6, 1.0 - 1.0e-6))
        resultant = float(np.sum(responsibility * values) / np.maximum(responsibility.sum(), 1.0e-12))
        new_kappa = solve_kappa(resultant)
        if abs(new_pi - pi) < 1.0e-10 and abs(new_kappa - kappa) / max(kappa, 1.0) < 1.0e-9:
            pi, kappa = new_pi, new_kappa
            break
        pi, kappa = new_pi, new_kappa
    log_ratio = log_vmf_ratio(values, kappa)
    log_density_ratio = np.logaddexp(math.log1p(-pi), math.log(pi) + log_ratio)
    sigma = 1.0 / math.sqrt(kappa) if kappa > 0.0 else None
    return {
        "events": int(len(values)),
        "pi_core": pi,
        "kappa": kappa,
        "effective_sigma_mrad": sigma * 1.0e3 if sigma is not None else None,
        "log_likelihood": float(log_density_ratio.sum()),
    }


def mixture_log_likelihood(cosine: np.ndarray, model: dict[str, float | int]) -> np.ndarray:
    pi = float(model["pi_core"])
    kappa = float(model["kappa"])
    if pi <= 0.0 or kappa <= 0.0:
        return np.zeros_like(cosine, dtype=np.float64)
    log_ratio = log_vmf_ratio(np.clip(cosine, -1.0, 1.0), kappa)
    return np.logaddexp(math.log1p(-pi), math.log(pi) + log_ratio)


def fit_halfnormal_uniform(residual: np.ndarray) -> dict[str, float | int | None]:
    """Fit |normal·truth direction| with a half-normal core plus uniform tail."""
    values = np.clip(np.asarray(residual, dtype=np.float64), 0.0, 1.0)
    if len(values) < 50:
        return {"events": int(len(values)), "pi_core": 0.0, "sigma": None,
                "log_likelihood": 0.0}
    pi = 0.9
    sigma = max(float(np.sqrt(np.mean(values ** 2))), 1.0e-6)
    for _ in range(100):
        log_core = (0.5 * math.log(2.0 / math.pi) - math.log(sigma)
                    - 0.5 * (values / sigma) ** 2)
        logit = math.log(max(pi, 1.0e-12)) - math.log(max(1.0 - pi, 1.0e-12)) + log_core
        responsibility = np.empty_like(logit)
        positive = logit >= 0.0
        responsibility[positive] = 1.0 / (1.0 + np.exp(-np.minimum(logit[positive], 700.0)))
        exp_logit = np.exp(np.maximum(logit[~positive], -700.0))
        responsibility[~positive] = exp_logit / (1.0 + exp_logit)
        new_pi = float(np.clip(responsibility.mean(), 1.0e-6, 1.0 - 1.0e-6))
        new_sigma = float(np.sqrt(
            np.sum(responsibility * values ** 2) / np.maximum(responsibility.sum(), 1.0e-12)
        ))
        new_sigma = max(new_sigma, 1.0e-6)
        if abs(new_pi - pi) < 1.0e-10 and abs(new_sigma - sigma) / sigma < 1.0e-9:
            pi, sigma = new_pi, new_sigma
            break
        pi, sigma = new_pi, new_sigma
    log_core = (0.5 * math.log(2.0 / math.pi) - math.log(sigma)
                - 0.5 * (values / sigma) ** 2)
    log_density = np.logaddexp(math.log1p(-pi), math.log(pi) + log_core)
    return {"events": int(len(values)), "pi_core": pi, "sigma": sigma,
            "log_likelihood": float(log_density.sum())}


def ip_log_likelihood(residual: np.ndarray, model: dict[str, float | int | None]) -> np.ndarray:
    pi = float(model["pi_core"])
    sigma_value = model["sigma"]
    if pi <= 0.0 or sigma_value is None:
        return np.zeros_like(residual, dtype=np.float64)
    sigma = float(sigma_value)
    log_core = (0.5 * math.log(2.0 / math.pi) - math.log(sigma)
                - 0.5 * (np.clip(residual, 0.0, 1.0) / sigma) ** 2)
    return np.logaddexp(math.log1p(-pi), math.log(pi) + log_core)


def fit_log_length_response(log_length_over_p: np.ndarray) -> dict[str, float | int]:
    """Robust Student-t response for log(SV flight length / tau momentum)."""
    values = np.asarray(log_length_over_p, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 50:
        return {"events": int(len(values)), "location": 0.0, "scale": 1.0, "df": 3.0}
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = max(1.4826 * mad, 1.0e-3)
    return {"events": int(len(values)), "location": location, "scale": scale, "df": 3.0}


def length_log_likelihood(length: np.ndarray, momentum: np.ndarray,
                          model: dict[str, float | int]) -> np.ndarray:
    if int(model["events"]) < 50:
        return np.zeros_like(momentum, dtype=np.float64)
    residual = np.log(np.maximum(length, 1.0e-12)) - np.log(np.maximum(momentum, 1.0e-12))
    z = (residual - float(model["location"])) / float(model["scale"])
    df = float(model["df"])
    return -0.5 * (df + 1.0) * np.log1p(z ** 2 / df)


def opening_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def unit_from_eta_phi(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    value = np.stack((np.cos(phi), np.sin(phi), np.sinh(eta)), axis=-1)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1.0e-300)


def matched_direction_shuffle(direction: np.ndarray, available: np.ndarray, modes: np.ndarray,
                              tau_pt: np.ndarray, tau_eta: np.ndarray, tau_phi: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Shuffle SV directions within coarse topology and reco-direction cells."""
    out = np.array(direction, copy=True)
    log_pt = np.log(np.maximum(tau_pt, 1.0))
    pt_edges = np.quantile(log_pt[available], [0.0, 0.25, 0.5, 0.75, 1.0]) if available.any() \
        else np.array([0, 1, 2, 3, 4], dtype=float)
    eta_edges = np.array([-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf])
    phi_edges = np.linspace(-np.pi, np.pi, 9)
    pt_bin = np.clip(np.digitize(log_pt, pt_edges[1:-1]), 0, 3)
    eta_bin = np.clip(np.digitize(tau_eta, eta_edges[1:-1]), 0, 4)
    phi_wrapped = (tau_phi + np.pi) % (2.0 * np.pi) - np.pi
    phi_bin = np.clip(np.digitize(phi_wrapped, phi_edges[1:-1]), 0, 7)
    for side in (0, 1):
        for mode in MODE_NAMES:
            for p in range(4):
                for e in range(5):
                    for a in range(8):
                        group = np.flatnonzero(
                            available[:, side]
                            & (modes[:, side] == mode)
                            & (pt_bin[:, side] == p)
                            & (eta_bin[:, side] == e)
                            & (phi_bin[:, side] == a)
                        )
                        if len(group) > 1:
                            out[group, side] = direction[rng.permutation(group), side]
    return out


def matched_scalar_shuffle(value: np.ndarray, available: np.ndarray, modes: np.ndarray,
                           tau_pt: np.ndarray, tau_eta: np.ndarray, tau_phi: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Shuffle a per-side scalar in the same matching cells as the vector control."""
    out = np.array(value, copy=True)
    log_pt = np.log(np.maximum(tau_pt, 1.0))
    pt_edges = np.quantile(log_pt[available], [0.0, 0.25, 0.5, 0.75, 1.0]) if available.any() \
        else np.array([0, 1, 2, 3, 4], dtype=float)
    eta_edges = np.array([-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf])
    phi_edges = np.linspace(-np.pi, np.pi, 9)
    pt_bin = np.clip(np.digitize(log_pt, pt_edges[1:-1]), 0, 3)
    eta_bin = np.clip(np.digitize(tau_eta, eta_edges[1:-1]), 0, 4)
    phi_bin = np.clip(np.digitize((tau_phi + np.pi) % (2.0 * np.pi) - np.pi,
                                  phi_edges[1:-1]), 0, 7)
    for side in (0, 1):
        for mode in MODE_NAMES:
            for p in range(4):
                for e in range(5):
                    for a in range(8):
                        group = np.flatnonzero(
                            available[:, side] & (modes[:, side] == mode)
                            & (pt_bin[:, side] == p) & (eta_bin[:, side] == e)
                            & (phi_bin[:, side] == a)
                        )
                        if len(group) > 1:
                            out[group, side] = value[rng.permutation(group), side]
    return out


def logmeanexp(values: np.ndarray, valid: np.ndarray, axis: int = 0) -> np.ndarray:
    masked = np.where(valid, values, -np.inf)
    maximum = np.max(masked, axis=axis)
    safe_max = np.where(np.isfinite(maximum), maximum, 0.0)
    total = np.sum(np.where(valid, np.exp(masked - np.expand_dims(safe_max, axis)), 0.0), axis=axis)
    count = np.sum(valid, axis=axis)
    result = np.full_like(maximum, -np.inf, dtype=np.float64)
    present = count > 0
    result[present] = safe_max[present] + np.log(total[present] / count[present])
    return result


def posterior_from_log_evidence(log_evidence: np.ndarray) -> np.ndarray:
    finite_row = np.isfinite(log_evidence).any(axis=1)
    maximum = np.zeros((len(log_evidence), 1), dtype=np.float64)
    maximum[finite_row] = np.max(log_evidence[finite_row], axis=1, keepdims=True)
    shifted = np.full_like(log_evidence, -np.inf, dtype=np.float64)
    shifted[finite_row] = log_evidence[finite_row] - maximum[finite_row]
    weight = np.where(np.isfinite(shifted), np.exp(np.maximum(shifted, -700.0)), 0.0)
    posterior = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1.0e-300)
    posterior[~finite_row] = 0.25
    return posterior


def temperature_posterior(log_evidence: np.ndarray, temperature: float) -> np.ndarray:
    return posterior_from_log_evidence(log_evidence / max(float(temperature), 1.0e-12))


def fit_temperature(log_evidence: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    """Fit one scalar temperature on a held-out training calibration partition."""
    evidence = log_evidence[valid]
    target = truth[valid]
    if not len(target):
        return {"events": 0, "temperature": 1.0, "nll_before": math.nan, "nll_after": math.nan}

    def objective(log_temperature: float) -> float:
        posterior = temperature_posterior(evidence, math.exp(log_temperature))
        p_true = posterior[np.arange(len(target)), target]
        return float(-np.log(np.maximum(p_true, 1.0e-300)).mean())

    low, high = -4.0, 4.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = high - ratio * (high - low)
    x2 = low + ratio * (high - low)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(80):
        if f1 > f2:
            low, x1, f1 = x1, x2, f2
            x2 = low + ratio * (high - low)
            f2 = objective(x2)
        else:
            high, x2, f2 = x2, x1, f1
            x1 = high - ratio * (high - low)
            f1 = objective(x1)
    best = 0.5 * (low + high)
    return {
        "events": int(len(target)),
        "temperature": float(math.exp(best)),
        "nll_before": objective(0.0),
        "nll_after": objective(best),
    }


def evaluate_rows(selected: np.ndarray, source: dict[str, np.ndarray],
                  geometry: dict[str, np.ndarray], response: dict[str, dict[int, dict]],
                  ms: object, ss: object, args: argparse.Namespace,
                  shuffle_seed: int) -> dict[str, np.ndarray]:
    """Integrate actual and matched-shuffle SV likelihood over all four sheets."""
    labels = source["labels"][selected].astype(np.int8)
    global_ids = source["global_indices"].astype(np.int64)
    ids = global_ids[selected]
    truth_modes = source["modes"][selected].astype(np.int8)
    reco_modes = source["reco_h_mode"][selected].astype(np.int8)
    visible = source["truth_visible_tau_lab4"][selected].astype(np.float64)
    truth_nu = source["truth_neutrino_lab4"][selected, ..., :3].astype(np.float64)
    met = truth_nu[:, 0, :2] + truth_nu[:, 1, :2]
    truth_sheet = ss.truth_sheet(visible, truth_nu)["sheet"].astype(np.int8)
    available = geometry["sv_available"][ids] > 0
    measured = geometry["sv_direction"][ids].astype(np.float64)
    tau_pt = geometry["tau_pt"][ids]
    tau_eta = geometry["tau_eta"][ids]
    tau_phi = geometry["tau_phi"][ids]
    shuffled = matched_direction_shuffle(
        measured, available, reco_modes, tau_pt, tau_eta, tau_phi,
        np.random.default_rng(shuffle_seed),
    )
    measured_length = geometry["sv_length"][ids].astype(np.float64)
    shuffled_length = matched_scalar_shuffle(
        measured_length, available, reco_modes, tau_pt, tau_eta, tau_phi,
        np.random.default_rng(shuffle_seed + 11),
    )
    reco_tau_direction = unit_from_eta_phi(tau_eta, tau_phi)
    impact_direction = geometry["ip_direction"][ids].astype(np.float64)
    ip_normal = np.cross(reco_tau_direction, impact_direction)
    ip_norm = np.linalg.norm(ip_normal, axis=-1, keepdims=True)
    ip_available = (geometry["ip_track_count"][ids] > 0) & (ip_norm[..., 0] > 1.0e-12)
    ip_normal /= np.maximum(ip_norm, 1.0e-300)
    shuffled_ip_normal = matched_direction_shuffle(
        ip_normal, ip_available, reco_modes, tau_pt, tau_eta, tau_phi,
        np.random.default_rng(shuffle_seed + 19),
    )

    n = len(selected)
    evidence = {
        name: np.full((n, 4), -np.inf)
        for name in (
            "sv_actual", "sv_shuffle", "ip_actual", "ip_shuffle",
            "length_actual", "length_shuffle", "sv_length_actual", "sv_length_shuffle",
            "all_geometry_actual", "all_geometry_shuffle",
        )
    }
    retained = np.zeros(n, dtype=np.int16)
    rng = np.random.default_rng(shuffle_seed + 1)
    for start in range(0, n, args.row_chunk):
        stop = min(start + args.row_chunk, n)
        block = np.arange(start, stop)
        vis = visible[block]
        met_block = met[block]
        unit = ms.lattice(args.proposal_lattice, len(block), rng).transpose(1, 0, 2)
        drawn = ms.sample_region(vis, met_block, unit)
        picked = ms.compact(drawn["accepted"], args.keep, rng)
        retained[block] = np.sum(picked >= 0, axis=0).astype(np.int16)
        nu_t = np.take_along_axis(drawn["nu_t"], np.maximum(picked, 0)[..., None], axis=0)
        solved = ms.solution_sheets(vis, met_block, nu_t)
        valid = solved["valid"] & (picked >= 0)[:, None, :]
        tau = vis[None, None, ..., :3] + solved["nu"]
        tau /= np.maximum(np.linalg.norm(tau, axis=-1, keepdims=True), 1.0e-300)

        channel_log_like: dict[str, np.ndarray] = {}
        for channel, measurement, availability, models in (
            ("sv_actual", measured[block], available[block], response["sv"]),
            ("sv_shuffle", shuffled[block], available[block], response["sv"]),
        ):
            log_like = np.zeros(valid.shape, dtype=np.float64)
            for side in (0, 1):
                cosine = np.sum(tau[..., side, :] * measurement[None, None, :, side, :], axis=-1)
                for mode, model in models.items():
                    rows = (reco_modes[block, side] == mode) & availability[:, side]
                    if rows.any():
                        log_like[..., rows] += mixture_log_likelihood(cosine[..., rows], model)
            channel_log_like[channel] = log_like
        tau_momentum = np.linalg.norm(vis[None, None, ..., :3] + solved["nu"], axis=-1)
        for channel, measurement in (
            ("length_actual", measured_length[block]),
            ("length_shuffle", shuffled_length[block]),
        ):
            log_like = np.zeros(valid.shape, dtype=np.float64)
            for side in (0, 1):
                for mode, model in response["length"].items():
                    rows = (reco_modes[block, side] == mode) & available[block, side]
                    if rows.any():
                        log_like[..., rows] += length_log_likelihood(
                            measurement[None, None, rows, side],
                            tau_momentum[..., rows, side], model,
                        )
            channel_log_like[channel] = log_like
        for channel, measurement, availability, models in (
            ("ip_actual", ip_normal[block], ip_available[block], response["ip"]),
            ("ip_shuffle", shuffled_ip_normal[block], ip_available[block], response["ip"]),
        ):
            log_like = np.zeros(valid.shape, dtype=np.float64)
            for side in (0, 1):
                residual = np.abs(np.sum(
                    tau[..., side, :] * measurement[None, None, :, side, :], axis=-1
                ))
                for mode, model in models.items():
                    rows = (reco_modes[block, side] == mode) & availability[:, side]
                    if rows.any():
                        log_like[..., rows] += ip_log_likelihood(residual[..., rows], model)
            channel_log_like[channel] = log_like
        channel_log_like["sv_length_actual"] = (
            channel_log_like["sv_actual"] + channel_log_like["length_actual"]
        )
        channel_log_like["sv_length_shuffle"] = (
            channel_log_like["sv_shuffle"] + channel_log_like["length_shuffle"]
        )
        channel_log_like["all_geometry_actual"] = (
            channel_log_like["sv_actual"] + channel_log_like["length_actual"]
            + channel_log_like["ip_actual"]
        )
        channel_log_like["all_geometry_shuffle"] = (
            channel_log_like["sv_shuffle"] + channel_log_like["length_shuffle"]
            + channel_log_like["ip_shuffle"]
        )
        for channel, log_like in channel_log_like.items():
            for sheet in range(4):
                evidence[channel][block, sheet] = logmeanexp(
                    log_like[:, sheet], valid[:, sheet], axis=0
                )

    valid_surface = np.isfinite(evidence["sv_actual"]).all(axis=1) & (retained > 0)
    output = {
        "global_indices": ids, "labels": labels, "truth_modes": truth_modes,
        "reco_modes": reco_modes,
        "truth_sheet": truth_sheet, "available": available, "ip_available": ip_available,
        "valid_surface": valid_surface, "retained": retained,
    }
    output.update({f"log_evidence_{name}": value for name, value in evidence.items()})
    return output


def metric_block(posterior: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    p = posterior[mask]
    y = truth[mask]
    if not len(y):
        return {"events": 0}
    p_true = p[np.arange(len(y)), y]
    onehot = np.eye(4)[y]
    prediction = np.argmax(p, axis=1)
    recalls = []
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    for cls in range(4):
        cls_mask = y == cls
        recalls.append(float((prediction[cls_mask] == cls).mean()) if cls_mask.any() else None)
    confidence = np.max(p, axis=1)
    correct = prediction == y
    reliability = []
    for low, high in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        in_bin = (confidence >= low) & ((confidence < high) | ((high == 1.0) & (confidence <= high)))
        reliability.append({
            "low": float(low), "high": float(high), "events": int(in_bin.sum()),
            "mean_confidence": float(confidence[in_bin].mean()) if in_bin.any() else None,
            "accuracy": float(correct[in_bin].mean()) if in_bin.any() else None,
        })
    selective_accuracy = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        selected = confidence >= threshold
        selective_accuracy.append({
            "confidence_threshold": threshold,
            "events": int(selected.sum()),
            "coverage": float(selected.mean()),
            "accuracy": float(correct[selected].mean()) if selected.any() else None,
        })
    return {
        "events": int(len(y)),
        "nll": float(-np.log(np.maximum(p_true, 1.0e-300)).mean()),
        "brier": float(np.sum((p - onehot) ** 2, axis=1).mean()),
        "top1_accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean([r for r in recalls if r is not None])),
        "per_class_recall": recalls,
        "mean_truth_probability": float(p_true.mean()),
        "median_truth_probability": float(np.median(p_true)),
        "confusion": confusion.tolist(),
        "reliability": reliability,
        "selective_accuracy": selective_accuracy,
    }


def fit_topology_prior(reco_modes: np.ndarray, sv_available: np.ndarray,
                       truth_sheet: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """P(sheet | reco mode pair, SV-availability pattern) with Laplace smoothing."""
    table = np.full((3, 3, 2, 2, 4), float(smoothing), dtype=np.float64)
    mode_index = np.vectorize(MODE_INDEX.__getitem__)(reco_modes)
    for row in range(len(truth_sheet)):
        table[mode_index[row, 0], mode_index[row, 1],
              int(sv_available[row, 0]), int(sv_available[row, 1]),
              int(truth_sheet[row])] += 1.0
    table /= table.sum(axis=-1, keepdims=True)
    return table


def apply_topology_prior(table: np.ndarray, reco_modes: np.ndarray,
                         sv_available: np.ndarray) -> np.ndarray:
    mode_index = np.vectorize(MODE_INDEX.__getitem__)(reco_modes)
    return table[mode_index[:, 0], mode_index[:, 1],
                 sv_available[:, 0].astype(int), sv_available[:, 1].astype(int)]


def paired_mean_bootstrap(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                          draws: int = 5000, seed: int = 20260922) -> dict[str, float | int]:
    difference = (a - b)[mask]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        samples[draw] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {
        "events": int(len(difference)), "difference": float(difference.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_sign": float(2.0 * min((samples <= 0).mean(), (samples >= 0).mean())),
        "draws": draws, "seed": seed,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.module_dir))
    import metspace as ms  # type: ignore
    import sheetspace as ss  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs, allow_pickle=False) as data:
        needed = ("labels", "global_indices", "modes", "reco_h_mode", "truth_visible_tau_lab4",
                  "truth_neutrino_lab4")
        source = {name: np.asarray(data[name]) for name in needed}
    with np.load(args.geometry, allow_pickle=False) as data:
        geometry = {name: np.asarray(data[name]) for name in (
            "global_indices", "sv_available", "sv_direction", "sv_delta_theta",
            "tau_pt", "tau_eta", "tau_phi", "sv_length", "ip_direction",
            "ip_track_count", "truth_tau_direction",
        )}
    if not np.array_equal(geometry["global_indices"], np.arange(len(geometry["global_indices"]))):
        raise RuntimeError("Geometry audit global indices are not canonical row indices")

    global_ids = source["global_indices"].astype(np.int64)
    hashes = splitmix64(global_ids, args.seed + 17)
    response_rows = (hashes % np.uint64(10)) < np.uint64(4)
    posterior_pool = np.flatnonzero((hashes % np.uint64(10)) == np.uint64(4))
    evaluation_pool = np.flatnonzero((hashes % np.uint64(10)) == np.uint64(5))
    posterior_order = np.argsort(hashes[posterior_pool])
    posterior_selected = posterior_pool[
        posterior_order[:min(args.posterior_calibration_events, len(posterior_order))]
    ]
    evaluation_order = np.argsort(hashes[evaluation_pool])
    selected = evaluation_pool[evaluation_order[:min(args.max_events, len(evaluation_order))]]

    response_ids = global_ids[response_rows]
    response_modes = source["reco_h_mode"][response_rows]
    response_sv: dict[int, dict[str, float | int]] = {}
    response_ip: dict[int, dict[str, float | int | None]] = {}
    response_length: dict[int, dict[str, float | int]] = {}
    response_tau_direction = unit_from_eta_phi(
        geometry["tau_eta"][response_ids], geometry["tau_phi"][response_ids]
    )
    response_ip_direction = geometry["ip_direction"][response_ids]
    response_ip_normal = np.cross(response_tau_direction, response_ip_direction)
    response_ip_norm = np.linalg.norm(response_ip_normal, axis=-1, keepdims=True)
    response_ip_available = ((geometry["ip_track_count"][response_ids] > 0)
                             & (response_ip_norm[..., 0] > 1.0e-12))
    response_ip_normal /= np.maximum(response_ip_norm, 1.0e-300)
    response_truth_tau_p = np.linalg.norm(
        source["truth_visible_tau_lab4"][response_rows, ..., :3]
        + source["truth_neutrino_lab4"][response_rows, ..., :3], axis=-1
    )
    response_truth_sheet = ss.truth_sheet(
        source["truth_visible_tau_lab4"][response_rows].astype(np.float64),
        source["truth_neutrino_lab4"][response_rows, ..., :3].astype(np.float64),
    )["sheet"].astype(np.int8)
    response_sv_available = geometry["sv_available"][response_ids] > 0
    topology_prior = fit_topology_prior(
        response_modes, response_sv_available, response_truth_sheet,
    )
    for mode in MODE_NAMES:
        dots = []
        residuals = []
        log_length_over_p = []
        for side in (0, 1):
            ids = response_ids
            available = geometry["sv_available"][ids, side] > 0
            subset = available & (response_modes[:, side] == mode)
            angle = geometry["sv_delta_theta"][ids[subset], side]
            angle = angle[np.isfinite(angle)]
            dots.append(np.cos(angle))
            ip_subset = response_ip_available[:, side] & (response_modes[:, side] == mode)
            residuals.append(np.abs(np.sum(
                response_ip_normal[ip_subset, side]
                * geometry["truth_tau_direction"][ids[ip_subset], side], axis=-1
            )))
            length_subset = available & (response_modes[:, side] == mode)
            log_length_over_p.append(
                np.log(np.maximum(geometry["sv_length"][ids[length_subset], side], 1.0e-12))
                - np.log(np.maximum(response_truth_tau_p[length_subset, side], 1.0e-12))
            )
        response_sv[mode] = fit_vmf_uniform(np.concatenate(dots) if dots else np.empty(0))
        response_ip[mode] = fit_halfnormal_uniform(
            np.concatenate(residuals) if residuals else np.empty(0)
        )
        response_length[mode] = fit_log_length_response(
            np.concatenate(log_length_over_p) if log_length_over_p else np.empty(0)
        )
    response = {"sv": response_sv, "ip": response_ip, "length": response_length}

    posterior_data = evaluate_rows(
        posterior_selected, source, geometry, response, ms, ss, args, args.seed + 101,
    )
    temperature_fit = {}
    for channel, availability in (
        ("sv", posterior_data["available"].any(axis=1)),
        ("ip", posterior_data["ip_available"].any(axis=1)),
        ("length", posterior_data["available"].any(axis=1)),
        ("sv_length", posterior_data["available"].any(axis=1)),
        ("all_geometry", (posterior_data["available"] | posterior_data["ip_available"]).any(axis=1)),
    ):
        calibration_valid = posterior_data["valid_surface"] & availability
        temperature_fit[channel] = fit_temperature(
            posterior_data[f"log_evidence_{channel}_actual"],
            posterior_data["truth_sheet"], calibration_valid,
        )

    evaluated = evaluate_rows(
        selected, source, geometry, response, ms, ss, args, args.seed + 1001,
    )
    ids = evaluated["global_indices"]
    labels = evaluated["labels"]
    truth_modes = evaluated["truth_modes"]
    reco_modes = evaluated["reco_modes"]
    truth_sheet = evaluated["truth_sheet"]
    available = evaluated["available"]
    ip_available = evaluated["ip_available"]
    valid_surface = evaluated["valid_surface"]
    retained = evaluated["retained"]
    n = len(selected)
    uniform = np.full((n, 4), 0.25)
    onehot = np.eye(4)[truth_sheet]

    arms = {"uniform": uniform, "truth_sheet_oracle": onehot}
    arms["topology_availability_prior"] = apply_topology_prior(
        topology_prior, reco_modes, available,
    )
    for channel in ("sv", "ip", "length", "sv_length", "all_geometry"):
        temperature = float(temperature_fit[channel]["temperature"])
        for suffix, evidence_name in (
            ("actual", f"log_evidence_{channel}_actual"),
            ("matched_shuffle", f"log_evidence_{channel}_shuffle"),
        ):
            evidence = evaluated[evidence_name]
            arms[f"{channel}_{suffix}_raw"] = posterior_from_log_evidence(evidence)
            arms[f"{channel}_{suffix}"] = temperature_posterior(evidence, temperature)
    categories: dict[str, np.ndarray] = {"inclusive": valid_surface}
    categories["any_sv"] = valid_surface & available.any(axis=1)
    categories["both_sv"] = valid_surface & available.all(axis=1)
    categories["any_ip"] = valid_surface & ip_available.any(axis=1)
    categories["both_ip"] = valid_surface & ip_available.all(axis=1)
    categories["reco_rho_rho"] = valid_surface & (reco_modes == 1).all(axis=1)
    categories["reco_threeprong_threeprong"] = valid_surface & (reco_modes == 3).all(axis=1)

    metrics = {
        category: {arm: metric_block(posterior, truth_sheet, mask)
                   for arm, posterior in arms.items()}
        for category, mask in categories.items()
    }
    row_index = np.arange(n)
    loss = {
        name: -np.log(np.maximum(posterior[row_index, truth_sheet], 1.0e-300))
        for name, posterior in arms.items()
    }
    nll_comparisons = (
        ("sv_minus_uniform", "sv_actual", "uniform"),
        ("sv_minus_topology", "sv_actual", "topology_availability_prior"),
        ("sv_minus_shuffle", "sv_actual", "sv_matched_shuffle"),
        ("ip_minus_uniform", "ip_actual", "uniform"),
        ("length_minus_uniform", "length_actual", "uniform"),
        ("all_geometry_minus_sv", "all_geometry_actual", "sv_actual"),
    )
    paired_nll = {
        category: {
            name: paired_mean_bootstrap(
                loss[arm_a], loss[arm_b], mask, 5000,
                args.seed + 100 * category_index + comparison_index,
            )
            for comparison_index, (name, arm_a, arm_b) in enumerate(nll_comparisons)
        }
        for category_index, (category, mask) in enumerate(categories.items())
        if mask.any()
    }
    permutation_rng = np.random.default_rng(args.seed + 909)
    permutations = np.argsort(permutation_rng.random((n, 4)), axis=1)
    inverse = np.argsort(permutations, axis=1)
    permuted = np.take_along_axis(arms["sv_actual"], permutations, axis=1)
    permuted_truth = inverse[row_index, truth_sheet]
    permutation_delta = float(np.max(np.abs(
        arms["sv_actual"][valid_surface, truth_sheet[valid_surface]]
        - permuted[valid_surface, permuted_truth[valid_surface]]
    )))
    report = {
        "contract": {
            "split": "train only; disjoint hash response-fit, posterior-calibration, and evaluation partitions",
            "surface": "truth visible + exact tau-neutrino MET",
            "measure": "flat in nuT, integrated separately over four global sheets",
            "response": "mode-conditional vMF core + uniform outlier mixture, fit without angular cuts; one scalar posterior temperature fit on a disjoint partition",
            "ip_response": "mode-conditional half-normal plane-residual core + uniform outlier mixture",
            "length_response": "mode-conditional robust Student-t in log(SV length / tau momentum)",
            "selector_inputs": "measured SV/IP geometry and reconstructed decay-mode category",
            "topology_control": "P(sheet | reco mode pair, SV availability), fit on response partition",
            "test_loaded": False,
            "seed": args.seed,
        },
        "counts": {
            "input_events": int(len(global_ids)),
            "response_fit_events": int(response_rows.sum()),
            "posterior_calibration_events": int(len(posterior_selected)),
            "evaluation_events": int(n),
            "surface_valid": int(valid_surface.sum()),
            "any_sv": int((valid_surface & available.any(axis=1)).sum()),
            "both_sv": int((valid_surface & available.all(axis=1)).sum()),
            "any_ip": int((valid_surface & ip_available.any(axis=1)).sum()),
            "both_ip": int((valid_surface & ip_available.all(axis=1)).sum()),
        },
        "response_models": {
            channel: {MODE_NAMES[key]: value for key, value in models.items()}
            for channel, models in response.items()
        },
        "temperature_calibration": temperature_fit,
        "permutation_invariance_max_abs_truth_probability": permutation_delta,
        "metrics": metrics,
        "paired_nll_differences": paired_nll,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output / "sheet_posteriors.npz",
        global_indices=ids, labels=labels, truth_modes=truth_modes, reco_modes=reco_modes,
        truth_sheet=truth_sheet,
        sv_available=available, ip_available=ip_available,
        valid_surface=valid_surface, retained=retained,
        **{f"posterior_{name}": value for name, value in arms.items()},
        **{name: value for name, value in evaluated.items() if name.startswith("log_evidence_")},
        temperature_sv=np.asarray(temperature_fit["sv"]["temperature"]),
        temperature_ip=np.asarray(temperature_fit["ip"]["temperature"]),
        temperature_length=np.asarray(temperature_fit["length"]["temperature"]),
        temperature_sv_length=np.asarray(temperature_fit["sv_length"]["temperature"]),
        temperature_all_geometry=np.asarray(temperature_fit["all_geometry"]["temperature"]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    calibration_angle = {}
    for mode in MODE_NAMES:
        values = []
        for side in (0, 1):
            subset = (response_modes[:, side] == mode) \
                & (geometry["sv_available"][response_ids, side] > 0)
            angle = geometry["sv_delta_theta"][response_ids[subset], side]
            values.append(angle[np.isfinite(angle)])
        calibration_angle[mode] = np.concatenate(values) if values else np.empty(0)
    bins = np.geomspace(1.0e-4, 1.0, 80)
    for mode, name in MODE_NAMES.items():
        if len(calibration_angle[mode]):
            axes[0, 0].hist(calibration_angle[mode], bins=bins, density=True,
                            histtype="step", lw=2, label=f"{name} (n={len(calibration_angle[mode])})")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set(xlabel="SV–truth tau angle [rad]", ylabel="density",
                   title="Training-calibration SV response")
    axes[0, 0].legend(fontsize=8)

    mask = valid_surface & available.any(axis=1)
    for key, color in (("sv_actual", "#4C78A8"), ("sv_matched_shuffle", "#E45756")):
        p = arms[key][mask, truth_sheet[mask]]
        axes[0, 1].hist(p, bins=np.linspace(0.0, 1.0, 41), density=True,
                        histtype="step", lw=2, label=key, color=color)
    axes[0, 1].axvline(0.25, color="0.3", ls="--", label="uniform")
    axes[0, 1].set(xlabel="posterior probability of truth sheet", ylabel="density",
                   title="Events with at least one reconstructed SV")
    axes[0, 1].legend(fontsize=8)

    actual_confusion = np.asarray(metrics["any_sv"]["sv_actual"]["confusion"])
    row = actual_confusion / np.maximum(actual_confusion.sum(axis=1, keepdims=True), 1)
    image = axes[1, 0].imshow(row, vmin=0.0, vmax=max(0.5, float(row.max())), cmap="Blues")
    for i in range(4):
        for j in range(4):
            axes[1, 0].text(j, i, f"{row[i, j]:.2f}\n({actual_confusion[i, j]})",
                            ha="center", va="center", fontsize=8)
    axes[1, 0].set(xlabel="predicted sheet", ylabel="truth sheet",
                   title="SV-direction selector confusion (any-SV)")
    axes[1, 0].set_xticks(range(4)); axes[1, 0].set_yticks(range(4))
    fig.colorbar(image, ax=axes[1, 0], label="row fraction")

    reliability = metrics["any_sv"]["sv_actual"]["reliability"]
    x = [item["mean_confidence"] for item in reliability if item["events"]]
    y = [item["accuracy"] for item in reliability if item["events"]]
    count = [item["events"] for item in reliability if item["events"]]
    axes[1, 1].plot([0, 1], [0, 1], color="0.5", ls="--")
    axes[1, 1].scatter(x, y, s=np.maximum(20, np.sqrt(count) * 5), color="#4C78A8")
    axes[1, 1].set_xlim(0.2, 1.0); axes[1, 1].set_ylim(0.2, 1.0)
    axes[1, 1].set(xlabel="mean predicted confidence", ylabel="observed accuracy",
                   title="SV-direction reliability (any-SV; marker size ∝ √events)")

    fig.suptitle("TauSpin WP-B: sheet information in momentum-independent geometry", fontsize=14)
    fig.savefig(args.output / "geometry_sheet_information.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report["response_models"], indent=2, sort_keys=True))
    print(json.dumps(metrics["inclusive"], indent=2, sort_keys=True))
    print(json.dumps(metrics["reco_threeprong_threeprong"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
