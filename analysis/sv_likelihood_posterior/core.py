"""Shared physics and statistics utilities for the SV-likelihood update."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


MODE_NAMES = {0: "1p0n", 1: "rho", 3: "3p0n"}


def opening_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.arctan2(
        np.linalg.norm(np.cross(a, b), axis=-1),
        np.sum(a * b, axis=-1),
    )


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-300)


def log_vmf_ratio(cosine: np.ndarray, kappa: float) -> np.ndarray:
    """log[vMF(direction|axis)/Uniform(S2)] with stable large-kappa algebra."""
    cosine = np.asarray(cosine, dtype=np.float64)
    if kappa <= 1.0e-10:
        return np.zeros_like(cosine)
    if kappa < 40.0:
        return math.log(kappa) - math.log(math.sinh(kappa)) + kappa * cosine
    return math.log(2.0 * kappa) + kappa * (cosine - 1.0)


def mixture_log_likelihood(
    cosine: np.ndarray, model: dict[str, float | int | None]
) -> np.ndarray:
    """vMF-core plus uniform-tail log likelihood ratio to Uniform(S2)."""
    pi = float(model["pi_core"])
    kappa = float(model["kappa"])
    if pi <= 0.0 or kappa <= 0.0:
        return np.zeros_like(cosine, dtype=np.float64)
    ratio = log_vmf_ratio(np.clip(cosine, -1.0, 1.0), kappa)
    return np.logaddexp(math.log1p(-pi), math.log(pi) + ratio)


def normalized_weights(log_weight: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    log_weight = np.asarray(log_weight, dtype=np.float64)
    if log_weight.ndim != 2:
        raise ValueError(f"Expected [event,draw] log weights, got {log_weight.shape}")
    maximum = np.max(log_weight, axis=1, keepdims=True)
    shifted = log_weight - maximum
    raw = np.exp(np.maximum(shifted, -745.0))
    weight = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.0e-300)
    log_safe = np.log(np.maximum(weight, 1.0e-300))
    entropy = -np.sum(weight * log_safe, axis=1)
    draws = log_weight.shape[1]
    diagnostics = {
        "ess": 1.0 / np.maximum(np.sum(weight * weight, axis=1), 1.0e-300),
        "max_weight": np.max(weight, axis=1),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(draws),
        "effective_populated": np.exp(entropy),
        "above_uniform_count": np.sum(weight >= (1.0 / draws), axis=1).astype(np.int16),
        "log_weight_span": np.max(log_weight, axis=1) - np.min(log_weight, axis=1),
    }
    return weight, diagnostics


def candidate_tau_direction(visible: np.ndarray, neutrino_xyz: np.ndarray) -> np.ndarray:
    visible = np.asarray(visible, dtype=np.float64)
    neutrino_xyz = np.asarray(neutrino_xyz, dtype=np.float64)
    if visible.ndim != 3 or visible.shape[1:] != (2, 4):
        raise ValueError(f"Unexpected visible shape {visible.shape}")
    if neutrino_xyz.ndim != 4 or neutrino_xyz.shape[2:] != (2, 3):
        raise ValueError(f"Unexpected neutrino shape {neutrino_xyz.shape}")
    return unit_vector(visible[:, None, :, :3] + neutrino_xyz)


def direction_local_coordinates(
    direction: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a direction in a deterministic orthonormal frame around axis."""
    direction = unit_vector(direction)
    axis = unit_vector(axis)
    reference = np.zeros_like(axis)
    use_z = np.abs(axis[..., 2]) < 0.9
    reference[..., 2] = use_z
    reference[..., 0] = ~use_z
    first = unit_vector(np.cross(reference, axis))
    second = np.cross(axis, first)
    coordinates = np.stack(
        (
            np.sum(direction * first, axis=-1),
            np.sum(direction * second, axis=-1),
            np.sum(direction * axis, axis=-1),
        ),
        axis=-1,
    )
    return coordinates, first, second


def direction_from_local_coordinates(
    coordinates: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """Map local angular coordinates back around a target axis."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    axis = unit_vector(axis)
    _, first, second = direction_local_coordinates(axis, axis)
    return unit_vector(
        coordinates[..., 0, None] * first
        + coordinates[..., 1, None] * second
        + coordinates[..., 2, None] * axis
    )


def offset_preserving_shuffle(
    direction: np.ndarray,
    visible_axis: np.ndarray,
    source_index: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """Shuffle SV-minus-visible offsets, then anchor them on each event's axis."""
    direction = np.asarray(direction, dtype=np.float64)
    visible_axis = unit_vector(visible_axis)
    source_index = np.asarray(source_index, dtype=np.int64)
    available = np.asarray(available, dtype=bool)
    if direction.shape != visible_axis.shape or direction.shape[:2] != source_index.shape:
        raise ValueError("Offset-shuffle shape mismatch")
    coordinates, _, _ = direction_local_coordinates(direction, visible_axis)
    shuffled_coordinates = np.array(coordinates, copy=True)
    for side in (0, 1):
        rows = np.flatnonzero(available[:, side])
        shuffled_coordinates[rows, side] = coordinates[source_index[rows, side], side]
    result = direction_from_local_coordinates(shuffled_coordinates, visible_axis)
    return np.where(available[..., None], result, direction)


def sv_log_weight(
    tau_direction: np.ndarray,
    measurement: np.ndarray,
    available: np.ndarray,
    reco_modes: np.ndarray,
    response: dict[int, dict[str, float | int | None]],
    temperature: float = 1.0,
) -> np.ndarray:
    """Evaluate the frozen per-side SV response for every posterior draw."""
    tau_direction = np.asarray(tau_direction, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    reco_modes = np.asarray(reco_modes, dtype=np.int8)
    result = np.zeros(tau_direction.shape[:2], dtype=np.float64)
    for side in (0, 1):
        cosine = np.sum(
            tau_direction[:, :, side] * measurement[:, None, side], axis=-1
        )
        for mode, model in response.items():
            rows = available[:, side] & (reco_modes[:, side] == int(mode))
            if rows.any():
                result[rows] += mixture_log_likelihood(cosine[rows], model)
    return result / max(float(temperature), 1.0e-12)


def gaussian_log_weight(
    tau_direction: np.ndarray,
    measurement: np.ndarray,
    available: np.ndarray,
    sigma_rad: float,
) -> np.ndarray:
    result = np.zeros(tau_direction.shape[:2], dtype=np.float64)
    for side in (0, 1):
        rows = np.asarray(available[:, side], dtype=bool)
        if rows.any():
            angle = opening_angle(
                tau_direction[rows, :, side], measurement[rows, None, side]
            )
            result[rows] += -0.5 * np.square(angle / float(sigma_rad))
    return result


def matched_direction_shuffle(
    direction: np.ndarray,
    available: np.ndarray,
    modes: np.ndarray,
    tau_pt: np.ndarray,
    tau_eta: np.ndarray,
    tau_phi: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle directions within the same topology/kinematic cells as WP-B."""
    direction = np.asarray(direction, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    out = np.array(direction, copy=True)
    source_index = np.broadcast_to(np.arange(len(direction))[:, None], available.shape).copy()
    log_pt = np.log(np.maximum(tau_pt, 1.0))
    if available.any():
        pt_edges = np.quantile(log_pt[available], [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        pt_edges = np.arange(5, dtype=float)
    eta_edges = np.array([-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf])
    phi_edges = np.linspace(-np.pi, np.pi, 9)
    pt_bin = np.clip(np.digitize(log_pt, pt_edges[1:-1]), 0, 3)
    eta_bin = np.clip(np.digitize(tau_eta, eta_edges[1:-1]), 0, 4)
    phi_wrapped = (tau_phi + np.pi) % (2.0 * np.pi) - np.pi
    phi_bin = np.clip(np.digitize(phi_wrapped, phi_edges[1:-1]), 0, 7)
    for side in (0, 1):
        for mode in MODE_NAMES:
            for p_bin in range(4):
                for e_bin in range(5):
                    for a_bin in range(8):
                        group = np.flatnonzero(
                            available[:, side]
                            & (modes[:, side] == mode)
                            & (pt_bin[:, side] == p_bin)
                            & (eta_bin[:, side] == e_bin)
                            & (phi_bin[:, side] == a_bin)
                        )
                        if len(group) > 1:
                            permuted = rng.permutation(group)
                            out[group, side] = direction[permuted, side]
                            source_index[group, side] = permuted
    return out, source_index


def weighted_representations(
    h_samples: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h_samples = np.asarray(h_samples, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mean = np.sum(weight[:, :, None, None] * h_samples, axis=1)
    joint = np.einsum(
        "ns,nsi,nsj->nij", weight, h_samples[:, :, 0], h_samples[:, :, 1]
    )
    moments = np.concatenate((mean.reshape(len(mean), 6), joint.reshape(len(mean), 9)), axis=1)
    return mean, moments


def safe_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    dot = np.sum(prediction * truth, axis=-1)
    norm = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(truth, axis=-1)
    return np.divide(dot, norm, out=np.full_like(dot, np.nan, dtype=np.float64), where=norm > 0)


def h_event_values(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    squared_error = np.mean(np.square(prediction - truth), axis=(1, 2))
    cosine = np.nanmean(safe_cosine(prediction, truth), axis=1)
    return squared_error, cosine


def h_metric_block(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    finite = (
        np.isfinite(prediction).all(axis=(1, 2))
        & np.isfinite(truth).all(axis=(1, 2))
        & mask
    )
    if not finite.any():
        return {"events": 0}
    residual = prediction[finite] - truth[finite]
    cosine = safe_cosine(prediction[finite], truth[finite])
    component_mse = np.mean(np.square(residual), axis=(0, 1))
    correlations = []
    for component in range(3):
        x = prediction[finite, :, component].reshape(-1)
        y = truth[finite, :, component].reshape(-1)
        correlations.append(
            float(np.corrcoef(x, y)[0, 1])
            if np.std(x) > 0 and np.std(y) > 0 else None
        )
    event_error, event_cosine = h_event_values(prediction[finite], truth[finite])
    return {
        "events": int(finite.sum()),
        "mse": float(np.mean(np.square(residual))),
        "component_mse": component_mse.tolist(),
        "cosine": float(np.nanmean(cosine)),
        "component_correlation": correlations,
        "event_error_quantiles": np.quantile(event_error, [0, .16, .5, .84, .95, .99, 1]).tolist(),
        "event_cosine_quantiles": np.quantile(event_cosine, [0, .16, .5, .84, .95, .99, 1]).tolist(),
    }


def auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (len(labels) == len(scores) == len(weights)):
        raise ValueError("AUC input lengths differ")
    if len(labels) == 0 or not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("Invalid AUC input")
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0 or negative <= 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    w = weights[order]
    s = scores[order]
    tp = np.cumsum(w * (y == 1))
    fp = np.cumsum(w * (y == 0))
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tpr = np.r_[0.0, tp[distinct] / positive]
    fpr = np.r_[0.0, fp[distinct] / negative]
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(tpr, fpr))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
