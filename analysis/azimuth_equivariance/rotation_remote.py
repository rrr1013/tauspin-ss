"""Evaluate fixed point-h pipelines on global beam-axis rotations.

Absolute packed azimuth pairs are rotated analytically.  Geometry is rebuilt
from saved raw validation vectors for every angle, so Cartesian component-wise
clipping is applied only after the physical rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


C16 = np.asarray([2 * np.pi * j / 16 for j in range(16)], dtype=np.float64)
C17 = np.asarray([2 * np.pi * (j + 0.37) / 17 for j in range(17)], dtype=np.float64)


class FixedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rotate_xy(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    c, s = math.cos(alpha), math.sin(alpha)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-300)


def geometry_block(raw: dict[str, np.ndarray], rows: np.ndarray, alpha: float, arm: str):
    n = len(rows)
    out = np.zeros((n, 2, 22), dtype=np.float32)
    if arm == "none":
        return out, np.zeros(n, dtype=bool)
    ip = rotate_xy(raw["ip"][rows], alpha)
    track = rotate_xy(raw["track"][rows], alpha)
    sv = rotate_xy(raw["sv"][rows], alpha)
    basis = rotate_xy(raw["basis"][rows], alpha)
    ip_ok = raw["ip_ok"][rows]
    sv_ok = raw["sv_ok"][rows]
    if arm == "lab":
        saturated = np.zeros(n, dtype=bool)
        for j in range(3):
            scaled = np.nan_to_num(ip[:, :, j]) / 0.05
            good = ip_ok[:, :, j]
            saturated |= np.any(good[..., None] & (np.abs(scaled) >= 20), axis=(1, 2))
            block = np.concatenate(
                [np.clip(scaled, -20, 20), np.ones((n, 2, 1))], axis=-1
            )
            out[..., 4 * j:4 * j + 4] = np.where(good[..., None], block, 0)
        scaled_sv = np.nan_to_num(sv) / 1.0
        saturated |= np.any(sv_ok[..., None] & (np.abs(scaled_sv) >= 50), axis=(1, 2))
        sv_block = np.concatenate(
            [np.clip(scaled_sv, -50, 50), np.ones((n, 2, 1))], axis=-1
        )
        out[..., 12:16] = np.where(sv_ok[..., None], sv_block, 0)
        return out, saturated
    if arm != "local":
        raise ValueError(arm)
    for j in range(3):
        vector = ip[:, :, j]
        direction = unit(vector)
        good = ip_ok[:, :, j] & np.isfinite(direction).all(-1)
        track_direction = track[:, :, j]
        cols = [
            np.sum(direction * basis[:, :, 0], axis=-1),
            np.sum(direction * basis[:, :, 1], axis=-1),
            np.clip(np.log(np.maximum(np.linalg.norm(vector, axis=-1), 1e-9) / 0.05), -4, 4),
            np.ones((n, 2)),
            np.clip(np.sum(track_direction * basis[:, :, 0], axis=-1) / 0.01, -20, 20),
            np.clip(np.sum(track_direction * basis[:, :, 1], axis=-1) / 0.01, -20, 20),
        ]
        out[..., 6 * j:6 * j + 6] = np.where(good[..., None], np.stack(cols, -1), 0)
    sv_direction = unit(sv)
    length = np.linalg.norm(sv, axis=-1)
    cols = np.stack(
        [
            np.clip(np.sum(sv_direction * basis[:, :, 0], axis=-1) / 0.01, -20, 20),
            np.clip(np.sum(sv_direction * basis[:, :, 1], axis=-1) / 0.01, -20, 20),
            np.clip(np.log(np.maximum(length, 1e-6) / 1.0), -6, 6),
            np.ones((n, 2)),
        ],
        -1,
    )
    out[..., 18:22] = np.where(sv_ok[..., None], cols, 0)
    return np.nan_to_num(out), np.zeros(n, dtype=bool)


def rotate_pair(tensor: torch.Tensor, sin_index: int, cos_index: int, alpha: float) -> None:
    c, s = math.cos(alpha), math.sin(alpha)
    sin0 = tensor[..., sin_index].clone()
    cos0 = tensor[..., cos_index].clone()
    tensor[..., sin_index] = c * sin0 + s * cos0
    tensor[..., cos_index] = c * cos0 - s * sin0


def transformed_batch(base: dict, geometry: np.ndarray, alpha: float, device: torch.device) -> dict:
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    rotate_pair(batch["event_features"], 1, 2, alpha)
    rotate_pair(batch["tau_features"], 2, 3, alpha)
    rotate_pair(batch["track_features"], 2, 3, alpha)
    rotate_pair(batch["pfo_features"], 2, 3, alpha)
    block = torch.as_tensor(geometry, dtype=batch["tau_features"].dtype)
    tau_geometry = batch["tau_features"][..., 10:32]
    for source_side, packed_side in ((0, 1), (1, 2)):
        mask = (batch["object_type"] == 2) & (batch["tau_side"] == packed_side)
        if not torch.equal(mask.sum(1), torch.ones(len(mask), dtype=mask.sum(1).dtype)):
            raise RuntimeError("expected one tau token per event and side")
        tau_geometry[mask] = block[:, source_side]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def feature_controls(base: dict, raw: dict[str, np.ndarray], rows: np.ndarray, arm: str) -> dict:
    alpha, beta = 0.731, -1.117
    probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    direct = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    for name in ("event_features", "tau_features", "track_features", "pfo_features"):
        pair = (1, 2) if name == "event_features" else (2, 3)
        rotate_pair(probe[name], *pair, alpha)
        rotate_pair(probe[name], *pair, beta)
        rotate_pair(direct[name], *pair, alpha + beta)
    packed_composition = max(
        float(torch.max(torch.abs(probe[name] - direct[name])))
        for name in ("event_features", "tau_features", "track_features", "pfo_features")
    )
    inverse = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    for name in ("event_features", "tau_features", "track_features", "pfo_features"):
        pair = (1, 2) if name == "event_features" else (2, 3)
        rotate_pair(inverse[name], *pair, alpha)
        rotate_pair(inverse[name], *pair, -alpha)
    packed_inverse = max(
        float(torch.max(torch.abs(inverse[name] - base[name])))
        for name in ("event_features", "tau_features", "track_features", "pfo_features")
    )
    g0, _ = geometry_block(raw, rows, 0.0, arm)
    local_max = 0.0
    if arm == "local":
        local_max = max(
            float(np.max(np.abs(geometry_block(raw, rows, angle, arm)[0] - g0)))
            for angle in np.r_[C16, C17]
        )
    return {
        "packed_composition_max_abs": packed_composition,
        "packed_inverse_max_abs": packed_inverse,
        "local_geometry_invariance_max_abs": local_max,
    }


def score_array(readout: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    flat = torch.from_numpy(values.astype(np.float32, copy=False).reshape(-1, 6))
    out = []
    with torch.inference_mode():
        for start in range(0, len(flat), 2048):
            out.append(torch.sigmoid(readout(flat[start:start + 2048].to(device))).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--geometry-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--arm", choices=("none", "lab", "local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    os.environ["SAMPLE"] = "atlas"
    os.environ["GEO_FEATURES"] = str(args.geometry_features)
    os.environ["GEO_KEY"] = args.arm
    os.environ["GEO_DIM"] = "22"
    os.environ["SEED"] = "42"
    sys.path.insert(0, str(args.checkout / "analysis/meeting_followup/ipsv"))
    import train_ipsv as train  # noqa: E402

    dataset = train.AtlasReduced("validation")
    collate = train.P.T.collate_reco_h
    device = torch.device("cuda")
    train.P.T.configure_runtime(device)
    model = train.P.T.RecoModel("geo16").to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    network = torch.compile(model, dynamic=True)
    readout = FixedMLP().to(device)
    readout_payload = torch.load(args.readout, map_location="cpu", weights_only=True)
    readout.load_state_dict(readout_payload["model_state_dict"], strict=True)
    readout.eval()

    with np.load(args.geometry_features) as source:
        ids = np.asarray(source["validation_global_indices"], np.int64)
        raw = {
            "ip": np.asarray(source["res_ip"]),
            "ip_ok": np.asarray(source["res_ip_ok"], bool),
            "track": np.asarray(source["res_track_dir"]),
            "sv": np.asarray(source["res_sv"]),
            "sv_ok": np.asarray(source["res_sv_ok"], bool),
            "basis": np.asarray(source["res_basis"]),
        }
    dataset_ids = dataset.global_index.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(ids, dataset_ids):
        raise RuntimeError("raw geometry / dataset identity mismatch")
    n = len(dataset)
    if args.limit is not None:
        n = min(n, args.limit)
    rows_all = np.arange(n, dtype=np.int64)
    h16 = np.empty((n, len(C16), 2, 3), np.float32)
    h17 = np.empty((n, len(C17), 2, 3), np.float32)
    s16 = np.empty((n, len(C16)), np.float32)
    s17 = np.empty((n, len(C17)), np.float32)
    repeat_max = 0.0
    twopi_max = 0.0
    saturated = np.zeros((n, len(C16)), bool)
    controls = None

    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            rows = rows_all[start:start + args.batch_size]
            events = [dataset[int(i)] for i in rows]
            base = collate(events)
            if controls is None:
                controls = feature_controls(base, raw, rows, args.arm)
            alpha_zero_prediction = None
            for target_h, target_s, angles, is_primary in (
                (h16, s16, C16, True), (h17, s17, C17, False)
            ):
                for j, alpha in enumerate(angles):
                    geometry, sat = geometry_block(raw, rows, float(alpha), args.arm)
                    batch = transformed_batch(base, geometry, float(alpha), device)
                    pred = network(batch)[1]
                    score = torch.sigmoid(readout(pred.reshape(-1, 6)))
                    target_h[start:start + len(rows), j] = pred.cpu().numpy()
                    target_s[start:start + len(rows), j] = score.cpu().numpy()
                    if is_primary:
                        saturated[start:start + len(rows), j] = sat
                    if is_primary and j == 0:
                        alpha_zero_prediction = pred
            geometry0, _ = geometry_block(raw, rows, 0.0, args.arm)
            repeat = network(transformed_batch(base, geometry0, 0.0, device))[1]
            geometry2, _ = geometry_block(raw, rows, 2 * np.pi, args.arm)
            twopi = network(transformed_batch(base, geometry2, 2 * np.pi, device))[1]
            repeat_max = max(repeat_max, float(torch.max(torch.abs(repeat - alpha_zero_prediction))))
            twopi_max = max(twopi_max, float(torch.max(torch.abs(twopi - alpha_zero_prediction))))

    with np.load(args.reference_predictions) as reference:
        ref_ids = np.asarray(reference["global_indices"], np.int64)
        if not np.array_equal(ref_ids[:n], dataset_ids[:n]):
            raise RuntimeError("reference prediction identity mismatch")
        ref_h = np.asarray(reference["h_pred"], np.float32)[:n]
    parity_max = float(np.max(np.abs(h16[:, 0] - ref_h)))
    if parity_max > 2e-5:
        raise RuntimeError(f"alpha=0 prediction parity failed: {parity_max}")
    if controls is None:
        raise RuntimeError("empty dataset")
    if controls["packed_composition_max_abs"] > 5e-6 or controls["packed_inverse_max_abs"] > 5e-6:
        raise RuntimeError(f"packed group-action closure failed: {controls}")
    if args.arm == "local" and controls["local_geometry_invariance_max_abs"] > 5e-6:
        raise RuntimeError(f"local geometry invariance failed: {controls}")

    score_c16_average_h = score_array(readout, h16.mean(axis=1), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_indices=dataset_ids[:n],
        labels=dataset.labels.detach().cpu().numpy()[:n],
        truth_modes=dataset.truth_modes.detach().cpu().numpy()[:n],
        overlap_weights=dataset.overlap_weight.detach().cpu().numpy()[:n],
        h_exact=dataset.h.detach().cpu().numpy()[:n],
        angles_c16=C16,
        angles_c17=C17,
        h_c16=h16,
        h_c17=h17,
        score_c16=s16,
        score_c17=s17,
        score_c16_average_h=score_c16_average_h,
        lab_saturated_c16=saturated,
    )
    report = {
        "status": "complete",
        "arm": args.arm,
        "rows": n,
        "pilot_limit": args.limit,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint), "epoch": int(checkpoint["epoch"])},
        "reference_predictions": {"path": str(args.reference_predictions), "sha256": sha256(args.reference_predictions)},
        "readout": {"path": str(args.readout), "sha256": sha256(args.readout)},
        "geometry": {"path": str(args.geometry_features), "sha256": sha256(args.geometry_features)},
        "alpha0_reference_parity_max_abs": parity_max,
        "repeat_inference_max_abs": repeat_max,
        "twopi_prediction_max_abs": twopi_max,
        "feature_controls": controls,
        "lab_saturation_fraction_by_c16_angle": saturated.mean(0).tolist(),
        "lab_fully_unsaturated_events": int((~saturated.any(1)).sum()),
        "device": torch.cuda.get_device_name(0),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
