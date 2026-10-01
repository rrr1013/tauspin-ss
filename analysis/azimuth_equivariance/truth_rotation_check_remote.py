"""Independently rebuild canonical exact h after global z rotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


C16 = np.asarray([2 * np.pi * j / 16 for j in range(16)], dtype=np.float64)
C17 = np.asarray([2 * np.pi * (j + 0.37) / 17 for j in range(17)], dtype=np.float64)


def rotate_four(values: np.ndarray, alpha: float) -> np.ndarray:
    out = values.copy()
    c, s = np.cos(alpha), np.sin(alpha)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def fill_h(frames, modes, charges, canonical_h) -> np.ndarray:
    out = np.full((len(modes), 2, 3), np.nan, dtype=np.float64)
    for side in (0, 1):
        for mode in (0, 1, 3):
            selected, values = canonical_h(frames, modes, charges, side, mode)
            out[selected, side] = values
    return out


def basis_metrics(basis: np.ndarray) -> tuple[float, float]:
    gram = np.einsum("...ai,...bi->...ab", basis, basis)
    eye = np.eye(3)
    orth = float(np.max(np.abs(gram - eye)))
    determinant = np.linalg.det(basis)
    handed = float(np.max(np.abs(determinant - 1.0)))
    return orth, handed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout / "analysis/gen3pi_teacher"))
    sys.path.insert(0, str(args.checkout / "analysis/mode_pair_auc_origin"))
    import build_targets  # noqa: E402
    from polarimeter import canonical_h, frames  # noqa: E402

    with np.load(args.targets) as source:
        valid = np.asarray(source["h_valid"], bool)
        labels = np.asarray(source["labels"])[valid]
        file_indices = np.asarray(source["file_indices"])[valid]
        entry_indices = np.asarray(source["entry_indices"])[valid]
        event_numbers = np.asarray(source["event_numbers"])[valid]
        modes = np.asarray(source["modes"])[valid]
        stored_h = np.asarray(source["h"], np.float64)[valid]
    if args.limit is not None:
        labels = labels[:args.limit]
        file_indices = file_indices[:args.limit]
        entry_indices = entry_indices[:args.limit]
        event_numbers = event_numbers[:args.limit]
        modes = modes[:args.limit]
        stored_h = stored_h[:args.limit]
    truth = build_targets.read_truth(labels, file_indices, entry_indices, event_numbers, modes)
    base_frames = frames(truth["pions"], truth["pi0"], truth["nu4"])
    base_h = fill_h(base_frames, modes, truth["charges"], canonical_h)
    k = base_frames["basis"][:, 2]
    near_beam = np.linalg.norm(np.cross(np.array([0.0, 0.0, -1.0]), k), axis=-1) < 1e-6
    keep = ~near_beam
    stored_closure = float(np.max(np.abs(base_h[keep] - stored_h[keep])))
    orth0, handed0 = basis_metrics(base_frames["basis"][keep])
    rows = []
    maximum_h = 0.0
    maximum_basis = 0.0
    for grid, angles in (("c16", C16), ("c17", C17)):
        for alpha in angles:
            rotated = {
                "pions": rotate_four(truth["pions"], float(alpha)),
                "pi0": rotate_four(truth["pi0"], float(alpha)),
                "nu4": rotate_four(truth["nu4"], float(alpha)),
            }
            fr = frames(rotated["pions"], rotated["pi0"], rotated["nu4"])
            h = fill_h(fr, modes, truth["charges"], canonical_h)
            h_diff = np.linalg.norm(h[keep] - base_h[keep], axis=-1)
            expected_basis = rotate_four(base_frames["basis"], float(alpha))
            basis_diff = np.abs(fr["basis"][keep] - expected_basis[keep])
            orth, handed = basis_metrics(fr["basis"][keep])
            row = {
                "grid": grid,
                "alpha": float(alpha),
                "h_max_norm": float(np.max(h_diff)),
                "h_p99_norm": float(np.quantile(h_diff, 0.99)),
                "basis_covariance_max_abs": float(np.max(basis_diff)),
                "basis_orthonormality_max_abs": orth,
                "basis_handedness_max_abs": handed,
            }
            maximum_h = max(maximum_h, row["h_max_norm"])
            maximum_basis = max(maximum_basis, row["basis_covariance_max_abs"])
            rows.append(row)
    if maximum_h > 2e-5 or maximum_basis > 2e-5:
        raise RuntimeError(f"truth rotation closure failed: h={maximum_h}, basis={maximum_basis}")
    report = {
        "status": "complete",
        "rows": int(len(modes)),
        "pilot_limit": args.limit,
        "near_beam_rows_excluded": int(near_beam.sum()),
        "stored_target_max_abs": stored_closure,
        "base_basis_orthonormality_max_abs": orth0,
        "base_basis_handedness_max_abs": handed0,
        "rotation_h_max_norm": maximum_h,
        "rotation_basis_covariance_max_abs": maximum_basis,
        "angles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
