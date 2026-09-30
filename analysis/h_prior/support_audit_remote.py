"""Audit the numerical support of the Stage-2 hybrid constituent inputs.

This script intentionally does not evaluate a network.  It reloads the exact
donor identities saved by ``counterfactual_remote.py``, constructs the aligned
donor constituents, and checks coordinate closure and validation-range
excursions for every primary recipient, donor map, and replaced side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from counterfactual_remote import align_constituents, raw_value, wrap_phi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p11-dir", type=Path, required=True)
    p.add_argument("--targets-dir", type=Path, required=True)
    p.add_argument("--geometry-features", type=Path, required=True)
    p.add_argument("--geometry-key", default="full")
    p.add_argument("--geometry-dim", type=int, default=22)
    p.add_argument("--counterfactual-output", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def angle_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_phi(a - b)


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {str(p): float(np.percentile(x, p)) for p in (0, 50, 90, 99, 100)}


def main() -> None:
    args = parse_args()
    os.environ["GEO_DIM"] = str(args.geometry_dim)
    os.environ["GEO_FEATURES"] = str(args.geometry_features)
    os.environ["GEO_KEY"] = args.geometry_key
    sys.path.insert(0, str(args.p11_dir))
    import p11_train_full_geometry as p11  # noqa: E402

    class GeneratorTargetDataset(p11.Geo16Dataset):
        def __init__(self, split: str):
            super().__init__(split, targets_dir=args.targets_dir)

    dataset = GeneratorTargetDataset("validation")
    stats = json.loads(args.stats.read_text())
    with np.load(args.counterfactual_output) as saved:
        cf = {key: np.asarray(saved[key]) for key in saved.files}

    global_index = dataset.global_index.detach().cpu().numpy()
    position = {int(value): i for i, value in enumerate(global_index)}
    rows = np.asarray([position[int(value)] for value in cf["global_indices"]], dtype=np.int64)

    original_range: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("track", "pfo"):
        feature_key = f"{kind}_features"
        lo = None
        hi = None
        for i in range(len(dataset)):
            values = dataset[i][feature_key].numpy()
            if len(values) == 0:
                continue
            vmin, vmax = values.min(axis=0), values.max(axis=0)
            lo = vmin if lo is None else np.minimum(lo, vmin)
            hi = vmax if hi is None else np.maximum(hi, vmax)
        if lo is None or hi is None:
            raise RuntimeError(f"no {kind} constituents in validation")
        original_range[kind] = (lo, hi)

    closure = {
        "track_eta": [], "track_phi": [], "track_pt_fraction": [],
        "pfo_eta": [], "pfo_phi": [], "pfo_pt_fraction": [], "pfo_e_over_pt": [],
    }
    match = {"tau_logpt_standardized_abs": [], "tau_eta_standardized_abs": [], "tau_phi_abs_rad": []}
    token_counts = {"track_recipient": [], "track_donor": [], "pfo_recipient": [], "pfo_donor": []}
    outside = {
        "track_checked_values": 0, "track_below_or_above": 0,
        "pfo_checked_values": 0, "pfo_below_or_above": 0,
    }
    nonfinite = 0

    for seed_index in range(3):
        donor_by_side = (
            np.asarray([position[int(value)] for value in cf[f"donor_minus_seed{seed_index}"]], dtype=np.int64),
            np.asarray([position[int(value)] for value in cf[f"donor_plus_seed{seed_index}"]], dtype=np.int64),
        )
        for row_position, recipient_index in enumerate(rows):
            recipient = dataset[int(recipient_index)]
            for side in (0, 1):
                donor = dataset[int(donor_by_side[side][row_position])]
                recipient_tau = recipient["tau_features"][side]
                donor_tau = donor["tau_features"][side]
                match["tau_logpt_standardized_abs"].append(float(torch.abs(recipient_tau[0] - donor_tau[0])))
                match["tau_eta_standardized_abs"].append(float(torch.abs(recipient_tau[1] - donor_tau[1])))
                rphi = torch.atan2(recipient_tau[2], recipient_tau[3])
                dphi = torch.atan2(donor_tau[2], donor_tau[3])
                match["tau_phi_abs_rad"].append(float(torch.abs(angle_delta(rphi, dphi))))

                for kind in ("track", "pfo"):
                    feature_key, side_key = f"{kind}_features", f"{kind}_sides"
                    recipient_values = recipient[feature_key][recipient[side_key] == side]
                    donor_values = donor[feature_key][donor[side_key] == side]
                    aligned = align_constituents(donor_values, kind, recipient_tau, stats)
                    token_counts[f"{kind}_recipient"].append(len(recipient_values))
                    token_counts[f"{kind}_donor"].append(len(donor_values))
                    nonfinite += int((~torch.isfinite(aligned)).sum())
                    if len(aligned) == 0:
                        continue

                    group = stats[kind]
                    tau_logpt = raw_value(recipient_tau[0], stats["tau"], 0)
                    tau_pt = torch.expm1(tau_logpt).clamp_min(1e-8)
                    tau_eta = raw_value(recipient_tau[1], stats["tau"], 1)
                    tau_phi = torch.atan2(recipient_tau[2], recipient_tau[3])
                    if kind == "track":
                        rel_eta_i, sin_dphi_i, cos_dphi_i, logfrac_i = 15, 16, 17, 18
                        adjusted = (0, 1, 2, 3)
                    else:
                        rel_eta_i, sin_dphi_i, cos_dphi_i, logfrac_i = 6, 7, 8, 9
                        adjusted = (0, 1, 2, 3, 4)
                    abs_eta = raw_value(aligned[:, 1], group, 1)
                    rel_eta = raw_value(aligned[:, rel_eta_i], group, rel_eta_i)
                    abs_phi = torch.atan2(aligned[:, 2], aligned[:, 3])
                    rel_phi = torch.atan2(aligned[:, sin_dphi_i], aligned[:, cos_dphi_i])
                    abs_pt = torch.expm1(raw_value(aligned[:, 0], group, 0)).clamp_min(0)
                    fraction = torch.expm1(raw_value(aligned[:, logfrac_i], group, logfrac_i)).clamp_min(0)
                    closure[f"{kind}_eta"].extend(torch.abs(abs_eta - tau_eta - rel_eta).tolist())
                    closure[f"{kind}_phi"].extend(torch.abs(angle_delta(abs_phi - tau_phi, rel_phi)).tolist())
                    closure[f"{kind}_pt_fraction"].extend(torch.abs(abs_pt / tau_pt - fraction).tolist())
                    if kind == "pfo":
                        old_pt = torch.expm1(raw_value(donor_values[:, 0], group, 0)).clamp_min(1e-8)
                        old_e = torch.expm1(raw_value(donor_values[:, 4], group, 4)).clamp_min(0)
                        new_e = torch.expm1(raw_value(aligned[:, 4], group, 4)).clamp_min(0)
                        closure["pfo_e_over_pt"].extend(torch.abs(new_e / abs_pt.clamp_min(1e-8) - old_e / old_pt).tolist())

                    lo, hi = original_range[kind]
                    values = aligned[:, adjusted].numpy()
                    outside[f"{kind}_checked_values"] += int(values.size)
                    outside[f"{kind}_below_or_above"] += int(np.sum((values < lo[list(adjusted)]) | (values > hi[list(adjusted)])))

    report = {
        "validation_rows": int(len(dataset)),
        "primary_rows": int(len(rows)),
        "hybrid_side_instances": int(3 * 2 * len(rows)),
        "nonfinite_aligned_values": int(nonfinite),
        "absolute_coordinate_closure": {key: quantiles(value) for key, value in closure.items()},
        "recipient_donor_tau_matching": {key: quantiles(value) for key, value in match.items()},
        "token_counts": {
            key: quantiles(value) for key, value in token_counts.items()
        } | {
            "track_exact_match_fraction": float(np.mean(np.asarray(token_counts["track_recipient"]) == np.asarray(token_counts["track_donor"]))),
            "pfo_exact_match_fraction": float(np.mean(np.asarray(token_counts["pfo_recipient"]) == np.asarray(token_counts["pfo_donor"]))),
        },
        "adjusted_features_outside_validation_extrema": {
            kind: {
                "checked_values": int(outside[f"{kind}_checked_values"]),
                "outside_values": int(outside[f"{kind}_below_or_above"]),
                "fraction": float(outside[f"{kind}_below_or_above"] / max(outside[f"{kind}_checked_values"], 1)),
            }
            for kind in ("track", "pfo")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
