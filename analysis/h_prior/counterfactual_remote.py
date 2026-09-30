"""Evaluate fixed point-h networks under a matched opposite-side constituent swap.

Run this on the ICEPP GPU node from a clean checkout.  The intervention keeps
the recipient event block and both recipient tau tokens, replacing only one
side's track/PFO constituents.  Donor constituents are deterministically
re-expressed around the recipient tau using their stored relative features.
This measures functional dependence under the specified hybrid input; it does
not identify a training-prior cause.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


SEEDS = (20261001, 20261002, 20261003)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p11-dir", type=Path, required=True)
    p.add_argument("--targets-dir", type=Path, required=True)
    p.add_argument("--geometry-features", type=Path, required=True)
    p.add_argument("--geometry-key", default="full")
    p.add_argument("--geometry-dim", type=int, default=22)
    p.add_argument("--arm", choices=("base16", "geo16"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference-predictions", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    return p.parse_args()


def raw_value(x: torch.Tensor, group: dict, index: int) -> torch.Tensor:
    return x * float(group["std"][index]) + float(group["mean"][index])


def standardized_value(x: torch.Tensor, group: dict, index: int) -> torch.Tensor:
    return (x - float(group["mean"][index])) / float(group["std"][index])


def wrap_phi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def align_constituents(features: torch.Tensor, kind: str, recipient_tau: torch.Tensor, stats: dict) -> torch.Tensor:
    """Move donor constituents to the recipient tau axis, preserving relative features."""
    if len(features) == 0:
        return features.clone()
    out = features.clone()
    tau_stats = stats["tau"]
    group = stats[kind]
    tau_logpt = raw_value(recipient_tau[0], tau_stats, 0)
    tau_pt = torch.expm1(tau_logpt).clamp_min(1e-8)
    tau_eta = raw_value(recipient_tau[1], tau_stats, 1)
    tau_phi = torch.atan2(recipient_tau[2], recipient_tau[3])

    if kind == "track":
        rel_eta_i, sin_dphi_i, cos_dphi_i, logfrac_i = 15, 16, 17, 18
        logpt_i, eta_i, sin_phi_i, cos_phi_i = 0, 1, 2, 3
        rel_eta = raw_value(features[:, rel_eta_i], group, rel_eta_i)
        dphi = torch.atan2(features[:, sin_dphi_i], features[:, cos_dphi_i])
        logfrac = raw_value(features[:, logfrac_i], group, logfrac_i)
        fraction = torch.expm1(logfrac).clamp_min(0)
        new_pt = (fraction * tau_pt).clamp_min(0)
        out[:, logpt_i] = standardized_value(torch.log1p(new_pt), group, logpt_i)
        out[:, eta_i] = standardized_value(tau_eta + rel_eta, group, eta_i)
        phi = wrap_phi(tau_phi + dphi)
        out[:, sin_phi_i], out[:, cos_phi_i] = torch.sin(phi), torch.cos(phi)
    elif kind == "pfo":
        rel_eta_i, sin_dphi_i, cos_dphi_i, logfrac_i = 6, 7, 8, 9
        logpt_i, eta_i, sin_phi_i, cos_phi_i, loge_i = 0, 1, 2, 3, 4
        rel_eta = raw_value(features[:, rel_eta_i], group, rel_eta_i)
        dphi = torch.atan2(features[:, sin_dphi_i], features[:, cos_dphi_i])
        logfrac = raw_value(features[:, logfrac_i], group, logfrac_i)
        fraction = torch.expm1(logfrac).clamp_min(0)
        old_pt = torch.expm1(raw_value(features[:, logpt_i], group, logpt_i)).clamp_min(1e-8)
        old_e = torch.expm1(raw_value(features[:, loge_i], group, loge_i)).clamp_min(0)
        new_pt = (fraction * tau_pt).clamp_min(0)
        new_e = old_e * new_pt / old_pt
        out[:, logpt_i] = standardized_value(torch.log1p(new_pt), group, logpt_i)
        out[:, loge_i] = standardized_value(torch.log1p(new_e), group, loge_i)
        out[:, eta_i] = standardized_value(tau_eta + rel_eta, group, eta_i)
        phi = wrap_phi(tau_phi + dphi)
        out[:, sin_phi_i], out[:, cos_phi_i] = torch.sin(phi), torch.cos(phi)
    else:
        raise ValueError(kind)
    return out


def hybrid_event(recipient: dict, donor: dict, replace_side: int, stats: dict) -> dict:
    """Replace one side's constituent tokens; all recipient event/tau blocks remain fixed."""
    if int(recipient["global_index"]) == int(donor["global_index"]):
        return recipient
    out = dict(recipient)
    tau = recipient["tau_features"][replace_side]
    rkeep = recipient["track_sides"] != replace_side
    dsel = donor["track_sides"] == replace_side
    donor_track = align_constituents(donor["track_features"][dsel], "track", tau, stats)
    out["track_features"] = torch.cat((recipient["track_features"][rkeep], donor_track), dim=0)
    out["track_sides"] = torch.cat((recipient["track_sides"][rkeep], donor["track_sides"][dsel]), dim=0)

    rkeep = recipient["pfo_sides"] != replace_side
    dsel = donor["pfo_sides"] == replace_side
    donor_pfo = align_constituents(donor["pfo_features"][dsel], "pfo", tau, stats)
    out["pfo_features"] = torch.cat((recipient["pfo_features"][rkeep], donor_pfo), dim=0)
    out["pfo_sides"] = torch.cat((recipient["pfo_sides"][rkeep], donor["pfo_sides"][dsel]), dim=0)
    return out


def deranged_maps(meta: dict[str, np.ndarray], seed: int) -> tuple[list[np.ndarray], dict]:
    """One-to-one derangements within predeclared label/mode/nTracks/kinematic cells."""
    maps, reports = [], []
    for side in (0, 1):
        label = meta["label"]
        mode = meta["reco_mode"][:, side]
        ntracks = meta["ntracks"][:, side]
        x = meta["tau_match"][:, side]
        bins = np.zeros((len(label), 3), dtype=np.int16)
        # Quantiles are fixed within label x reco-mode x nTracks strata.
        for lab in np.unique(label):
            for md in np.unique(mode[label == lab]):
                for nt in np.unique(ntracks[(label == lab) & (mode == md)]):
                    idx = np.flatnonzero((label == lab) & (mode == md) & (ntracks == nt))
                    if len(idx) < 2:
                        continue
                    for j in range(3):
                        edges = np.unique(np.quantile(x[idx, j], (0.25, 0.5, 0.75)))
                        bins[idx, j] = np.searchsorted(edges, x[idx, j], side="right")
        keys = np.stack((label, mode, ntracks, bins[:, 0], bins[:, 1], bins[:, 2]), axis=1)
        donor = np.full(len(label), -1, dtype=np.int64)
        rng = np.random.default_rng(seed + 1000 * side)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        sizes = []
        for cell in range(len(unique)):
            idx = np.flatnonzero(inverse == cell)
            sizes.append(len(idx))
            if len(idx) < 2:
                continue
            order = rng.permutation(idx)
            donor[order] = np.roll(order, 1)
        valid = donor >= 0
        if np.any(donor[valid] == np.flatnonzero(valid)):
            raise RuntimeError("derangement contains a self donor")
        maps.append(donor)
        reports.append({
            "side": side,
            "cells": int(len(unique)),
            "singleton_rows": int(np.sum(~valid)),
            "cell_size_min": int(min(sizes)),
            "cell_size_median": float(np.median(sizes)),
            "cell_size_max": int(max(sizes)),
            "donor_unique": int(len(np.unique(donor[valid]))),
        })
    return maps, {"seed": seed, "sides": reports}


def evaluate(model, collate, dataset, indices: np.ndarray, device, batch_size: int,
             donor_maps: list[np.ndarray] | None = None, variant: str = "original",
             stats: dict | None = None) -> np.ndarray:
    out = np.empty((len(indices), 2, 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            if variant == "original":
                events = [dataset[int(i)] for i in rows]
                batch = collate(events)
                pred = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})[1]
                out[start:start + len(rows)] = pred.detach().cpu().numpy()
                continue
            if donor_maps is None or stats is None:
                raise ValueError("hybrid evaluation requires donor maps and stats")
            for target_side in (0, 1):
                replace_side = 1 - target_side if variant == "opposite" else target_side
                events = []
                for i in rows:
                    recipient = dataset[int(i)]
                    donor_index = int(donor_maps[replace_side][int(i)])
                    if donor_index < 0:
                        raise RuntimeError("invalid donor entered evaluation")
                    events.append(hybrid_event(recipient, dataset[donor_index], replace_side, stats))
                batch = collate(events)
                pred = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})[1]
                out[start:start + len(rows), target_side] = pred[:, target_side].detach().cpu().numpy()
    return out


def main() -> None:
    args = parse_args()
    os.environ["GEO_DIM"] = str(args.geometry_dim)
    os.environ["GEO_FEATURES"] = str(args.geometry_features)
    os.environ["GEO_KEY"] = args.geometry_key
    os.environ.setdefault("SEED", "42")
    sys.path.insert(0, str(args.p11_dir))
    import p11_train_full_geometry as p11  # noqa: E402

    base_dataset = p11.Geo16Dataset

    class GeneratorTargetDataset(base_dataset):
        def __init__(self, split: str):
            super().__init__(split, targets_dir=args.targets_dir)

    dataset = GeneratorTargetDataset("validation")
    collate = p11.T.collate_reco_h
    device = torch.device("cuda")
    model = p11.T.RecoModel(args.arm).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stats = json.loads(args.stats.read_text())

    n = len(dataset)
    label = dataset.labels.detach().cpu().numpy().astype(np.int8)
    exact_h = dataset.h.detach().cpu().numpy()
    truth_mode = dataset.truth_modes.detach().cpu().numpy().astype(np.int8)
    global_index = dataset.global_index.detach().cpu().numpy()
    tau_match = np.empty((n, 2, 3), dtype=np.float32)
    reco_mode = np.empty((n, 2), dtype=np.int16)
    ntracks = np.empty((n, 2), dtype=np.int16)
    for i in range(n):
        event = dataset[i]
        tau = event["tau_features"]
        tau_match[i, :, 0] = tau[:, 0].numpy()
        tau_match[i, :, 1] = tau[:, 1].numpy()
        tau_match[i, :, 2] = np.arctan2(tau[:, 2].numpy(), tau[:, 3].numpy())
        reco_mode[i] = event["tau_decay_mode"].numpy()
        raw_ntracks = raw_value(tau[:, 5], stats["tau"], 5).numpy()
        ntracks[i] = np.rint(raw_ntracks).astype(np.int16)
    meta = {"label": label, "reco_mode": reco_mode, "ntracks": ntracks, "tau_match": tau_match}

    maps, map_reports = [], []
    for seed in SEEDS:
        m, report = deranged_maps(meta, seed)
        maps.append(m); map_reports.append(report)
    primary = (label == 0) & (truth_mode[:, 0] == 1) & (truth_mode[:, 1] == 1)
    for m in maps:
        primary &= (m[0] >= 0) & (m[1] >= 0)
    rows = np.flatnonzero(primary)
    if len(rows) < 1000:
        raise RuntimeError(f"unexpectedly small primary cohort: {len(rows)}")

    original = evaluate(model, collate, dataset, rows, device, args.batch_size)
    with np.load(args.reference_predictions) as ref:
        pos = {int(v): i for i, v in enumerate(ref["global_indices"])}
        reference = np.asarray(ref["h_pred"])[[pos[int(global_index[i])] for i in rows]]
    parity_max = float(np.max(np.abs(original - reference)))
    if parity_max > 2e-5:
        raise RuntimeError(f"checkpoint prediction parity failed: {parity_max}")

    # No-op through the same hybrid function must be an exact Python-level identity.
    no_op_events = [hybrid_event(dataset[int(i)], dataset[int(i)], 1, stats) for i in rows[:args.batch_size]]
    batch = collate(no_op_events)
    with torch.inference_mode():
        no_op = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})[1].cpu().numpy()
    no_op_max = float(np.max(np.abs(no_op - original[:len(no_op)])))
    if no_op_max > 1e-7:
        raise RuntimeError(f"self-donor no-op failed: {no_op_max}")

    arrays = {
        "global_indices": global_index[rows],
        "labels": label[rows],
        "truth_modes": truth_mode[rows],
        "h_exact": exact_h[rows],
        "h_original": original,
    }
    for j, m in enumerate(maps):
        arrays[f"donor_minus_seed{j}"] = global_index[m[0][rows]]
        arrays[f"donor_plus_seed{j}"] = global_index[m[1][rows]]
        arrays[f"h_opposite_seed{j}"] = evaluate(
            model, collate, dataset, rows, device, args.batch_size, m, "opposite", stats
        )
        arrays[f"h_target_seed{j}"] = evaluate(
            model, collate, dataset, rows, device, args.batch_size, m, "target", stats
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_rows": n,
        "primary_rows": int(len(rows)),
        "primary_excluded_by_singleton_cells": int(np.sum((label == 0) & (truth_mode[:, 0] == 1) & (truth_mode[:, 1] == 1)) - len(rows)),
        "reference_parity_max_abs": parity_max,
        "self_donor_noop_max_abs": no_op_max,
        "donor_maps": map_reports,
        "output": str(args.output),
        "device": torch.cuda.get_device_name(0),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
