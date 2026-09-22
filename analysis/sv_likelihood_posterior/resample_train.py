"""Regenerate paired 64-draw train neutrino/h samples from the frozen flow.

The historical train posterior retained h draws but not per-draw neutrinos.
This inference-only pass recreates paired neutrino and h draws with the frozen
flow checkpoint and original seed.  It never fits or updates the flow.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-nn", type=Path, required=True)
    parser.add_argument("--flow-targets", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--existing-posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    sys.path[:0] = [
        str(args.source_nn),
        str(args.source_nn / "neutrino_statistics_20260907"),
    ]
    from flow_data import RecoNuDataset, collate_reco_nu  # type: ignore
    from flow_model import NeutrinoFlow, c  # type: ignore
    from hybrid_polarimeter import HybridPolarimeter  # type: ignore

    source = load_npz(
        args.inputs,
        (
            "global_indices", "labels", "file_indices", "entry_indices", "reco_basis",
            "reco_h_pions_lab4", "reco_h_charges", "reco_h_counts",
            "reco_h_neutral_lab4", "reco_h_neutral_counts", "reco_h_mode",
        ),
    )
    existing = load_npz(
        args.existing_posterior,
        ("global_indices", "labels", "h_samples", "all_draws_valid"),
    )
    if not np.array_equal(source["global_indices"], existing["global_indices"]):
        raise RuntimeError("Input/existing posterior global index mismatch")
    if not np.array_equal(source["labels"], existing["labels"]):
        raise RuntimeError("Input/existing posterior label mismatch")

    dataset = RecoNuDataset("train", args.flow_targets)
    if len(dataset) != len(source["global_indices"]):
        raise RuntimeError("Unexpected train dataset size")
    if not np.array_equal(
        dataset.global_index.detach().cpu().numpy(), source["global_indices"]
    ):
        raise RuntimeError("Flow/input train identity mismatch")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = json.loads((args.flow_dir / "config.json").read_text())
    model = NeutrinoFlow(cfg["normalization"]["mean"], cfg["normalization"]["std"])
    checkpoint = torch.load(args.flow_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    polarimeter = HybridPolarimeter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 1)

    events = len(dataset)
    neutrino_path = args.output / "train_nu_samples_xyz.npy"
    h_path = args.output / "train_h_samples.npy"
    valid_path = args.output / "train_draw_valid.npy"
    neutrino_out = np.lib.format.open_memmap(
        neutrino_path, mode="w+", dtype=np.float32, shape=(events, args.samples, 2, 3)
    )
    h_out = np.lib.format.open_memmap(
        h_path, mode="w+", dtype=np.float32, shape=(events, args.samples, 2, 3)
    )
    valid_out = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=np.bool_, shape=(events, args.samples)
    )
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_reco_nu,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    cursor = 0
    max_abs_h_delta = 0.0
    sum_abs_h_delta = 0.0
    finite_delta_count = 0
    started = time.time()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = c.move_batch(cpu_batch, device)
            batch_size = int(batch["labels"].shape[0])
            rows = np.arange(cursor, cursor + batch_size)
            local_t = model.posterior(batch, args.samples)
            if not torch.isfinite(local_t).all():
                raise RuntimeError("Non-finite flow draw")
            local = local_t.detach().cpu().numpy().astype(np.float64)
            basis = source["reco_basis"][rows].astype(np.float64)
            lab_xyz = np.einsum("baji,btaj->btai", basis, local)
            pions = np.broadcast_to(
                source["reco_h_pions_lab4"][rows, None],
                (batch_size, args.samples, 2, 3, 4),
            )
            charges = np.broadcast_to(
                source["reco_h_charges"][rows, None],
                (batch_size, args.samples, 2, 3),
            )
            counts = np.broadcast_to(
                source["reco_h_counts"][rows, None],
                (batch_size, args.samples, 2),
            )
            neutral = np.broadcast_to(
                source["reco_h_neutral_lab4"][rows, None],
                (batch_size, args.samples, 2, 4),
            )
            neutral_counts = np.broadcast_to(
                source["reco_h_neutral_counts"][rows, None],
                (batch_size, args.samples, 2),
            )
            modes = np.broadcast_to(
                source["reco_h_mode"][rows, None],
                (batch_size, args.samples, 2),
            )
            polar = polarimeter(
                pions, charges, counts, neutral, neutral_counts, modes, lab_xyz
            )
            h = np.asarray(polar["h"], dtype=np.float64)
            valid = np.asarray(polar["valid"], dtype=bool) & np.isfinite(h).all(axis=(-1, -2))
            h[~valid] = np.nan
            stored = existing["h_samples"][rows].astype(np.float64)
            finite = np.isfinite(h) & np.isfinite(stored)
            if finite.any():
                delta = np.abs(h[finite] - stored[finite])
                max_abs_h_delta = max(max_abs_h_delta, float(delta.max()))
                sum_abs_h_delta += float(delta.sum())
                finite_delta_count += int(delta.size)
            neutrino_out[rows] = lab_xyz.astype(np.float32)
            h_out[rows] = h.astype(np.float32)
            valid_out[rows] = valid
            cursor += batch_size
            print(json.dumps({"events": cursor, "total": events}), flush=True)
    neutrino_out.flush(); h_out.flush(); valid_out.flush()
    report = {
        "events": events,
        "samples": args.samples,
        "strict_all_draw_valid": int(np.all(valid_out, axis=1).sum()),
        "existing_strict_all_draw_valid": int(existing["all_draws_valid"].sum()),
        "parity_max_abs_h_delta": max_abs_h_delta,
        "parity_mean_abs_h_delta": sum_abs_h_delta / max(finite_delta_count, 1),
        "parity_exact_within_1e-6": bool(max_abs_h_delta <= 1.0e-6),
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "seed": args.seed + 1,
        "flow_checkpoint": str(args.flow_dir / "checkpoint.pt"),
        "test_loaded": False,
        "outputs": {
            "neutrino_xyz": str(neutrino_path),
            "h_samples": str(h_path),
            "draw_valid": str(valid_path),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
