"""Rebuild and freeze the deterministic arm-specific H/Z readout.

Only the arm's saved unrotated train predictions are used for fitting.  The
saved validation score cache is used solely for parity after the weights have
been fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    tensor = torch.from_numpy(values.astype(np.float32, copy=False))
    with torch.inference_mode():
        for start in range(0, len(tensor), 512):
            out.append(torch.sigmoid(model(tensor[start:start + 512].to(device))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def wauc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    score, label = score[order], label[order]
    wp = (label == 1).astype(np.float64)
    wn = (label == 0).astype(np.float64)
    _, first = np.unique(score, return_index=True)
    bounds = np.append(first, len(score))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.run / "train_predictions.npz"
    validation_path = args.run / "validation_predictions.npz"
    cache_path = args.run / "readout_scores.npz"
    with np.load(train_path) as train, np.load(validation_path) as validation:
        train_x = np.asarray(train["h_pred"], np.float32).reshape(-1, 6)
        train_y = np.asarray(train["labels"], np.float32)
        validation_x = np.asarray(validation["h_pred"], np.float32).reshape(-1, 6)
        validation_y = np.asarray(validation["labels"], np.int64)
        validation_ids = np.asarray(validation["global_indices"], np.int64)

    device = torch.device("cuda")
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    model = FixedMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4, fused=True)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    for epoch in range(50):
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(20260916 + epoch),
        )
        model.train()
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(values), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    scores = predict(model, validation_x, device)
    with np.load(cache_path) as cache:
        if not np.array_equal(validation_ids, cache["global_indices"]):
            raise RuntimeError("cached readout identity mismatch")
        reference = np.asarray(cache["scores"], np.float64)
    max_abs = float(np.max(np.abs(scores - reference)))
    auc = wauc(scores, validation_y)
    auc_reference = wauc(reference, validation_y)
    if max_abs > 2e-5 or abs(auc - auc_reference) > 1e-8:
        raise RuntimeError(
            f"readout parity failed: max_abs={max_abs}, auc_delta={auc - auc_reference}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": "6-64-64-1 GELU",
        "seed": 20260916,
        "epochs": 50,
        "source_run": str(args.run),
    }
    torch.save(payload, args.output)
    report = {
        "status": "complete",
        "source_run": str(args.run),
        "train_predictions": {"path": str(train_path), "sha256": sha256(train_path)},
        "validation_predictions": {"path": str(validation_path), "sha256": sha256(validation_path)},
        "reference_scores": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "frozen_readout": {"path": str(args.output), "sha256": sha256(args.output)},
        "validation_rows": int(len(scores)),
        "parity_max_abs_score": max_abs,
        "auc": auc,
        "auc_reference": auc_reference,
        "auc_delta": auc - auc_reference,
        "device": torch.cuda.get_device_name(0),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
