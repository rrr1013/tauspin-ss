from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import BATCH_SIZE, PROCESSED_DIR
from dataset import TauSpinDataset, collate_events
from model import TauSpinTransformer


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the prepared TauSpin dataset and Transformer. "
            "The full training loop is intentionally the next step."
        )
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        (args.processed_dir / "metadata.json").read_text()
    )
    dataset = TauSpinDataset(
        args.processed_dir,
        split="train",
        shuffle=True,
        balanced=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_events,
    )
    batch = next(iter(loader))

    device = choose_device(args.device)
    model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
    ).to(device)
    batch = move_batch(batch, device)

    model.train()
    logits = model(batch)
    loss = nn.BCEWithLogitsLoss()(logits, batch["labels"])
    loss.backward()

    print("Smoke test passed")
    print(f"  device: {device}")
    print(f"  batch size: {batch['labels'].shape[0]}")
    print(f"  tokens before CLS: {batch['object_type'].shape[1]}")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  finite loss: {float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
