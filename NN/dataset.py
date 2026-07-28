from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from config import PROCESSED_DIR, RANDOM_SEED


PAD_TYPE = 0
EVENT_TYPE = 1
TAU_TYPE = 2
TRACK_TYPE = 3
PFO_TYPE = 4
CLS_TYPE = 5

NO_SIDE = 0
MINUS_SIDE = 1
PLUS_SIDE = 2


class TauSpinDataset(IterableDataset):
    """Stream packed tensor shards without loading the full dataset."""

    def __init__(
        self,
        processed_dir: str | Path = PROCESSED_DIR,
        split: str = "train",
        *,
        shuffle: bool = False,
        balanced: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.shuffle = shuffle
        self.balanced = balanced
        self.seed = seed
        self.epoch = 0
        self.metadata = json.loads(
            (self.processed_dir / "metadata.json").read_text()
        )
        if split not in self.metadata["shards"]:
            raise ValueError(f"Unknown split: {split}")
        self.records = self.metadata["shards"][split]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _records_for_worker(self, sample: str) -> list[dict]:
        records = list(self.records[sample])
        worker = get_worker_info()
        if worker is not None:
            records = records[worker.id :: worker.num_workers]
        return records

    def _class_iterator(
        self,
        sample: str,
        target: int | None,
        rng: random.Random,
    ) -> Iterator[dict[str, torch.Tensor]]:
        records = self._records_for_worker(sample)
        if not records:
            return
        yielded = 0
        while target is None or yielded < target:
            if self.shuffle:
                rng.shuffle(records)
            for record in records:
                shard = torch.load(
                    self.processed_dir / record["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                n_events = int(shard["labels"].shape[0])
                indices = list(range(n_events))
                if self.shuffle:
                    rng.shuffle(indices)
                for index in indices:
                    if target is not None and yielded >= target:
                        return
                    yielded += 1
                    yield extract_event(shard, index)
            if target is None:
                return

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(self.seed + 1009 * self.epoch + worker_id)

        if not self.balanced:
            samples = ["H", "Z"]
            if self.shuffle:
                rng.shuffle(samples)
            for sample in samples:
                yield from self._class_iterator(sample, None, rng)
            return

        local_counts = {
            sample: sum(
                int(record["events"])
                for record in self._records_for_worker(sample)
            )
            for sample in ("H", "Z")
        }
        target = max(local_counts.values())
        h_iter = self._class_iterator("H", target, rng)
        z_iter = self._class_iterator("Z", target, rng)
        pairs = zip(h_iter, z_iter)
        for h_event, z_event in pairs:
            if rng.random() < 0.5:
                yield h_event
                yield z_event
            else:
                yield z_event
                yield h_event

    def __len__(self) -> int:
        counts = self.metadata["counts"][self.split]
        if self.balanced:
            return 2 * max(int(counts["H"]), int(counts["Z"]))
        return int(counts["total"])


def extract_event(
    shard: dict[str, torch.Tensor],
    index: int,
) -> dict[str, torch.Tensor]:
    track_start = int(shard["track_offsets"][index])
    track_stop = int(shard["track_offsets"][index + 1])
    pfo_start = int(shard["pfo_offsets"][index])
    pfo_stop = int(shard["pfo_offsets"][index + 1])
    return {
        "event_features": shard["event_features"][index],
        "tau_features": shard["tau_features"][index],
        "tau_decay_mode": shard["tau_decay_mode"][index],
        "track_features": shard["track_features"][track_start:track_stop],
        "track_sides": shard["track_sides"][track_start:track_stop],
        "pfo_features": shard["pfo_features"][pfo_start:pfo_stop],
        "pfo_sides": shard["pfo_sides"][pfo_start:pfo_stop],
        "label": shard["labels"][index],
        "event_number": shard["event_numbers"][index],
    }


def collate_events(
    events: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not events:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(events)
    event_dim = int(events[0]["event_features"].shape[-1])
    tau_dim = int(events[0]["tau_features"].shape[-1])
    track_dim = int(events[0]["track_features"].shape[-1])
    pfo_dim = int(events[0]["pfo_features"].shape[-1])
    lengths = [
        3 + event["track_features"].shape[0] + event["pfo_features"].shape[0]
        for event in events
    ]
    max_length = max(lengths)

    event_features = torch.zeros(batch_size, max_length, event_dim)
    tau_features = torch.zeros(batch_size, max_length, tau_dim)
    track_features = torch.zeros(batch_size, max_length, track_dim)
    pfo_features = torch.zeros(batch_size, max_length, pfo_dim)
    object_type = torch.full(
        (batch_size, max_length), PAD_TYPE, dtype=torch.int64
    )
    tau_side = torch.full(
        (batch_size, max_length), NO_SIDE, dtype=torch.int64
    )
    decay_mode = torch.zeros(batch_size, max_length, dtype=torch.int64)
    padding_mask = torch.ones(batch_size, max_length, dtype=torch.bool)

    for row, event in enumerate(events):
        position = 0
        event_features[row, position] = event["event_features"]
        object_type[row, position] = EVENT_TYPE
        padding_mask[row, position] = False
        position += 1

        for source_side, side_id in ((0, MINUS_SIDE), (1, PLUS_SIDE)):
            tau_features[row, position] = event["tau_features"][source_side]
            object_type[row, position] = TAU_TYPE
            tau_side[row, position] = side_id
            decay_mode[row, position] = event["tau_decay_mode"][source_side]
            padding_mask[row, position] = False
            position += 1

            track_mask = event["track_sides"] == source_side
            n_tracks = int(track_mask.sum())
            if n_tracks:
                end = position + n_tracks
                track_features[row, position:end] = event["track_features"][
                    track_mask
                ]
                object_type[row, position:end] = TRACK_TYPE
                tau_side[row, position:end] = side_id
                padding_mask[row, position:end] = False
                position = end

            pfo_mask = event["pfo_sides"] == source_side
            n_pfos = int(pfo_mask.sum())
            if n_pfos:
                end = position + n_pfos
                pfo_features[row, position:end] = event["pfo_features"][
                    pfo_mask
                ]
                object_type[row, position:end] = PFO_TYPE
                tau_side[row, position:end] = side_id
                padding_mask[row, position:end] = False
                position = end

        if position != lengths[row]:
            raise RuntimeError("Constructed token length does not match input")

    return {
        "event_features": event_features,
        "tau_features": tau_features,
        "track_features": track_features,
        "pfo_features": pfo_features,
        "object_type": object_type,
        "tau_side": tau_side,
        "decay_mode": decay_mode,
        "padding_mask": padding_mask,
        "labels": torch.stack([event["label"] for event in events]).float(),
        "event_numbers": torch.stack(
            [event["event_number"] for event in events]
        ),
    }
