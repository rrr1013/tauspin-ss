from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uproot


EXPECTED_DIMENSIONS = {"event": 13, "tau": 10, "track": 19, "pfo": 10}
EXPECTED_FEATURE_SET = "absolute-plus-parent-relative-v3"
LABELS = {"H": 1, "Z": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a packed v3 TauSpin dataset.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--matching-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_finite(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    matching_dir = args.matching_dir.resolve()
    metadata_path = processed_dir / "metadata.json"
    stats_path = processed_dir / "stats.json"
    matching_manifest_path = matching_dir / "pt_matching_manifest.json"
    selection_csv_path = matching_dir / "event_selection.csv.gz"
    metadata = json.loads(metadata_path.read_text())
    stats = json.loads(stats_path.read_text())
    matching = json.loads(matching_manifest_path.read_text())
    failures: list[str] = []

    if metadata["feature_set"] != EXPECTED_FEATURE_SET:
        failures.append("feature_set")
    if metadata["feature_dimensions"] != EXPECTED_DIMENSIONS:
        failures.append("feature_dimensions")
    if metadata["counts"] != matching["selected_counts"]:
        failures.append("metadata_matching_counts")
    if metadata["event_selection"]["manifest_sha256"] != sha256_file(
        matching_manifest_path
    ):
        failures.append("matching_manifest_hash")

    selected_counts: Counter[tuple[str, str]] = Counter()
    identities: set[tuple[str, str, int]] = set()
    duplicate_identities = 0
    split_by_identity: dict[tuple[str, str, int], str] = {}
    selected_rows_by_split: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    with gzip.open(selection_csv_path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["selected"]) != 1:
                continue
            selected_rows_by_split[row["split"]].append(dict(row))
            identity = (
                row["sample"],
                row["file_basename"],
                int(row["entry_index"]),
            )
            split = row["split"]
            if identity in identities:
                duplicate_identities += 1
            identities.add(identity)
            previous = split_by_identity.setdefault(identity, split)
            if previous != split:
                failures.append("identity_split_overlap")
            selected_counts[(split, row["sample"])] += 1
    if duplicate_identities:
        failures.append("duplicate_selected_identity")

    shard_records: list[dict[str, Any]] = []
    observed_counts: Counter[tuple[str, str]] = Counter()
    object_counts: Counter[tuple[str, str]] = Counter()
    packed_event_number_parts: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split in ("train", "validation", "test"):
        for sample in ("H", "Z"):
            expected_records = metadata["shards"][split][sample]
            for expected in expected_records:
                path = processed_dir / expected["path"]
                record: dict[str, Any] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "split": split,
                    "sample": sample,
                    "failures": [],
                }
                shard = torch.load(path, map_location="cpu", weights_only=True)
                events = int(shard["labels"].numel())
                tracks = int(shard["track_features"].shape[0])
                pfos = int(shard["pfo_features"].shape[0])
                observed_counts[(split, sample)] += events
                object_counts[(split, "track")] += tracks
                object_counts[(split, "pfo")] += pfos
                packed_event_number_parts[split].append(
                    shard["event_numbers"].numpy()
                )
                expected_shape = {
                    "event_features": (events, EXPECTED_DIMENSIONS["event"]),
                    "tau_features": (events, 2, EXPECTED_DIMENSIONS["tau"]),
                    "tau_decay_mode": (events, 2),
                    "track_features": (tracks, EXPECTED_DIMENSIONS["track"]),
                    "track_sides": (tracks,),
                    "pfo_features": (pfos, EXPECTED_DIMENSIONS["pfo"]),
                    "pfo_sides": (pfos,),
                    "labels": (events,),
                    "event_numbers": (events,),
                    "track_offsets": (events + 1,),
                    "pfo_offsets": (events + 1,),
                }
                if set(shard) != set(expected_shape):
                    record["failures"].append("keys")
                for name, shape in expected_shape.items():
                    if tuple(shard[name].shape) != shape:
                        record["failures"].append(f"shape:{name}")
                    if not tensor_finite(shard[name]):
                        record["failures"].append(f"nonfinite:{name}")
                if events != int(expected["events"]):
                    record["failures"].append("metadata_events")
                if tracks != int(expected["tracks"]):
                    record["failures"].append("metadata_tracks")
                if pfos != int(expected["pfos"]):
                    record["failures"].append("metadata_pfos")
                if path.stat().st_size != int(expected["bytes"]):
                    record["failures"].append("metadata_bytes")
                if not bool((shard["labels"] == LABELS[sample]).all()):
                    record["failures"].append("labels")
                for kind, rows in (("track", tracks), ("pfo", pfos)):
                    offsets = shard[f"{kind}_offsets"]
                    sides = shard[f"{kind}_sides"]
                    if (
                        int(offsets[0]) != 0
                        or int(offsets[-1]) != rows
                        or not bool((offsets[1:] >= offsets[:-1]).all())
                    ):
                        record["failures"].append(f"{kind}_offsets")
                    if sides.numel() and not bool(
                        ((sides == 0) | (sides == 1)).all()
                    ):
                        record["failures"].append(f"{kind}_sides")
                decay = shard["tau_decay_mode"]
                if not bool(
                    ((decay >= 0) & (decay < metadata["tau_decay_num_embeddings"])).all()
                ):
                    record["failures"].append("tau_decay_mode")
                record.update({"events": events, "tracks": tracks, "pfos": pfos})
                shard_records.append(record)
                if record["failures"]:
                    failures.append(f"shard:{expected['path']}")

    for split in ("train", "validation", "test"):
        for sample in ("H", "Z"):
            expected = int(metadata["counts"][split][sample])
            if observed_counts[(split, sample)] != expected:
                failures.append(f"shard_count:{split}:{sample}")
            if selected_counts[(split, sample)] != expected:
                failures.append(f"selection_count:{split}:{sample}")

    repository_nn = matching_dir.parents[2]
    source_paths = {
        Path(path).name: (repository_nn / path).resolve()
        for sample in ("H", "Z")
        for path in matching["input_files"][sample]
    }
    prediction_order_alignment: dict[str, Any] = {}
    sample_order = {"H": 0, "Z": 1}
    for split, rows in selected_rows_by_split.items():
        rows.sort(
            key=lambda row: (
                sample_order[row["sample"]],
                row["file_basename"],
                int(row["entry_index"]),
            )
        )
        expected_numbers = np.empty(len(rows), dtype=np.int64)
        positions_by_file: dict[str, list[tuple[int, int]]] = {}
        for output_index, row in enumerate(rows):
            positions_by_file.setdefault(row["file_basename"], []).append(
                (output_index, int(row["entry_index"]))
            )
        for basename, positions in positions_by_file.items():
            entries = np.asarray([entry for _, entry in positions], dtype=np.int64)
            with uproot.open(source_paths[basename]) as root_file:
                source_numbers = root_file["tauspin"]["eventNumber"].array(
                    library="np"
                )
            for (output_index, _), value in zip(
                positions, source_numbers[entries]
            ):
                expected_numbers[output_index] = int(value)
        packed_numbers = np.concatenate(packed_event_number_parts[split])
        aligned = bool(np.array_equal(expected_numbers, packed_numbers))
        if not aligned:
            failures.append(f"prediction_order:{split}")
        prediction_order_alignment[split] = {
            "events": len(rows),
            "sample_file_entry_order": "H then Z; basename; file-local entry",
            "source_and_packed_event_numbers_equal": aligned,
            "ordered_identity_sha256": hashlib.sha256(
                "\n".join(
                    f"{row['sample']},{row['file_basename']},{row['entry_index']}"
                    for row in rows
                ).encode()
            ).hexdigest(),
            "packed_event_numbers_sha256": hashlib.sha256(
                packed_numbers.tobytes()
            ).hexdigest(),
        }

    train_events = int(metadata["counts"]["train"]["total"])
    if any(int(value) != train_events for value in stats["event"]["count"]):
        failures.append("event_stats_not_train_only")
    if any(int(value) != 2 * train_events for value in stats["tau"]["count"]):
        failures.append("tau_stats_not_train_only")
    if any(
        int(value) != object_counts[("train", "track")]
        for value in stats["track"]["count"]
    ):
        failures.append("track_stats_not_train_only")
    if any(
        int(value) != object_counts[("train", "pfo")]
        for value in stats["pfo"]["count"]
    ):
        failures.append("pfo_stats_not_train_only")
    for kind in ("event", "tau", "track", "pfo"):
        if len(stats[kind]["names"]) != EXPECTED_DIMENSIONS[kind]:
            failures.append(f"stats_dimensions:{kind}")
        for key in ("mean", "std"):
            values = torch.tensor(stats[kind][key], dtype=torch.float64)
            if not bool(torch.isfinite(values).all()):
                failures.append(f"stats_nonfinite:{kind}:{key}")
        if any(float(value) <= 0 for value in stats[kind]["std"]):
            failures.append(f"stats_nonpositive_std:{kind}")

    report = {
        "format_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(),
        "processed_dir": str(processed_dir),
        "metadata_sha256": sha256_file(metadata_path),
        "stats_sha256": sha256_file(stats_path),
        "matching_manifest_sha256": sha256_file(matching_manifest_path),
        "event_selection_csv_sha256": sha256_file(selection_csv_path),
        "feature_set": metadata["feature_set"],
        "feature_dimensions": metadata["feature_dimensions"],
        "counts": metadata["counts"],
        "selected_identity_count": len(identities),
        "selected_identity_duplicates": duplicate_identities,
        "normalization_scope_check": {
            "event_stats_count": stats["event"]["count"][0],
            "tau_stats_count": stats["tau"]["count"][0],
            "train_track_count": object_counts[("train", "track")],
            "train_pfo_count": object_counts[("train", "pfo")],
            "scope": "train only",
        },
        "prediction_order_alignment": prediction_order_alignment,
        "shards": shard_records,
        "failures": sorted(set(failures)),
        "approved_for_gpu_smoke": not failures,
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "metadata_sha256": report["metadata_sha256"],
                "counts": report["counts"],
                "selected_identity_count": report["selected_identity_count"],
                "shard_count": len(shard_records),
                "failures": report["failures"],
                "approved_for_gpu_smoke": report["approved_for_gpu_smoke"],
            },
            indent=2,
        )
    )
    return 0 if report["approved_for_gpu_smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
