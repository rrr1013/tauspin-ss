from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import uproot

from build_dataset import SPLIT_NAMES, split_for_entry
from config import H_INPUT_FILES, SPLIT_SEED, TREE_NAME, Z_INPUT_FILES


MATCHING_SEED = 42


@dataclass(frozen=True)
class Candidate:
    sample: str
    file_path: str
    file_basename: str
    entry_index: int
    split: str
    pt: float
    bin_index: int
    rank: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic H/Z truth-parent-pT matching manifest."
        )
    )
    parser.add_argument("--h-pattern", default=H_INPUT_FILES)
    parser.add_argument("--z-pattern", default=Z_INPUT_FILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=20.0)
    parser.add_argument("--overflow-edge", type=float, default=1000.0)
    parser.add_argument("--matching-seed", type=int, default=MATCHING_SEED)
    return parser.parse_args()


def input_files(pattern: str) -> list[str]:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ROOT files matched: {pattern}")
    return files


def matching_bin(pt: float, bin_width: float, overflow_edge: float) -> int:
    if not math.isfinite(pt) or pt < 0:
        raise ValueError(f"Invalid truth_boson_pt: {pt}")
    if pt >= overflow_edge:
        return int(round(overflow_edge / bin_width))
    return int(math.floor(pt / bin_width))


def candidate_rank(
    seed: int,
    sample: str,
    file_basename: str,
    entry_index: int,
) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{sample}:{file_basename}:{entry_index}".encode(),
        digest_size=16,
    ).digest()


def read_candidates(
    files: list[str],
    sample: str,
    bin_width: float,
    overflow_edge: float,
    seed: int,
) -> list[Candidate]:
    candidates = []
    for file_path in files:
        basename = Path(file_path).name
        with uproot.open(file_path) as root_file:
            values = np.asarray(
                root_file[TREE_NAME]["truth_boson_pt"].array(library="np"),
                dtype=np.float64,
            )
        for entry_index, pt in enumerate(values):
            split = split_for_entry(sample, basename, entry_index)
            candidates.append(
                Candidate(
                    sample=sample,
                    file_path=file_path,
                    file_basename=basename,
                    entry_index=entry_index,
                    split=split,
                    pt=float(pt),
                    bin_index=matching_bin(
                        float(pt), bin_width, overflow_edge
                    ),
                    rank=candidate_rank(
                        seed, sample, basename, entry_index
                    ),
                )
            )
    return candidates


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while (
            stop < len(scores)
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC AUC requires both classes")
    return float(
        (
            ranks[positive].sum()
            - n_positive * (n_positive + 1) / 2
        )
        / (n_positive * n_negative)
    )


def bin_label(
    bin_index: int, bin_width: float, overflow_edge: float
) -> str:
    low = bin_index * bin_width
    if low >= overflow_edge:
        return f"[{overflow_edge:g}, inf)"
    return f"[{low:g}, {low + bin_width:g})"


def write_plot(
    output_path: Path,
    all_candidates: list[Candidate],
    selected_ids: set[tuple[str, str, int]],
    overflow_edge: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return
    before = {
        sample: np.asarray(
            [item.pt for item in all_candidates if item.sample == sample]
        )
        for sample in ("H", "Z")
    }
    after = {
        sample: np.asarray(
            [
                item.pt
                for item in all_candidates
                if item.sample == sample
                and (
                    item.sample,
                    item.file_basename,
                    item.entry_index,
                )
                in selected_ids
            ]
        )
        for sample in ("H", "Z")
    }
    bins = np.arange(0.0, overflow_edge + 20.0, 20.0)
    figure, axes = plt.subplots(
        2, 1, figsize=(9, 9), layout="constrained"
    )
    for axis, values, title in (
        (axes[0], before, "Before pT matching"),
        (axes[1], after, "After 20 GeV-bin pT matching"),
    ):
        for sample, color in (("H", "tab:red"), ("Z", "tab:blue")):
            axis.hist(
                np.clip(values[sample], bins[0], bins[-1]),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=f"{sample} ({len(values[sample]):,})",
            )
        axis.set_title(title)
        axis.set_xlabel("truth parent boson pT [GeV]")
        axis.set_ylabel("Unit-normalized events")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    if arguments.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if arguments.overflow_edge <= 0:
        raise ValueError("--overflow-edge must be positive")
    ratio = arguments.overflow_edge / arguments.bin_width
    if not math.isclose(ratio, round(ratio)):
        raise ValueError("--overflow-edge must be divisible by --bin-width")
    if arguments.output_dir.exists() and any(
        arguments.output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Output directory is not empty: {arguments.output_dir}"
        )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    h_files = input_files(arguments.h_pattern)
    z_files = input_files(arguments.z_pattern)
    candidates = [
        *read_candidates(
            h_files,
            "H",
            arguments.bin_width,
            arguments.overflow_edge,
            arguments.matching_seed,
        ),
        *read_candidates(
            z_files,
            "Z",
            arguments.bin_width,
            arguments.overflow_edge,
            arguments.matching_seed,
        ),
    ]
    grouped: dict[tuple[str, int, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (candidate.split, candidate.bin_index, candidate.sample)
        ].append(candidate)

    selected: set[tuple[str, str, int]] = set()
    bin_rows = []
    all_bins = sorted({item.bin_index for item in candidates})
    for split in SPLIT_NAMES:
        for bin_index in all_bins:
            h_items = grouped.get((split, bin_index, "H"), [])
            z_items = grouped.get((split, bin_index, "Z"), [])
            keep = min(len(h_items), len(z_items))
            for item in sorted(h_items, key=lambda value: value.rank)[:keep]:
                selected.add(
                    (item.sample, item.file_basename, item.entry_index)
                )
            for item in sorted(z_items, key=lambda value: value.rank)[:keep]:
                selected.add(
                    (item.sample, item.file_basename, item.entry_index)
                )
            bin_rows.append(
                {
                    "split": split,
                    "bin_index": bin_index,
                    "bin": bin_label(
                        bin_index,
                        arguments.bin_width,
                        arguments.overflow_edge,
                    ),
                    "h_before": len(h_items),
                    "z_before": len(z_items),
                    "keep_per_class": keep,
                    "h_after": keep,
                    "z_after": keep,
                }
            )

    selected_entries: dict[str, dict[str, list[int]]] = {
        "H": defaultdict(list),
        "Z": defaultdict(list),
    }
    for sample, basename, entry_index in sorted(selected):
        selected_entries[sample][basename].append(entry_index)
    selected_entries = {
        sample: {
            basename: sorted(indices)
            for basename, indices in sorted(files.items())
        }
        for sample, files in selected_entries.items()
    }

    counts_before = {
        split: {
            sample: sum(
                1
                for item in candidates
                if item.split == split and item.sample == sample
            )
            for sample in ("H", "Z")
        }
        for split in SPLIT_NAMES
    }
    counts_after = {
        split: {
            sample: sum(
                1
                for item in candidates
                if item.split == split
                and item.sample == sample
                and (
                    item.sample,
                    item.file_basename,
                    item.entry_index,
                )
                in selected
            )
            for sample in ("H", "Z")
        }
        for split in SPLIT_NAMES
    }
    for counts in (counts_before, counts_after):
        for split in SPLIT_NAMES:
            counts[split]["total"] = (
                counts[split]["H"] + counts[split]["Z"]
            )

    labels_before = np.asarray(
        [1 if item.sample == "H" else 0 for item in candidates]
    )
    pt_before = np.asarray([item.pt for item in candidates])
    matched_candidates = [
        item
        for item in candidates
        if (item.sample, item.file_basename, item.entry_index) in selected
    ]
    labels_after = np.asarray(
        [1 if item.sample == "H" else 0 for item in matched_candidates]
    )
    pt_after = np.asarray([item.pt for item in matched_candidates])

    input_digest = hashlib.sha256()
    for path in (*h_files, *z_files):
        input_digest.update(path.encode())
        input_digest.update(b"\n")
    manifest = {
        "format_version": 1,
        "description": (
            "Split first with the existing deterministic split, then match "
            "H and Z counts independently in each split and 20 GeV truth "
            "parent boson pT bin. The >=1000 GeV region is one overflow bin."
        ),
        "tree_name": TREE_NAME,
        "truth_pt_branch": "truth_boson_pt",
        "split_seed": SPLIT_SEED,
        "matching_seed": arguments.matching_seed,
        "bin_width_gev": arguments.bin_width,
        "overflow_edge_gev": arguments.overflow_edge,
        "input_files_sha256": input_digest.hexdigest(),
        "input_files": {"H": h_files, "Z": z_files},
        "counts_before": counts_before,
        "selected_counts": counts_after,
        "selected_entries": selected_entries,
    }
    manifest_path = arguments.output_dir / "pt_matching_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with (arguments.output_dir / "bin_counts.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=bin_rows[0].keys())
        writer.writeheader()
        writer.writerows(bin_rows)

    with gzip.open(
        arguments.output_dir / "event_selection.csv.gz", "wt", newline=""
    ) as stream:
        fields = (
            "sample",
            "file_basename",
            "entry_index",
            "split",
            "truth_boson_pt_gev",
            "bin_index",
            "selected",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "sample": item.sample,
                    "file_basename": item.file_basename,
                    "entry_index": item.entry_index,
                    "split": item.split,
                    "truth_boson_pt_gev": item.pt,
                    "bin_index": item.bin_index,
                    "selected": int(
                        (
                            item.sample,
                            item.file_basename,
                            item.entry_index,
                        )
                        in selected
                    ),
                }
            )

    summary = {
        "completed": True,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "retention": {
            sample: (
                sum(counts_after[split][sample] for split in SPLIT_NAMES)
                / sum(
                    counts_before[split][sample]
                    for split in SPLIT_NAMES
                )
            )
            for sample in ("H", "Z")
        },
        "pt_only_auc_before": roc_auc(labels_before, pt_before),
        "pt_only_auc_after": roc_auc(labels_after, pt_after),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_plot(
        arguments.output_dir / "pt_matching_summary.pdf",
        candidates,
        selected,
        arguments.overflow_edge,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
