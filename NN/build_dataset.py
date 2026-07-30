from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterator

import awkward as ak
import numpy as np
import torch
import uproot

from config import (
    EVENTS_PER_SHARD,
    EVENT_FEATURES,
    EVENT_NUMBER_BRANCH,
    EVENT_OUTPUT_FEATURES,
    EVENT_UNSCALED_FEATURES,
    H_INPUT_FILES,
    H_LABEL,
    MISSING_SENTINELS,
    PFO_FEATURES,
    PFO_OUTPUT_FEATURES,
    PFO_TAU_INDEX_BRANCH,
    PFO_UNSCALED_FEATURES,
    PROCESSED_DIR,
    ROOT_STEP_SIZE,
    SPLIT_SEED,
    TAU_CATEGORICAL_FEATURES,
    TAU_CONTINUOUS_FEATURES,
    TAU_OUTPUT_FEATURES,
    TAU_UNSCALED_FEATURES,
    TEST_FRACTION,
    TRACK_FEATURES,
    TRACK_OUTPUT_FEATURES,
    TRACK_TAU_INDEX_BRANCH,
    TRACK_UNSCALED_FEATURES,
    TRAIN_FRACTION,
    TREE_NAME,
    VALIDATION_FRACTION,
    Z_INPUT_FILES,
    Z_LABEL,
)


SPLIT_NAMES = ("train", "validation", "test")
ABSOLUTE_FEATURE_SET = "absolute-v1"
PARENT_RELATIVE_FEATURE_SET = "absolute-plus-parent-relative-v1"
PARENT_RELATIVE_V3_FEATURE_SET = "absolute-plus-parent-relative-v3"

TRACK_PARENT_RELATIVE_INPUTS = (
    "track_dEta",
    "track_dPhi",
    "track_ptFraction",
)
PFO_PARENT_RELATIVE_INPUTS = (
    "pfo_dEta",
    "pfo_dPhi",
    "pfo_ptFraction",
)
TRACK_PARENT_RELATIVE_OUTPUTS = (
    "track_dEta",
    "sin_track_dPhi",
    "cos_track_dPhi",
    "log1p_track_ptFraction",
)
PFO_PARENT_RELATIVE_OUTPUTS = (
    "pfo_dEta",
    "sin_pfo_dPhi",
    "cos_pfo_dPhi",
    "log1p_pfo_ptFraction",
)
EVENT_PARENT_RELATIVE_V3_OUTPUTS = (
    "abs_tau_pair_dEta",
    "sin_tau_pair_dPhi",
    "cos_tau_pair_dPhi",
    "log_tau_minus_over_plus_pt",
    "sin_met_tau_minus_dPhi",
    "cos_met_tau_minus_dPhi",
    "sin_met_tau_plus_dPhi",
    "cos_met_tau_plus_dPhi",
    "met_over_tau_pair_pt",
)

FEATURE_SET = ABSOLUTE_FEATURE_SET


def configure_feature_set(feature_set: str) -> None:
    global FEATURE_SET
    global EVENT_OUTPUT_FEATURES, EVENT_UNSCALED_FEATURES
    global TRACK_FEATURES, TRACK_OUTPUT_FEATURES, TRACK_UNSCALED_FEATURES
    global PFO_FEATURES, PFO_OUTPUT_FEATURES, PFO_UNSCALED_FEATURES

    FEATURE_SET = feature_set
    if feature_set == ABSOLUTE_FEATURE_SET:
        return
    if feature_set not in (
        PARENT_RELATIVE_FEATURE_SET,
        PARENT_RELATIVE_V3_FEATURE_SET,
    ):
        raise ValueError(f"Unknown feature set: {feature_set}")

    TRACK_FEATURES = (*TRACK_FEATURES, *TRACK_PARENT_RELATIVE_INPUTS)
    TRACK_OUTPUT_FEATURES = (
        *TRACK_OUTPUT_FEATURES,
        *TRACK_PARENT_RELATIVE_OUTPUTS,
    )
    TRACK_UNSCALED_FEATURES = (
        *TRACK_UNSCALED_FEATURES,
        "sin_track_dPhi",
        "cos_track_dPhi",
    )
    PFO_FEATURES = (*PFO_FEATURES, *PFO_PARENT_RELATIVE_INPUTS)
    PFO_OUTPUT_FEATURES = (
        *PFO_OUTPUT_FEATURES,
        *PFO_PARENT_RELATIVE_OUTPUTS,
    )
    PFO_UNSCALED_FEATURES = (
        *PFO_UNSCALED_FEATURES,
        "sin_pfo_dPhi",
        "cos_pfo_dPhi",
    )
    if feature_set == PARENT_RELATIVE_V3_FEATURE_SET:
        EVENT_OUTPUT_FEATURES = (
            *EVENT_OUTPUT_FEATURES,
            *EVENT_PARENT_RELATIVE_V3_OUTPUTS,
        )
        EVENT_UNSCALED_FEATURES = (
            *EVENT_UNSCALED_FEATURES,
            "sin_tau_pair_dPhi",
            "cos_tau_pair_dPhi",
            "sin_met_tau_minus_dPhi",
            "cos_met_tau_minus_dPhi",
            "sin_met_tau_plus_dPhi",
            "cos_met_tau_plus_dPhi",
        )


def load_selection_manifest(
    path: Path,
) -> tuple[dict, dict[tuple[str, str], set[int]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError(
            "Unsupported selection manifest format: "
            f"{manifest.get('format_version')}"
        )
    selected: dict[tuple[str, str], set[int]] = {}
    for sample_name in ("H", "Z"):
        sample_entries = manifest["selected_entries"].get(sample_name, {})
        for file_basename, entry_indices in sample_entries.items():
            key = (sample_name, file_basename)
            values = {int(index) for index in entry_indices}
            if len(values) != len(entry_indices):
                raise ValueError(
                    f"Duplicate selected entries for {sample_name}/"
                    f"{file_basename}"
                )
            selected[key] = values
    return manifest, selected


def selection_mask(
    sample_name: str,
    file_basename: str,
    entry_start: int,
    n_events: int,
    selected_entries: dict[tuple[str, str], set[int]] | None,
) -> np.ndarray:
    if selected_entries is None:
        return np.ones(n_events, dtype=bool)
    allowed = selected_entries.get((sample_name, file_basename), set())
    return np.fromiter(
        (
            entry_start + offset in allowed
            for offset in range(n_events)
        ),
        dtype=bool,
        count=n_events,
    )


def find_input_files(pattern: str) -> list[str]:
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ROOT files matched:\n{pattern}")
    return files


def required_branches() -> list[str]:
    return sorted(
        {
            EVENT_NUMBER_BRANCH,
            TRACK_TAU_INDEX_BRANCH,
            PFO_TAU_INDEX_BRANCH,
            *EVENT_FEATURES,
            *TAU_CONTINUOUS_FEATURES,
            *TAU_CATEGORICAL_FEATURES,
            *TRACK_FEATURES,
            *PFO_FEATURES,
        }
    )


def iter_chunks(
    files: list[str],
    sample_name: str,
    label: int,
    step_size: str,
) -> Iterator[tuple[ak.Array, str, int, str, int]]:
    """Yield one ROOT chunk and enough provenance to reproduce its split."""
    for file_path in files:
        with uproot.open(file_path) as root_file:
            tree = root_file[TREE_NAME]
            for arrays, report in tree.iterate(
                expressions=required_branches(),
                step_size=step_size,
                library="ak",
                report=True,
            ):
                yield (
                    arrays,
                    sample_name,
                    label,
                    Path(file_path).name,
                    report.tree_entry_start,
                )


def validate_chunk(arrays: ak.Array) -> None:
    n_events = len(arrays)

    for branch_name in (
        *TAU_CONTINUOUS_FEATURES,
        *TAU_CATEGORICAL_FEATURES,
    ):
        counts = ak.num(arrays[branch_name], axis=1)
        if not bool(ak.all(counts == 2)):
            bad = ak.to_list(ak.local_index(counts)[counts != 2][:10])
            raise ValueError(
                f"{branch_name} does not have exactly two taus. "
                f"Chunk-local bad entries: {bad}"
            )

    track_counts = ak.num(arrays[TRACK_TAU_INDEX_BRANCH], axis=1)
    for branch_name in (TRACK_TAU_INDEX_BRANCH, *TRACK_FEATURES):
        if not bool(
            ak.all(ak.num(arrays[branch_name], axis=1) == track_counts)
        ):
            raise ValueError(f"Track branch length mismatch: {branch_name}")

    pfo_counts = ak.num(arrays[PFO_TAU_INDEX_BRANCH], axis=1)
    for branch_name in (PFO_TAU_INDEX_BRANCH, *PFO_FEATURES):
        if not bool(
            ak.all(ak.num(arrays[branch_name], axis=1) == pfo_counts)
        ):
            raise ValueError(f"PFO branch length mismatch: {branch_name}")

    for branch_name in (
        TRACK_TAU_INDEX_BRANCH,
        PFO_TAU_INDEX_BRANCH,
    ):
        values = ak.flatten(arrays[branch_name], axis=None)
        valid = (values == 0) | (values == 1)
        if not bool(ak.all(valid)):
            invalid = np.unique(ak.to_numpy(values[~valid])).tolist()
            raise ValueError(f"Invalid {branch_name} values: {invalid}")

    for branch_name in (
        *EVENT_FEATURES,
        *TAU_CONTINUOUS_FEATURES,
        *TRACK_FEATURES,
        *PFO_FEATURES,
    ):
        values = ak.flatten(arrays[branch_name], axis=None)
        if not bool(ak.all(np.isfinite(values))):
            raise ValueError(f"NaN or inf found in {branch_name}")
        for sentinel in MISSING_SENTINELS:
            count = int(ak.sum(values == sentinel))
            if count:
                raise ValueError(
                    "Missing sentinel found: "
                    f"branch={branch_name}, value={sentinel}, count={count}"
                )

    for branch_name in (
        "tau_nTracks",
        "tau_nIsolatedTracks",
        "track_numberOfPixelHits",
        "track_numberOfSCTHits",
        "track_numberOfTRTHits",
    ):
        values = ak.flatten(arrays[branch_name], axis=None)
        if bool(ak.any(values < 0)):
            raise ValueError(
                f"Negative count or hit sentinel found in {branch_name}"
            )

    for branch_name in (
        "track_isCore",
        "track_isIsolation",
        "track_isConversion",
        "track_isFake",
        "track_passTrkSelector",
        "pfo_isPi0",
    ):
        values = ak.flatten(arrays[branch_name], axis=None)
        if not bool(ak.all((values == 0) | (values == 1))):
            raise ValueError(f"Non-binary flag found in {branch_name}")

    if n_events == 0:
        raise ValueError("Unexpected empty ROOT chunk")


def split_for_entry(
    sample_name: str,
    file_basename: str,
    entry_index: int,
) -> str:
    # eventNumber is not unique in the merged private-MC ntuples. File basename
    # plus local entry is therefore the stable identity available at this stage.
    key = (
        f"{SPLIT_SEED}:{sample_name}:{file_basename}:{entry_index}"
    ).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    if fraction < TRAIN_FRACTION:
        return "train"
    if fraction < TRAIN_FRACTION + VALIDATION_FRACTION:
        return "validation"
    return "test"


def split_masks(
    sample_name: str,
    file_basename: str,
    entry_start: int,
    n_events: int,
) -> dict[str, np.ndarray]:
    names = np.asarray(
        [
            split_for_entry(
                sample_name,
                file_basename,
                entry_start + offset,
            )
            for offset in range(n_events)
        ]
    )
    return {split: names == split for split in SPLIT_NAMES}


def _flat_numpy(array: ak.Array, dtype=np.float32) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.flatten(array, axis=None)), dtype=dtype)


def _log1p(values: np.ndarray, name: str) -> np.ndarray:
    if np.any(values < 0):
        raise ValueError(f"Negative value found before log1p: {name}")
    return np.log1p(values)


def transform_chunk(arrays: ak.Array) -> dict[str, np.ndarray]:
    """Apply fixed, non-learned feature transformations."""
    met_et = np.asarray(ak.to_numpy(arrays["met_et"]), dtype=np.float32)
    met_phi = np.asarray(ak.to_numpy(arrays["met_phi"]), dtype=np.float32)
    tau_pt = np.asarray(ak.to_numpy(arrays["tau_pt"]), dtype=np.float32)
    tau_eta = np.asarray(ak.to_numpy(arrays["tau_eta"]), dtype=np.float32)
    tau_phi = np.asarray(ak.to_numpy(arrays["tau_phi"]), dtype=np.float32)
    event_columns = [
        _log1p(met_et, "met_et"),
        np.sin(met_phi),
        np.cos(met_phi),
        _log1p(
            np.asarray(
                ak.to_numpy(arrays["met_sumet"]), dtype=np.float32
            ),
            "met_sumet",
        ),
    ]
    if FEATURE_SET == PARENT_RELATIVE_V3_FEATURE_SET:
        if np.any(tau_pt <= 0):
            raise ValueError("Non-positive tau_pt found in Relative-v3")
        tau_pair_dphi = tau_phi[:, 0] - tau_phi[:, 1]
        met_tau_minus_dphi = met_phi - tau_phi[:, 0]
        met_tau_plus_dphi = met_phi - tau_phi[:, 1]
        event_columns.extend(
            (
                np.abs(tau_eta[:, 0] - tau_eta[:, 1]),
                np.sin(tau_pair_dphi),
                np.cos(tau_pair_dphi),
                np.log(tau_pt[:, 0] / tau_pt[:, 1]),
                np.sin(met_tau_minus_dphi),
                np.cos(met_tau_minus_dphi),
                np.sin(met_tau_plus_dphi),
                np.cos(met_tau_plus_dphi),
                met_et / (tau_pt[:, 0] + tau_pt[:, 1]),
            )
        )
    event = np.column_stack(event_columns).astype(np.float32)

    tau = np.stack(
        (
            _log1p(tau_pt, "tau_pt"),
            tau_eta,
            np.sin(tau_phi),
            np.cos(tau_phi),
            _log1p(
                np.asarray(ak.to_numpy(arrays["tau_m"]), dtype=np.float32),
                "tau_m",
            ),
            np.asarray(ak.to_numpy(arrays["tau_nTracks"]), dtype=np.float32),
            np.asarray(
                ak.to_numpy(arrays["tau_nIsolatedTracks"]),
                dtype=np.float32,
            ),
            np.asarray(
                ak.to_numpy(arrays["tau_rnnJetScoreSigTrans"]),
                dtype=np.float32,
            ),
            np.asarray(
                ak.to_numpy(arrays["tau_gntauScoreSigTrans_v0"]),
                dtype=np.float32,
            ),
            np.asarray(
                ak.to_numpy(arrays["tau_vertexDeltaZ"]),
                dtype=np.float32,
            ),
        ),
        axis=-1,
    ).astype(np.float32)

    track_order = ak.argsort(
        arrays["track_pt"], axis=1, ascending=False, stable=True
    )
    track_phi = _flat_numpy(arrays["track_phi"][track_order])
    track_columns = [
        _log1p(
            _flat_numpy(arrays["track_pt"][track_order]),
            "track_pt",
        ),
        _flat_numpy(arrays["track_eta"][track_order]),
        np.sin(track_phi),
        np.cos(track_phi),
        _flat_numpy(arrays["track_charge"][track_order]),
        _flat_numpy(arrays["track_d0"][track_order]),
        _flat_numpy(arrays["track_z0SinTheta"][track_order]),
        _flat_numpy(arrays["track_isCore"][track_order]),
        _flat_numpy(arrays["track_isIsolation"][track_order]),
        _flat_numpy(arrays["track_isConversion"][track_order]),
        _flat_numpy(arrays["track_isFake"][track_order]),
        _flat_numpy(arrays["track_passTrkSelector"][track_order]),
        _flat_numpy(arrays["track_numberOfPixelHits"][track_order]),
        _flat_numpy(arrays["track_numberOfSCTHits"][track_order]),
        _flat_numpy(arrays["track_numberOfTRTHits"][track_order]),
    ]
    if FEATURE_SET in (
        PARENT_RELATIVE_FEATURE_SET,
        PARENT_RELATIVE_V3_FEATURE_SET,
    ):
        track_dphi = _flat_numpy(arrays["track_dPhi"][track_order])
        track_columns.extend(
            (
                _flat_numpy(arrays["track_dEta"][track_order]),
                np.sin(track_dphi),
                np.cos(track_dphi),
                _log1p(
                    _flat_numpy(arrays["track_ptFraction"][track_order]),
                    "track_ptFraction",
                ),
            )
        )
    track = np.column_stack(track_columns).astype(np.float32)
    track_counts = np.asarray(
        ak.to_numpy(ak.num(arrays["track_pt"], axis=1)), dtype=np.int64
    )

    pfo_order = ak.argsort(
        arrays["pfo_pt"], axis=1, ascending=False, stable=True
    )
    pfo_phi = _flat_numpy(arrays["pfo_phi"][pfo_order])
    pfo_columns = [
        _log1p(_flat_numpy(arrays["pfo_pt"][pfo_order]), "pfo_pt"),
        _flat_numpy(arrays["pfo_eta"][pfo_order]),
        np.sin(pfo_phi),
        np.cos(pfo_phi),
        _log1p(_flat_numpy(arrays["pfo_e"][pfo_order]), "pfo_e"),
        _flat_numpy(arrays["pfo_isPi0"][pfo_order]),
    ]
    if FEATURE_SET in (
        PARENT_RELATIVE_FEATURE_SET,
        PARENT_RELATIVE_V3_FEATURE_SET,
    ):
        pfo_dphi = _flat_numpy(arrays["pfo_dPhi"][pfo_order])
        pfo_columns.extend(
            (
                _flat_numpy(arrays["pfo_dEta"][pfo_order]),
                np.sin(pfo_dphi),
                np.cos(pfo_dphi),
                _log1p(
                    _flat_numpy(arrays["pfo_ptFraction"][pfo_order]),
                    "pfo_ptFraction",
                ),
            )
        )
    pfo = np.column_stack(pfo_columns).astype(np.float32)
    pfo_counts = np.asarray(
        ak.to_numpy(ak.num(arrays["pfo_pt"], axis=1)), dtype=np.int64
    )

    return {
        "event_features": event,
        "tau_features": tau,
        "tau_decay_mode_raw": np.asarray(
            ak.to_numpy(arrays["tau_decayMode"]), dtype=np.int64
        ),
        "track_features": track,
        "track_counts": track_counts,
        "track_sides": _flat_numpy(
            arrays[TRACK_TAU_INDEX_BRANCH][track_order], dtype=np.int64
        ),
        "pfo_features": pfo,
        "pfo_counts": pfo_counts,
        "pfo_sides": _flat_numpy(
            arrays[PFO_TAU_INDEX_BRANCH][pfo_order], dtype=np.int64
        ),
        "event_numbers": np.asarray(
            ak.to_numpy(arrays[EVENT_NUMBER_BRANCH]), dtype=np.int64
        ),
    }


def select_events(
    arrays: ak.Array,
    event_mask: np.ndarray,
) -> ak.Array:
    return arrays[event_mask]


@dataclass
class RunningStats:
    names: tuple[str, ...]
    unscaled_names: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.names)
        self.count = np.zeros(size, dtype=np.int64)
        self.total = np.zeros(size, dtype=np.float64)
        self.total_square = np.zeros(size, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        finite = np.isfinite(values)
        if not np.all(finite):
            raise ValueError(f"Non-finite transformed values in {self.names}")
        self.count += values.shape[0]
        self.total += values.sum(axis=0, dtype=np.float64)
        self.total_square += np.square(
            values, dtype=np.float64
        ).sum(axis=0)

    def finish(self) -> dict[str, list[float] | list[int] | list[bool]]:
        if np.any(self.count == 0):
            missing = [
                name
                for name, count in zip(self.names, self.count)
                if count == 0
            ]
            raise ValueError(f"No train values for features: {missing}")
        mean = self.total / self.count
        variance = np.maximum(
            self.total_square / self.count - np.square(mean), 0.0
        )
        std = np.sqrt(variance)
        standardize = np.asarray(
            [name not in self.unscaled_names for name in self.names]
        )
        bad_std = standardize & (std < 1.0e-12)
        constant_names = [
            name for name, bad in zip(self.names, bad_std) if bad
        ]
        # A selected feature can be constant in a small test sample. Keep the
        # schema stable and map it to zero rather than dividing by zero.
        std[bad_std] = 1.0
        mean[~standardize] = 0.0
        std[~standardize] = 1.0
        return {
            "names": list(self.names),
            "count": self.count.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "standardize": standardize.tolist(),
            "constant_features": constant_names,
        }


def make_stats() -> dict[str, RunningStats]:
    return {
        "event": RunningStats(
            EVENT_OUTPUT_FEATURES, EVENT_UNSCALED_FEATURES
        ),
        "tau": RunningStats(TAU_OUTPUT_FEATURES, TAU_UNSCALED_FEATURES),
        "track": RunningStats(
            TRACK_OUTPUT_FEATURES, TRACK_UNSCALED_FEATURES
        ),
        "pfo": RunningStats(PFO_OUTPUT_FEATURES, PFO_UNSCALED_FEATURES),
    }


def compute_train_stats(
    samples: list[tuple[list[str], str, int]],
    step_size: str,
    selected_entries: dict[tuple[str, str], set[int]] | None = None,
) -> tuple[dict[str, dict], list[int], dict[str, dict[str, int]]]:
    running = make_stats()
    decay_modes: set[int] = set()
    counts = {
        split: {"H": 0, "Z": 0, "total": 0}
        for split in SPLIT_NAMES
    }

    for files, sample_name, label in samples:
        for arrays, name, _, basename, entry_start in iter_chunks(
            files, sample_name, label, step_size
        ):
            validate_chunk(arrays)
            masks = split_masks(name, basename, entry_start, len(arrays))
            selected_mask = selection_mask(
                name,
                basename,
                entry_start,
                len(arrays),
                selected_entries,
            )
            for split, mask in masks.items():
                mask &= selected_mask
                count = int(mask.sum())
                counts[split][sample_name] += count
                counts[split]["total"] += count

            train_arrays = select_events(arrays, masks["train"])
            if len(train_arrays) == 0:
                continue
            transformed = transform_chunk(train_arrays)
            running["event"].update(transformed["event_features"])
            running["tau"].update(
                transformed["tau_features"].reshape(
                    -1, len(TAU_OUTPUT_FEATURES)
                )
            )
            running["track"].update(transformed["track_features"])
            running["pfo"].update(transformed["pfo_features"])
            decay_modes.update(
                transformed["tau_decay_mode_raw"].reshape(-1).tolist()
            )

    stats = {name: accumulator.finish() for name, accumulator in running.items()}
    return stats, sorted(decay_modes), counts


def apply_standardization(
    values: np.ndarray,
    stats: dict,
) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def encode_decay_modes(
    raw: np.ndarray,
    mode_to_id: dict[int, int],
) -> np.ndarray:
    encoded = np.zeros_like(raw, dtype=np.int64)
    for mode, category_id in mode_to_id.items():
        encoded[raw == mode] = category_id
    return encoded


def make_offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )


def to_tensor_payload(
    transformed: dict[str, np.ndarray],
    label: int,
    stats: dict[str, dict],
    mode_to_id: dict[int, int],
) -> dict[str, torch.Tensor]:
    n_events = transformed["event_features"].shape[0]
    return {
        "event_features": torch.from_numpy(
            apply_standardization(
                transformed["event_features"], stats["event"]
            )
        ),
        "tau_features": torch.from_numpy(
            apply_standardization(
                transformed["tau_features"], stats["tau"]
            )
        ),
        "tau_decay_mode": torch.from_numpy(
            encode_decay_modes(
                transformed["tau_decay_mode_raw"], mode_to_id
            )
        ),
        "track_features": torch.from_numpy(
            apply_standardization(
                transformed["track_features"], stats["track"]
            )
        ),
        "track_offsets": torch.from_numpy(
            make_offsets(transformed["track_counts"])
        ),
        "track_sides": torch.from_numpy(transformed["track_sides"]),
        "pfo_features": torch.from_numpy(
            apply_standardization(
                transformed["pfo_features"], stats["pfo"]
            )
        ),
        "pfo_offsets": torch.from_numpy(
            make_offsets(transformed["pfo_counts"])
        ),
        "pfo_sides": torch.from_numpy(transformed["pfo_sides"]),
        "labels": torch.full((n_events,), label, dtype=torch.int64),
        "event_numbers": torch.from_numpy(transformed["event_numbers"]),
    }


def payload_size(payload: dict[str, torch.Tensor]) -> int:
    return int(payload["labels"].shape[0])


def slice_payload(
    payload: dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    track_start = int(payload["track_offsets"][start])
    track_stop = int(payload["track_offsets"][stop])
    pfo_start = int(payload["pfo_offsets"][start])
    pfo_stop = int(payload["pfo_offsets"][stop])
    return {
        "event_features": payload["event_features"][start:stop],
        "tau_features": payload["tau_features"][start:stop],
        "tau_decay_mode": payload["tau_decay_mode"][start:stop],
        "track_features": payload["track_features"][track_start:track_stop],
        "track_offsets": (
            payload["track_offsets"][start : stop + 1] - track_start
        ),
        "track_sides": payload["track_sides"][track_start:track_stop],
        "pfo_features": payload["pfo_features"][pfo_start:pfo_stop],
        "pfo_offsets": (
            payload["pfo_offsets"][start : stop + 1] - pfo_start
        ),
        "pfo_sides": payload["pfo_sides"][pfo_start:pfo_stop],
        "labels": payload["labels"][start:stop],
        "event_numbers": payload["event_numbers"][start:stop],
    }


def concatenate_payloads(
    payloads: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if len(payloads) == 1:
        return payloads[0]

    track_offsets = [torch.zeros(1, dtype=torch.int64)]
    pfo_offsets = [torch.zeros(1, dtype=torch.int64)]
    track_total = 0
    pfo_total = 0
    for payload in payloads:
        track_offsets.append(payload["track_offsets"][1:] + track_total)
        pfo_offsets.append(payload["pfo_offsets"][1:] + pfo_total)
        track_total += payload["track_features"].shape[0]
        pfo_total += payload["pfo_features"].shape[0]

    result = {}
    for key in (
        "event_features",
        "tau_features",
        "tau_decay_mode",
        "track_features",
        "track_sides",
        "pfo_features",
        "pfo_sides",
        "labels",
        "event_numbers",
    ):
        result[key] = torch.cat([payload[key] for payload in payloads])
    result["track_offsets"] = torch.cat(track_offsets)
    result["pfo_offsets"] = torch.cat(pfo_offsets)
    return result


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        sample_name: str,
        events_per_shard: int,
    ) -> None:
        self.directory = output_dir / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_name = sample_name
        self.events_per_shard = events_per_shard
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.buffered_events = 0
        self.shard_index = 0
        self.records: list[dict[str, int | str]] = []

    def add(self, payload: dict[str, torch.Tensor]) -> None:
        position = 0
        while position < payload_size(payload):
            room = self.events_per_shard - self.buffered_events
            take = min(room, payload_size(payload) - position)
            piece = slice_payload(payload, position, position + take)
            self.buffer.append(piece)
            self.buffered_events += take
            position += take
            if self.buffered_events == self.events_per_shard:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        payload = concatenate_payloads(self.buffer)
        validate_payload(payload)
        filename = f"{self.sample_name}_{self.shard_index:04d}.pt"
        path = self.directory / filename
        torch.save(payload, path)
        self.records.append(
            {
                "path": str(path.relative_to(self.directory.parent)),
                "events": payload_size(payload),
                "tracks": int(payload["track_features"].shape[0]),
                "pfos": int(payload["pfo_features"].shape[0]),
                "bytes": path.stat().st_size,
            }
        )
        self.shard_index += 1
        self.buffer = []
        self.buffered_events = 0


def validate_payload(payload: dict[str, torch.Tensor]) -> None:
    n_events = payload_size(payload)
    if payload["event_features"].shape[0] != n_events:
        raise ValueError("Event feature count mismatch")
    if tuple(payload["tau_features"].shape[:2]) != (n_events, 2):
        raise ValueError("Tau tensor must have shape [N, 2, features]")
    if tuple(payload["tau_decay_mode"].shape) != (n_events, 2):
        raise ValueError("Tau decay-mode tensor must have shape [N, 2]")
    for kind in ("track", "pfo"):
        offsets = payload[f"{kind}_offsets"]
        features = payload[f"{kind}_features"]
        sides = payload[f"{kind}_sides"]
        if offsets.shape[0] != n_events + 1:
            raise ValueError(f"{kind} offset length mismatch")
        if int(offsets[0]) != 0 or int(offsets[-1]) != features.shape[0]:
            raise ValueError(f"{kind} offsets do not cover all objects")
        if not bool(torch.all(offsets[1:] >= offsets[:-1])):
            raise ValueError(f"{kind} offsets are not monotonic")
        if sides.shape[0] != features.shape[0]:
            raise ValueError(f"{kind} side count mismatch")
    for tensor in payload.values():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("NaN or inf in output payload")


def build_shards(
    samples: list[tuple[list[str], str, int]],
    output_dir: Path,
    step_size: str,
    events_per_shard: int,
    stats: dict[str, dict],
    mode_to_id: dict[int, int],
    selected_entries: dict[tuple[str, str], set[int]] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    writers = {
        (split, name): ShardWriter(
            output_dir, split, name, events_per_shard
        )
        for split in SPLIT_NAMES
        for name in ("H", "Z")
    }

    for files, sample_name, label in samples:
        for arrays, name, _, basename, entry_start in iter_chunks(
            files, sample_name, label, step_size
        ):
            validate_chunk(arrays)
            masks = split_masks(name, basename, entry_start, len(arrays))
            selected_mask = selection_mask(
                name,
                basename,
                entry_start,
                len(arrays),
                selected_entries,
            )
            for split, mask in masks.items():
                mask &= selected_mask
                selected = select_events(arrays, mask)
                if len(selected) == 0:
                    continue
                transformed = transform_chunk(selected)
                payload = to_tensor_payload(
                    transformed, label, stats, mode_to_id
                )
                writers[(split, sample_name)].add(payload)

    for writer in writers.values():
        writer.flush()

    return {
        split: {
            name: writers[(split, name)].records
            for name in ("H", "Z")
        }
        for split in SPLIT_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build packed PyTorch datasets from TauSpin ROOT ntuples."
    )
    parser.add_argument("--h-pattern", default=H_INPUT_FILES)
    parser.add_argument("--z-pattern", default=Z_INPUT_FILES)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--step-size", default=ROOT_STEP_SIZE)
    parser.add_argument(
        "--events-per-shard", type=int, default=EVENTS_PER_SHARD
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output directory if it already exists.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "Optional deterministic event-selection manifest. Events not "
            "listed in the manifest are excluded before train statistics "
            "and shard writing."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            ABSOLUTE_FEATURE_SET,
            PARENT_RELATIVE_FEATURE_SET,
            PARENT_RELATIVE_V3_FEATURE_SET,
        ),
        default=ABSOLUTE_FEATURE_SET,
        help=(
            "Input feature schema. The default preserves the existing "
            "absolute-only dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_feature_set(args.feature_set)
    if args.events_per_shard <= 0:
        raise ValueError("--events-per-shard must be positive")
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{args.output_dir} already exists. Pass --force to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    h_files = find_input_files(args.h_pattern)
    z_files = find_input_files(args.z_pattern)
    samples = [
        (h_files, "H", H_LABEL),
        (z_files, "Z", Z_LABEL),
    ]
    selection_manifest = None
    selected_entries = None
    if args.selection_manifest is not None:
        selection_manifest, selected_entries = load_selection_manifest(
            args.selection_manifest.resolve()
        )

    print(
        f"Inputs: H={len(h_files)} files, Z={len(z_files)} files\n"
        "Pass 1/2: computing train-only statistics..."
    )
    stats, decay_modes, counts = compute_train_stats(
        samples, args.step_size, selected_entries
    )
    mode_to_id = {mode: index + 1 for index, mode in enumerate(decay_modes)}
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )

    print("Pass 2/2: writing packed Tensor shards...")
    shards = build_shards(
        samples,
        args.output_dir,
        args.step_size,
        args.events_per_shard,
        stats,
        mode_to_id,
        selected_entries,
    )

    metadata = {
        "format_version": 1,
        "feature_set": FEATURE_SET,
        "tree_name": TREE_NAME,
        "labels": {"Z": Z_LABEL, "H": H_LABEL},
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "split_identity": (
            "blake2b(seed, sample, ROOT basename, file-local entry index); "
            "eventNumber is retained for diagnostics but is not unique"
        ),
        "counts": counts,
        "feature_names": {
            "event": list(EVENT_OUTPUT_FEATURES),
            "tau": list(TAU_OUTPUT_FEATURES),
            "track": list(TRACK_OUTPUT_FEATURES),
            "pfo": list(PFO_OUTPUT_FEATURES),
        },
        "feature_dimensions": {
            "event": len(EVENT_OUTPUT_FEATURES),
            "tau": len(TAU_OUTPUT_FEATURES),
            "track": len(TRACK_OUTPUT_FEATURES),
            "pfo": len(PFO_OUTPUT_FEATURES),
        },
        "tau_decay_mode_to_id": {
            str(mode): category_id
            for mode, category_id in mode_to_id.items()
        },
        "tau_decay_unknown_id": 0,
        "tau_decay_num_embeddings": len(mode_to_id) + 1,
        "shards": shards,
        "input_files": {
            "H": h_files,
            "Z": z_files,
        },
        "root_step_size": args.step_size,
        "events_per_shard": args.events_per_shard,
        "event_selection": (
            {
                "manifest_path": str(
                    args.selection_manifest.resolve()
                ),
                "manifest_sha256": hashlib.sha256(
                    args.selection_manifest.read_bytes()
                ).hexdigest(),
                "description": selection_manifest.get("description"),
                "selected_counts": selection_manifest.get(
                    "selected_counts"
                ),
            }
            if selection_manifest is not None
            else None
        ),
        "parent_relative_features": (
            {
                "reference": "reconstructed parent tau",
                "track_inputs": list(TRACK_PARENT_RELATIVE_INPUTS),
                "pfo_inputs": list(PFO_PARENT_RELATIVE_INPUTS),
                "event_outputs": (
                    list(EVENT_PARENT_RELATIVE_V3_OUTPUTS)
                    if FEATURE_SET == PARENT_RELATIVE_V3_FEATURE_SET
                    else []
                ),
                "transform": {
                    "dEta": "train-standardized",
                    "dPhi": "sin/cos, not standardized",
                    "ptFraction": "log1p then train-standardized",
                    "tau_pair": (
                        "abs(dEta), sin/cos(dPhi), "
                        "log(tau_minus_pt/tau_plus_pt)"
                    ),
                    "met_relative": (
                        "sin/cos dPhi to each tau and "
                        "MET/(tau_minus_pt+tau_plus_pt)"
                    ),
                },
            }
            if FEATURE_SET in (
                PARENT_RELATIVE_FEATURE_SET,
                PARENT_RELATIVE_V3_FEATURE_SET,
            )
            else None
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    total = sum(counts[split]["total"] for split in SPLIT_NAMES)
    if selection_manifest is None:
        expected = sum(
            uproot.open(f"{path}:{TREE_NAME}").num_entries
            for path in (*h_files, *z_files)
        )
    else:
        expected = sum(
            int(selection_manifest["selected_counts"][split]["total"])
            for split in SPLIT_NAMES
        )
    if total != expected:
        raise ValueError(
            f"Split total {total} != expected selected total {expected}"
        )

    print("\nDataset build complete:")
    for split in SPLIT_NAMES:
        split_counts = counts[split]
        print(
            f"  {split:10s}: {split_counts['total']:6d} "
            f"(H={split_counts['H']}, Z={split_counts['Z']})"
        )
    print(f"  total     : {total}")
    print(f"  output    : {args.output_dir}")


if __name__ == "__main__":
    main()
