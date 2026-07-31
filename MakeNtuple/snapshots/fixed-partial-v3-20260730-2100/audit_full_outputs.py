from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot


EXPECTED_BRANCHES = 84
EXPECTED_PDG_ID = {"H": 25, "Z": 23}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit every fixed-partial-v3 ntuple chunk before dataset construction."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h-cluster", type=int, required=True)
    parser.add_argument("--z-cluster", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_shape_failures(arrays: ak.Array, prefix: str) -> int:
    fields = [name for name in arrays.fields if name.startswith(prefix)]
    if not fields:
        return 1
    reference = ak.to_numpy(ak.num(arrays[fields[0]], axis=1))
    return sum(
        int(np.count_nonzero(ak.to_numpy(ak.num(arrays[name], axis=1)) != reference))
        for name in fields[1:]
    )


def audit_root(path: Path, sample: str) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "open": False,
        "tree": False,
        "entries": 0,
        "branch_count": 0,
        "nonfinite_values": 0,
        "track_shape_failures": 0,
        "pfo_shape_failures": 0,
        "invalid_track_tau_index": 0,
        "invalid_pfo_tau_index": 0,
        "truth_boson_pdg_ids": [],
        "failures": [],
    }
    schema: tuple[tuple[str, str], ...] = ()
    try:
        with uproot.open(path) as root_file:
            record["open"] = True
            if "tauspin" not in root_file:
                record["failures"].append("missing_tauspin_tree")
                return record, schema
            record["tree"] = True
            tree = root_file["tauspin"]
            typenames = tree.typenames()
            schema = tuple(sorted(typenames.items()))
            record["entries"] = int(tree.num_entries)
            record["branch_count"] = len(typenames)
            if record["entries"] <= 0:
                record["failures"].append("empty_tree")
            if record["branch_count"] != EXPECTED_BRANCHES:
                record["failures"].append("unexpected_branch_count")
            pdg_ids: set[int] = set()
            float_fields = [
                name
                for name, typename in typenames.items()
                if "float" in typename or "double" in typename
            ]
            for arrays in tree.iterate(step_size="50 MB", library="ak"):
                record["track_shape_failures"] += vector_shape_failures(
                    arrays, "track_"
                )
                record["pfo_shape_failures"] += vector_shape_failures(
                    arrays, "pfo_"
                )
                for name in float_fields:
                    values = ak.to_numpy(ak.flatten(arrays[name], axis=None))
                    record["nonfinite_values"] += int(
                        np.count_nonzero(~np.isfinite(values))
                    )
                for name, key in (
                    ("track_tauIndex", "invalid_track_tau_index"),
                    ("pfo_tauIndex", "invalid_pfo_tau_index"),
                ):
                    values = ak.to_numpy(ak.flatten(arrays[name], axis=None))
                    record[key] += int(
                        np.count_nonzero((values != 0) & (values != 1))
                    )
                pdg_ids.update(
                    int(value)
                    for value in np.unique(
                        ak.to_numpy(arrays["truth_boson_pdgId"])
                    )
                )
            record["truth_boson_pdg_ids"] = sorted(pdg_ids)
            if record["nonfinite_values"]:
                record["failures"].append("nonfinite_float_values")
            if record["track_shape_failures"]:
                record["failures"].append("track_vector_shape_mismatch")
            if record["pfo_shape_failures"]:
                record["failures"].append("pfo_vector_shape_mismatch")
            if record["invalid_track_tau_index"]:
                record["failures"].append("invalid_track_tau_index")
            if record["invalid_pfo_tau_index"]:
                record["failures"].append("invalid_pfo_tau_index")
            if record["truth_boson_pdg_ids"] != [EXPECTED_PDG_ID[sample]]:
                record["failures"].append("unexpected_truth_boson_pdg_id")
    except Exception as error:
        record["failures"].append(f"open_or_read_error:{type(error).__name__}:{error}")
    return record, schema


def audit_condor(snapshot_dir: Path, sample: str, cluster: int, expected: int) -> dict[str, Any]:
    log_path = snapshot_dir / "logs" / f"{sample}_{cluster}.log"
    text = log_path.read_text()
    termination_count = len(re.findall(r"Job terminated\.", text))
    return_zero_count = len(
        re.findall(r"Normal termination \(return value 0\)", text)
    )
    held_count = len(re.findall(r"Job was held", text))
    error_logs = sorted(snapshot_dir.glob(f"logs/{sample}_*.{cluster}.*.err"))
    stdout_logs = sorted(snapshot_dir.glob(f"logs/{sample}_*.{cluster}.*.out"))
    nonempty_error_logs = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size}
        for path in error_logs
        if path.stat().st_size
    ]
    failures = []
    if termination_count != expected:
        failures.append("termination_count")
    if return_zero_count != expected:
        failures.append("return_zero_count")
    if held_count:
        failures.append("held_jobs")
    if len(error_logs) != expected or nonempty_error_logs:
        failures.append("stderr_logs")
    if len(stdout_logs) != expected:
        failures.append("stdout_logs")
    return {
        "cluster": cluster,
        "expected_jobs": expected,
        "termination_count": termination_count,
        "return_zero_count": return_zero_count,
        "held_count": held_count,
        "stderr_log_count": len(error_logs),
        "stdout_log_count": len(stdout_logs),
        "nonempty_stderr_logs": nonempty_error_logs,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    report: dict[str, Any] = {
        "format_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(),
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_manifest_path": str(manifest_path),
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "samples": {},
        "schema_equal": False,
        "submission_allowed_for_dataset": False,
        "failures": [],
    }
    schemas: dict[str, tuple[tuple[str, str], ...]] = {}
    all_schema: tuple[tuple[str, str], ...] | None = None
    clusters = {"H": args.h_cluster, "Z": args.z_cluster}
    for sample in ("H", "Z"):
        chunk_ids = manifest["chunks"][sample]["chunk_ids"]
        expected_outputs = {
            output_dir / sample / f"{sample}_chunk_{chunk_id}.root"
            for chunk_id in chunk_ids
        }
        actual_outputs = set((output_dir / sample).glob("*.root"))
        chunk_lists = [
            snapshot_dir / "inputs" / f"{sample}_chunk_{chunk_id}.txt"
            for chunk_id in chunk_ids
        ]
        input_paths = [
            line.strip()
            for chunk_list in chunk_lists
            for line in chunk_list.read_text().splitlines()
            if line.strip()
        ]
        manifest_paths = [
            record["path"] for record in records if record["sample"] == sample
        ]
        input_counts = Counter(input_paths)
        coverage_failures = []
        if actual_outputs != expected_outputs:
            coverage_failures.append("output_chunk_set")
        if len(input_paths) != manifest["chunks"][sample]["file_count"]:
            coverage_failures.append("input_file_count")
        if set(input_paths) != set(manifest_paths):
            coverage_failures.append("input_manifest_set")
        if any(count != 1 for count in input_counts.values()):
            coverage_failures.append("input_duplicate")
        root_records = []
        sample_schema: tuple[tuple[str, str], ...] | None = None
        for path in sorted(expected_outputs):
            if not path.is_file():
                continue
            root_record, schema = audit_root(path, sample)
            root_records.append(root_record)
            if sample_schema is None:
                sample_schema = schema
            elif schema != sample_schema:
                root_record["failures"].append("schema_mismatch_within_sample")
        schemas[sample] = sample_schema or ()
        condor = audit_condor(
            snapshot_dir,
            sample,
            clusters[sample],
            len(chunk_ids),
        )
        failures = list(coverage_failures)
        failures.extend(condor["failures"])
        if len(root_records) != len(chunk_ids):
            failures.append("audited_root_count")
        if any(record["failures"] for record in root_records):
            failures.append("root_content")
        report["samples"][sample] = {
            "expected_chunk_count": len(chunk_ids),
            "actual_root_count": len(actual_outputs),
            "input_file_count": len(input_paths),
            "unique_input_file_count": len(input_counts),
            "selected_entries": sum(record["entries"] for record in root_records),
            "output_bytes": sum(path.stat().st_size for path in actual_outputs),
            "coverage_failures": coverage_failures,
            "condor": condor,
            "root_outputs": root_records,
            "failures": failures,
        }
        if failures:
            report["failures"].append(f"{sample}_audit")
        if all_schema is None:
            all_schema = schemas[sample]
    report["schema_equal"] = schemas["H"] == schemas["Z"] and bool(schemas["H"])
    if not report["schema_equal"]:
        report["failures"].append("h_z_schema_mismatch")
    report["submission_allowed_for_dataset"] = not report["failures"]
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "snapshot": report["snapshot_id"],
                "H_selected_entries": report["samples"]["H"]["selected_entries"],
                "Z_selected_entries": report["samples"]["Z"]["selected_entries"],
                "schema_equal": report["schema_equal"],
                "failures": report["failures"],
                "submission_allowed_for_dataset": report[
                    "submission_allowed_for_dataset"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["submission_allowed_for_dataset"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
