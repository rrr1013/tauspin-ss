#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "snapshot_spec.json"
DEFAULT_OUTPUT = HERE / "audit_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the fixed 2026-07-30 21:00 JST DAOD candidate set. "
            "This command does not discover or add files."
        )
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing audit report explicitly",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def candidate_records(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    base = Path(spec["base_directory"])
    for sample in ("H", "Z"):
        sample_spec = spec["samples"][sample]
        dataset_id = str(sample_spec["dataset_id"])
        indices: List[int] = []
        for first, last in sample_spec["candidate_sub_ranges_inclusive"]:
            if first > last:
                raise ValueError(f"Invalid {sample} candidate range")
            indices.extend(range(int(first), int(last) + 1))
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate {sample} candidate sub index")
        expected = int(sample_spec["expected_candidate_count"])
        if len(indices) != expected:
            raise ValueError(
                f"{sample} candidate count is {len(indices)}, expected {expected}"
            )
        for sub_index in indices:
            directory = (
                f"test_{dataset_id}_sub{sub_index}_n2000"
            )
            yield {
                "sample": sample,
                "dataset_id": dataset_id,
                "sub_index": sub_index,
                "path": str(
                    base / directory / "DAOD_PHYS.pool.root.1"
                ),
                "job_report_path": str(base / directory / "jobReport.json"),
            }


def audit_daod(
    candidate: Dict[str, Any],
    tree_name: str,
    cutoff: datetime,
    root_module: Any,
) -> Dict[str, Any]:
    path = Path(candidate["path"])
    record: Dict[str, Any] = {
        **candidate,
        "status": None,
        "size_bytes": None,
        "mtime_ns": None,
        "mtime_utc": None,
        "mtime_after_cutoff": None,
        "sha256": None,
        "root_open": False,
        "is_zombie": None,
        "collection_tree_exists": False,
        "collection_tree_entries": None,
        "pool_guid": None,
        "error": None,
    }
    if not path.exists():
        record["status"] = "missing"
        return record
    if not path.is_file():
        record["status"] = "not_regular_file"
        return record

    try:
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        record["size_bytes"] = stat.st_size
        record["mtime_ns"] = stat.st_mtime_ns
        record["mtime_utc"] = mtime.isoformat()
        record["mtime_after_cutoff"] = mtime > cutoff.astimezone(timezone.utc)
        record["sha256"] = sha256_file(path)
    except Exception as error:
        record["status"] = "file_read_error"
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    root_file = None
    try:
        root_file = root_module.TFile.Open(str(path), "READ")
        record["root_open"] = bool(root_file)
        if not root_file:
            record["status"] = "unopenable"
            return record
        record["is_zombie"] = bool(root_file.IsZombie())
        if record["is_zombie"]:
            record["status"] = "zombie"
            return record
        tree = root_file.Get(tree_name)
        record["collection_tree_exists"] = bool(tree)
        if not tree:
            record["status"] = "missing_collection_tree"
            return record
        record["collection_tree_entries"] = int(tree.GetEntries())
        params = root_file.Get("##Params")
        if params:
            for index in range(int(params.GetEntries())):
                params.GetEntry(index)
                value = bytes(params.db_string).decode(
                    "utf-8", errors="replace"
                )
                match = re.search(
                    r"\[NAME=FID\]\[VALUE=([0-9A-Fa-f-]+)\]",
                    value,
                )
                if match:
                    record["pool_guid"] = match.group(1).upper()
                    break
        if not record["pool_guid"]:
            record["status"] = "missing_pool_guid"
            return record
        if record["mtime_after_cutoff"]:
            record["status"] = "mtime_after_cutoff"
            return record
        record["status"] = "ok"
        return record
    except Exception as error:
        record["status"] = "root_audit_error"
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        if root_file:
            root_file.Close()


def audit_job_report(
    path: Path,
    daod_name: str,
    cutoff: datetime,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "job_report_exists": False,
        "job_report_size_bytes": None,
        "job_report_mtime_ns": None,
        "job_report_mtime_utc": None,
        "job_report_mtime_after_cutoff": None,
        "job_report_sha256": None,
        "job_report_exit_code": None,
        "job_report_exit_acronym": None,
        "job_report_executors": [],
        "job_report_all_executors_status_ok": False,
        "job_report_daod_name": None,
        "job_report_daod_size_bytes": None,
        "job_report_daod_entries": None,
        "job_report_daod_guid": None,
        "job_report_error": None,
        "job_report_failures": [],
    }
    failures: List[str] = result["job_report_failures"]
    if not path.exists():
        failures.append("job_report_missing")
        return result
    if not path.is_file():
        failures.append("job_report_not_regular_file")
        return result
    result["job_report_exists"] = True
    try:
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        result["job_report_size_bytes"] = stat.st_size
        result["job_report_mtime_ns"] = stat.st_mtime_ns
        result["job_report_mtime_utc"] = mtime.isoformat()
        result["job_report_mtime_after_cutoff"] = (
            mtime > cutoff.astimezone(timezone.utc)
        )
        result["job_report_sha256"] = sha256_file(path)
        report = json.loads(path.read_text())
    except Exception as error:
        failures.append("job_report_read_error")
        result["job_report_error"] = f"{type(error).__name__}: {error}"
        return result

    result["job_report_exit_code"] = report.get("exitCode")
    result["job_report_exit_acronym"] = report.get("exitAcronym")
    if result["job_report_mtime_after_cutoff"]:
        failures.append("job_report_mtime_after_cutoff")
    if result["job_report_exit_code"] != 0:
        failures.append("job_report_exit_code_not_zero")
    if result["job_report_exit_acronym"] != "OK":
        failures.append("job_report_exit_acronym_not_ok")

    executors = report.get("executor")
    if not isinstance(executors, list) or not executors:
        failures.append("job_report_executors_missing")
        executors = []
    result["job_report_executors"] = [
        {
            "name": executor.get("name"),
            "statusOK": executor.get("statusOK"),
            "rc": executor.get("rc"),
        }
        for executor in executors
        if isinstance(executor, dict)
    ]
    result["job_report_all_executors_status_ok"] = bool(executors) and all(
        isinstance(executor, dict) and executor.get("statusOK") is True
        for executor in executors
    )
    if not result["job_report_all_executors_status_ok"]:
        failures.append("job_report_executor_status_not_ok")

    matches: List[Dict[str, Any]] = []
    files = report.get("files")
    output_groups = files.get("output", []) if isinstance(files, dict) else []
    if isinstance(output_groups, list):
        for group in output_groups:
            if not isinstance(group, dict):
                continue
            sub_files = group.get("subFiles", [])
            if not isinstance(sub_files, list):
                continue
            matches.extend(
                item
                for item in sub_files
                if isinstance(item, dict) and item.get("name") == daod_name
            )
    if len(matches) != 1:
        failures.append("job_report_daod_record_not_unique")
        return result
    output = matches[0]
    result["job_report_daod_name"] = output.get("name")
    result["job_report_daod_size_bytes"] = output.get("file_size")
    result["job_report_daod_entries"] = output.get("nentries")
    guid = output.get("file_guid")
    result["job_report_daod_guid"] = (
        str(guid).upper() if guid is not None else None
    )
    for field, failure in (
        ("job_report_daod_size_bytes", "job_report_daod_size_missing"),
        ("job_report_daod_entries", "job_report_daod_entries_missing"),
        ("job_report_daod_guid", "job_report_daod_guid_missing"),
    ):
        if result[field] is None:
            failures.append(failure)
    return result


def audit_candidate(
    candidate: Dict[str, Any],
    tree_name: str,
    cutoff: datetime,
    root_module: Any,
) -> Dict[str, Any]:
    record = audit_daod(candidate, tree_name, cutoff, root_module)
    daod_status = record["status"]
    report = audit_job_report(
        Path(candidate["job_report_path"]),
        Path(candidate["path"]).name,
        cutoff,
    )
    record.update(report)
    failures: List[str] = []
    if daod_status != "ok":
        failures.append(f"daod_{daod_status}")
    failures.extend(report["job_report_failures"])

    comparisons = {
        "size_matches_job_report": (
            record["size_bytes"] is not None
            and record["size_bytes"]
            == record["job_report_daod_size_bytes"]
        ),
        "entries_match_job_report": (
            record["collection_tree_entries"] is not None
            and record["collection_tree_entries"]
            == record["job_report_daod_entries"]
        ),
        "guid_matches_job_report": (
            record["pool_guid"] is not None
            and record["pool_guid"] == record["job_report_daod_guid"]
        ),
    }
    record.update(comparisons)
    for name, matches in comparisons.items():
        if not matches:
            failures.append(name.replace("_matches_job_report", "_mismatch"))
    record["inclusion_failures"] = failures
    record["inclusion_guard_passed"] = not failures
    record["daod_audit_status"] = daod_status
    record["status"] = "ok" if not failures else "inclusion_guard_failed"
    return record


def write_json_atomic(path: Path, document: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(
            f"Audit output already exists: {args.output}; use --replace"
        )
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    cutoff = datetime.fromisoformat(spec["cutoff"])
    if cutoff.tzinfo is None:
        raise ValueError("Snapshot cutoff must include a timezone")

    import ROOT

    ROOT.gROOT.SetBatch(True)
    started = datetime.now(timezone.utc)
    records = [
        audit_candidate(
            candidate,
            str(spec["tree_name"]),
            cutoff,
            ROOT,
        )
        for candidate in candidate_records(spec)
    ]
    for record in records:
        record["is_missing"] = record["daod_audit_status"] == "missing"
        record["is_unopenable"] = (
            record["daod_audit_status"] == "unopenable"
        )
        record["is_broken"] = not record["inclusion_guard_passed"]
    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    expected_total = sum(
        int(item["expected_candidate_count"])
        for item in spec["samples"].values()
    )
    report = {
        "format_version": 1,
        "snapshot_id": spec["snapshot_id"],
        "cutoff": spec["cutoff"],
        "spec_path": str(args.spec.resolve()),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
        "audit_started_utc": started.isoformat(),
        "audit_completed_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "tree_name": spec["tree_name"],
        "candidate_count_expected": expected_total,
        "candidate_count_audited": len(records),
        "status_counts": status_counts,
        "open_major_count": sum(
            count for status, count in status_counts.items() if status != "ok"
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": len(records),
        "status_counts": status_counts,
        "open_major_count": report["open_major_count"],
    }, indent=2))
    return 0 if report["open_major_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
