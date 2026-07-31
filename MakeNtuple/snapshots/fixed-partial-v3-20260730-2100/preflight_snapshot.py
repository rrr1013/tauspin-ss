#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "snapshot_manifest.json"
DEFAULT_OUTPUT = HERE / "preflight_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Immediately before ntuple submission, recheck every accepted "
            "DAOD SHA-256/size/mtime and jobReport SHA-256."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing preflight report explicitly",
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


def check_record(record: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "sample": record["sample"],
        "sub_index": record["sub_index"],
        "path": record["path"],
        "job_report_path": record["job_report_path"],
        "passed": False,
        "failures": [],
        "observed_size_bytes": None,
        "observed_mtime_ns": None,
        "observed_sha256": None,
        "observed_job_report_sha256": None,
    }
    failures: List[str] = result["failures"]
    if not record.get("inclusion_guard_passed"):
        failures.append("manifest_inclusion_guard_not_passed")

    daod = Path(record["path"])
    if not daod.is_file():
        failures.append("daod_missing_or_not_regular")
    else:
        try:
            stat = daod.stat()
            result["observed_size_bytes"] = stat.st_size
            result["observed_mtime_ns"] = stat.st_mtime_ns
            result["observed_sha256"] = sha256_file(daod)
        except Exception as error:
            failures.append(
                f"daod_read_error:{type(error).__name__}:{error}"
            )
        if result["observed_size_bytes"] != record.get("size_bytes"):
            failures.append("daod_size_mismatch")
        if result["observed_mtime_ns"] != record.get("mtime_ns"):
            failures.append("daod_mtime_mismatch")
        if result["observed_sha256"] != record.get("sha256"):
            failures.append("daod_sha256_mismatch")

    job_report = Path(record["job_report_path"])
    if not job_report.is_file():
        failures.append("job_report_missing_or_not_regular")
    else:
        try:
            result["observed_job_report_sha256"] = sha256_file(job_report)
        except Exception as error:
            failures.append(
                f"job_report_read_error:{type(error).__name__}:{error}"
            )
        if (
            result["observed_job_report_sha256"]
            != record.get("job_report_sha256")
        ):
            failures.append("job_report_sha256_mismatch")

    result["passed"] = not failures
    return result


def write_json_atomic(path: Path, document: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(
            f"Preflight output already exists: {args.output}; use --replace"
        )
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [check_record(record) for record in manifest["records"]]
    failed = [record for record in records if not record["passed"]]
    report = {
        "format_version": 1,
        "snapshot_id": manifest["snapshot_id"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "preflight_tool_sha256": sha256_file(Path(__file__).resolve()),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "candidate_count": len(records),
        "passed_count": len(records) - len(failed),
        "failed_count": len(failed),
        "submission_allowed": not failed,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": len(records),
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
        "submission_allowed": report["submission_allowed"],
    }, indent=2))
    if failed:
        print(json.dumps({"failed_records": failed}, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
