#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from audit_snapshot import candidate_records


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "snapshot_spec.json"
DEFAULT_AUDIT = HERE / "audit_report.json"
PROJECT_DIR = Path("/home/rbaba/tauspin-ss/MakeNtuple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze audited fixed-partial-v3 inputs and create chunk/submit "
            "files. Refuses any incomplete or non-ok audit."
        )
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
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


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def submit_document(
    snapshot_id: str,
    sample: str,
    chunk_ids: List[str],
    snapshot_dir: Path,
) -> str:
    chunk_lines = "\n".join(chunk_ids)
    return f"""universe = vanilla
executable = {PROJECT_DIR}/run_condor.sh
arguments = {sample} {snapshot_dir}/inputs/{sample}_chunk_$(chunk).txt {snapshot_id}
output = {snapshot_dir}/logs/{sample}_$(chunk).$(ClusterId).$(ProcId).out
error = {snapshot_dir}/logs/{sample}_$(chunk).$(ClusterId).$(ProcId).err
log = {snapshot_dir}/logs/{sample}_$(ClusterId).log
request_cpus = 1
request_memory = 5GB
getenv = True

queue chunk from (
{chunk_lines}
)
"""


def main() -> int:
    args = parse_args()
    spec_bytes = args.spec.read_bytes()
    audit_bytes = args.audit.read_bytes()
    spec: Dict[str, Any] = json.loads(spec_bytes)
    audit: Dict[str, Any] = json.loads(audit_bytes)
    if audit["snapshot_id"] != spec["snapshot_id"]:
        raise ValueError("Audit snapshot_id does not match the spec")
    if audit["spec_sha256"] != hashlib.sha256(spec_bytes).hexdigest():
        raise ValueError("Audit was produced from a different spec")
    if int(audit["open_major_count"]) != 0:
        raise RuntimeError("Open Major is not zero; snapshot cannot be frozen")

    expected = list(candidate_records(spec))
    records = list(audit["records"])
    expected_keys = [
        (
            item["sample"],
            int(item["sub_index"]),
            item["path"],
            item["job_report_path"],
        )
        for item in expected
    ]
    actual_keys = [
        (
            item["sample"],
            int(item["sub_index"]),
            item["path"],
            item.get("job_report_path"),
        )
        for item in records
    ]
    if actual_keys != expected_keys:
        raise ValueError("Audit records do not exactly match the candidate set")
    if any(item["status"] != "ok" for item in records):
        raise RuntimeError("Audit contains a non-ok candidate")
    if any(not item.get("inclusion_guard_passed") for item in records):
        raise RuntimeError("Audit contains a failed inclusion guard")
    if any(not item.get("sha256") for item in records):
        raise RuntimeError("Audit contains a candidate without SHA-256")
    if any(not item.get("job_report_sha256") for item in records):
        raise RuntimeError(
            "Audit contains a candidate without jobReport SHA-256"
        )
    if len({item["path"] for item in records}) != len(records):
        raise RuntimeError("Audit contains duplicate paths")

    snapshot_dir = args.output_dir.resolve()
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    inputs_dir = snapshot_dir / "inputs"
    if manifest_path.exists() or inputs_dir.exists():
        raise FileExistsError(
            "Frozen manifest or inputs already exist; refusing to overwrite"
        )
    inputs_dir.mkdir(parents=True)
    (snapshot_dir / "logs").mkdir(exist_ok=True)

    chunk_size = int(spec["chunk_size"])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_summary: Dict[str, Any] = {}
    for sample in ("H", "Z"):
        paths = [
            item["path"] for item in records if item["sample"] == sample
        ]
        (inputs_dir / f"{sample}_all.txt").write_text(
            "\n".join(paths) + "\n"
        )
        chunk_ids: List[str] = []
        for chunk_index, chunk_paths in enumerate(chunks(paths, chunk_size)):
            chunk_id = f"{chunk_index:03d}"
            chunk_ids.append(chunk_id)
            (inputs_dir / f"{sample}_chunk_{chunk_id}.txt").write_text(
                "\n".join(chunk_paths) + "\n"
            )
        (snapshot_dir / f"submit_{sample}.sub").write_text(
            submit_document(
                str(spec["snapshot_id"]),
                sample,
                chunk_ids,
                snapshot_dir,
            )
        )
        chunk_summary[sample] = {
            "file_count": len(paths),
            "chunk_count": len(chunk_ids),
            "chunk_size": chunk_size,
            "chunk_ids": chunk_ids,
        }

    manifest = {
        "format_version": 1,
        "snapshot_id": spec["snapshot_id"],
        "cutoff": spec["cutoff"],
        "tree_name": spec["tree_name"],
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "audit_report_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "audit_tool_sha256": audit["audit_tool_sha256"],
        "candidate_count": len(records),
        "status_counts": audit["status_counts"],
        "chunks": chunk_summary,
        "required_preflight_command": (
            f"python {snapshot_dir}/preflight_snapshot.py "
            f"--manifest {manifest_path}"
        ),
        "excluded_at_cutoff": {
            sample: spec["samples"][sample]["excluded_at_cutoff"]
            for sample in ("H", "Z")
        },
        "records": records,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "candidate_count": len(records),
        "chunks": chunk_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
