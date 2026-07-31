from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


EPOCHS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain selected v3 parameters from scratch for exactly "
            "50 epochs without early stopping or test loading."
        )
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument(
        "--event-selection-manifest", type=Path, required=True
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=Path(__file__).resolve().parent / "final_hpo_worker.py",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: Dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    selection = json.loads(args.selection_json.read_text())
    if selection.get("test_split_loaded"):
        raise RuntimeError("Selection artifact reports test access")
    binding = selection.get("data_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Selection artifact lacks data binding")
    actual = {
        "processed_metadata_sha256": sha256_file(
            args.processed_dir / "metadata.json"
        ),
        "event_selection_manifest_sha256": sha256_file(
            args.event_selection_manifest
        ),
        "snapshot_manifest_sha256": sha256_file(args.snapshot_manifest),
    }
    if actual != binding:
        raise RuntimeError(
            f"Retraining data binding mismatch: {actual} != {binding}"
        )
    command = [
        sys.executable,
        str(args.worker_script.resolve()),
        "--processed-dir",
        str(args.processed_dir.resolve()),
        "--trial-dir",
        str(output_dir),
        "--trial-number",
        str(int(selection["selected_trial_number"])),
        "--parameters-json",
        str(args.selection_json.resolve()),
        "--event-selection-manifest",
        str(args.event_selection_manifest.resolve()),
        "--snapshot-manifest",
        str(args.snapshot_manifest.resolve()),
        "--expected-metadata-sha256",
        binding["processed_metadata_sha256"],
        "--expected-selection-sha256",
        binding["event_selection_manifest_sha256"],
        "--expected-snapshot-sha256",
        binding["snapshot_manifest_sha256"],
        "--max-epochs",
        str(EPOCHS),
        "--objective-start-epoch",
        "2",
        "--early-stop-start-epoch",
        str(EPOCHS),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"50-epoch worker failed with return code {completed.returncode}"
        )
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text())
    if int(result["epochs_completed"]) != EPOCHS:
        raise RuntimeError("Retraining did not complete exactly 50 epochs")
    if result["stopped_early"]:
        raise RuntimeError("Retraining unexpectedly stopped early")
    if result["test_split_loaded"]:
        raise RuntimeError("Retraining loaded the test split")
    invocation = {
        "format_version": 1,
        "purpose": (
            "Selected-parameter v3 retraining from scratch for exactly "
            "50 epochs"
        ),
        "command": command,
        "selection_json": str(args.selection_json.resolve()),
        "selection_json_sha256": sha256_file(args.selection_json),
        "data_binding": binding,
        "epochs_required": EPOCHS,
        "early_stopping_enabled": False,
        "test_split_loaded": False,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
    }
    write_json(output_dir / "retrain_invocation.json", invocation)
    print(json.dumps(invocation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
