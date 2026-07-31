from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import optuna


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


processed_dir = Path(
    "/tmp/rbaba-tauspin-fixed-partial-v3-20260730-2100/processed"
)
selection = Path(
    "/home/rbaba/tauspin-ss/NN/outputs/data-preparation/"
    "fixed-partial-v3-20260730-2100-ptmatched20/"
    "pt_matching_manifest.json"
)
snapshot = Path(
    "/home/rbaba/tauspin-ss/MakeNtuple/snapshots/"
    "fixed-partial-v3-20260730-2100/snapshot_manifest.json"
)
worker = Path("/home/rbaba/tauspin-ss/NN/final_hpo_worker.py")
controller = Path(
    "/home/rbaba/tauspin-ss/MakeNtuple/snapshots/"
    "fixed-partial-v3-20260730-2100/"
    "final_hpo_controller.resume_candidate.py"
)
study_name = "resume-controller-functional-test"

with tempfile.TemporaryDirectory(prefix="v3-controller-resume-") as root:
    output_root = Path(root)
    study_dir = output_root / study_name
    study_dir.mkdir()
    (study_dir / "trials").mkdir()
    (study_dir / "logs").mkdir()
    storage = f"sqlite:///{study_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
    )
    trial = study.ask()
    trial.suggest_categorical("model_profile", ["small"])
    study.tell(trial, 0.5)
    binding = {
        "processed_metadata_path": str(
            (processed_dir / "metadata.json").resolve()
        ),
        "processed_metadata_sha256": sha256_file(
            processed_dir / "metadata.json"
        ),
        "event_selection_manifest_path": str(selection.resolve()),
        "event_selection_manifest_sha256": sha256_file(selection),
        "snapshot_manifest_path": str(snapshot.resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot),
    }
    config = {
        "study_name": study_name,
        "processed_dir": str(processed_dir.resolve()),
        "worker_script": str(worker.resolve()),
        "gpus": [0, 1, 2, 3, 4],
        "max_epochs": 32,
        "objective_start_epoch": 8,
        "early_stop_start_epoch": 20,
        "data_binding": binding,
    }
    (study_dir / "study_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    stop_at = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
    command = [
        sys.executable,
        str(controller),
        "--processed-dir",
        str(processed_dir),
        "--event-selection-manifest",
        str(selection),
        "--snapshot-manifest",
        str(snapshot),
        "--study-name",
        study_name,
        "--output-root",
        str(output_root),
        "--worker-script",
        str(worker),
        "--gpus",
        "0,1,2,3,4",
        "--stop-new-at",
        stop_at,
        "--max-epochs",
        "32",
        "--objective-start-epoch",
        "8",
        "--early-stop-start-epoch",
        "20",
        "--target-total-trials",
        "1",
        "--resume-existing",
        "--status-file",
        "extension_status.json",
        "--completion-file",
        "extension_complete.json",
    ]
    subprocess.run(command, check=True)
    completion = json.loads(
        (study_dir / "extension_complete.json").read_text()
    )
    assert completion["trial_state_counts"] == {"COMPLETE": 1}
    extension = json.loads(
        (study_dir / "controller_extension_config.json").read_text()
    )
    assert extension["prior_trial_count"] == 1
    assert extension["target_total_trials"] == 1
    print("resume controller functional test passed")
