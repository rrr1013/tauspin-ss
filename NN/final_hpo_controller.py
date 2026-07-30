from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState


PROFILES = ["small", "current", "deep", "wide", "large"]
SCHEDULES = ["constant", "cosine_warmup5"]
BASELINE = {
    "learning_rate": 1.0e-4,
    "dropout": 0.1,
    "schedule_profile": "constant",
}
INITIAL_TRIALS = [
    {**BASELINE, "model_profile": profile} for profile in PROFILES
] + [
    {
        **BASELINE,
        "model_profile": "current",
        "schedule_profile": "cosine_warmup5",
    },
    {
        **BASELINE,
        "model_profile": "wide",
        "schedule_profile": "cosine_warmup5",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Central ask/tell controller for two-GPU final HPO."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--stop-new-at", required=True)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--objective-start-epoch", type=int, default=8)
    parser.add_argument("--early-stop-start-epoch", type=int, default=20)
    parser.add_argument("--target-total-trials", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "model_profile": trial.suggest_categorical(
            "model_profile", PROFILES
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 7.0e-5, 2.0e-4, log=True
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
        "schedule_profile": trial.suggest_categorical(
            "schedule_profile", SCHEDULES
        ),
    }


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective": trial.value,
            **trial.params,
        }
        row.update(
            {
                f"user_{key}": value
                for key, value in trial.user_attrs.items()
            }
        )
        rows.append(row)
    return rows


def save_study(study: optuna.Study, study_dir: Path) -> None:
    rows = trial_rows(study)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with (study_dir / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [
        row for row in rows if row["state"] == "COMPLETE"
    ]
    complete.sort(
        key=lambda row: float(row["objective"]), reverse=True
    )
    write_json(study_dir / "ranking.json", complete)


def controller_status(
    *,
    study: optuna.Study,
    active: dict[int, dict[str, Any]],
    stop_new_at: datetime,
    started_at: datetime,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return {
        "controller_pid": os.getpid(),
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now().astimezone().isoformat(),
        "stop_new_at": stop_new_at.isoformat(),
        "trial_state_counts": counts,
        "active": [
            {
                "gpu": gpu,
                "trial_number": item["trial"].number,
                "pid": item["process"].pid,
                "parameters": item["parameters"],
                "trial_dir": str(item["trial_dir"]),
            }
            for gpu, item in sorted(active.items())
        ],
    }


def main() -> None:
    args = parse_args()
    gpus = [int(item) for item in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("--gpus must contain distinct indices")
    stop_new_at = datetime.fromisoformat(args.stop_new_at)
    if stop_new_at.tzinfo is None:
        raise ValueError("--stop-new-at must include a timezone")
    started_at = datetime.now().astimezone()
    study_dir = args.output_root / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(exist_ok=True)
    (study_dir / "logs").mkdir(exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'optuna.db').resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=7,
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    if not study.trials:
        for parameters in INITIAL_TRIALS:
            study.enqueue_trial(
                parameters,
                user_attrs={"source": "required_initial_trial"},
                skip_if_exists=True,
            )
    write_json(
        study_dir / "study_config.json",
        {
            "purpose": (
                "Final overnight HPO for summer-school analysis on the "
                "frozen fixed-partial-v2 snapshot."
            ),
            "study_name": args.study_name,
            "processed_dir": str(args.processed_dir.resolve()),
            "worker_script": str(args.worker_script.resolve()),
            "gpus": gpus,
            "stop_new_at": stop_new_at.isoformat(),
            "max_epochs": args.max_epochs,
            "objective_start_epoch": args.objective_start_epoch,
            "early_stop_start_epoch": args.early_stop_start_epoch,
            "target_total_trials": args.target_total_trials,
            "search_space": {
                "model_profile": PROFILES,
                "learning_rate": {
                    "low": 7.0e-5,
                    "high": 2.0e-4,
                    "log": True,
                },
                "dropout": {"low": 0.0, "high": 0.25},
                "schedule_profile": SCHEDULES,
            },
            "fixed": {
                "feature_set": (
                    "absolute-plus-parent-relative-v3"
                ),
                "optimizer": "AdamW",
                "weight_decay": 1.0e-4,
                "batch_size": 128,
                "seed": 42,
                "workers": 2,
                "precision": "TF32",
                "test_split_policy": "never load during HPO",
            },
            "sampler": {
                "name": "TPESampler",
                "seed": 42,
                "n_startup_trials": 7,
                "constant_liar": True,
            },
            "pruner": "NopPruner",
            "initial_trials": INITIAL_TRIALS,
        },
    )

    active: dict[int, dict[str, Any]] = {}
    log_stream = (study_dir / "logs" / "controller_events.jsonl").open(
        "a", buffering=1
    )

    def log_event(kind: str, **payload: Any) -> None:
        log_stream.write(
            json.dumps(
                {
                    "time": datetime.now().astimezone().isoformat(),
                    "kind": kind,
                    **payload,
                }
            )
            + "\n"
        )

    try:
        while True:
            for gpu, item in list(active.items()):
                process = item["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                item["stdout"].close()
                item["stderr"].close()
                trial = item["trial"]
                result_path = item["trial_dir"] / "result.json"
                if returncode == 0 and result_path.is_file():
                    result = json.loads(result_path.read_text())
                    value = float(result["objective"])
                    for key in (
                        "best_center_epoch",
                        "best_single_validation_auc",
                        "best_single_validation_auc_epoch",
                        "minimum_validation_loss",
                        "minimum_validation_loss_epoch",
                        "epochs_completed",
                        "optimizer_steps",
                        "stopped_early",
                        "stop_reason",
                        "elapsed_seconds",
                        "parameter_counts",
                        "checkpoint_sha256",
                        "test_split_loaded",
                    ):
                        trial.set_user_attr(key, result.get(key))
                    study.tell(trial, value)
                    log_event(
                        "trial_complete",
                        gpu=gpu,
                        trial_number=trial.number,
                        objective=value,
                        returncode=returncode,
                    )
                else:
                    trial.set_user_attr("returncode", returncode)
                    trial.set_user_attr(
                        "result_json_exists", result_path.is_file()
                    )
                    study.tell(trial, state=TrialState.FAIL)
                    log_event(
                        "trial_failed",
                        gpu=gpu,
                        trial_number=trial.number,
                        returncode=returncode,
                    )
                del active[gpu]
                save_study(study, study_dir)

            completed = sum(
                trial.state == TrialState.COMPLETE
                for trial in study.trials
            )
            finished = sum(
                trial.state in (TrialState.COMPLETE, TrialState.FAIL)
                for trial in study.trials
            )
            can_start = (
                datetime.now().astimezone() < stop_new_at
                and finished + len(active) < args.target_total_trials
            )
            if can_start:
                for gpu in gpus:
                    if gpu in active:
                        continue
                    if (
                        finished + len(active)
                        >= args.target_total_trials
                    ):
                        break
                    trial = study.ask()
                    parameters = suggest_parameters(trial)
                    trial_dir = (
                        study_dir / "trials" / f"trial_{trial.number:03d}"
                    )
                    parameter_path = (
                        study_dir
                        / "trials"
                        / f"trial_{trial.number:03d}_parameters.json"
                    )
                    write_json(parameter_path, parameters)
                    stdout_path = (
                        study_dir
                        / "logs"
                        / f"trial_{trial.number:03d}.stdout.log"
                    )
                    stderr_path = (
                        study_dir
                        / "logs"
                        / f"trial_{trial.number:03d}.stderr.log"
                    )
                    stdout = stdout_path.open("w")
                    stderr = stderr_path.open("w")
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    command = [
                        sys.executable,
                        str(args.worker_script.resolve()),
                        "--processed-dir",
                        str(args.processed_dir.resolve()),
                        "--trial-dir",
                        str(trial_dir.resolve()),
                        "--trial-number",
                        str(trial.number),
                        "--parameters-json",
                        str(parameter_path.resolve()),
                        "--max-epochs",
                        str(args.max_epochs),
                        "--objective-start-epoch",
                        str(args.objective_start_epoch),
                        "--early-stop-start-epoch",
                        str(args.early_stop_start_epoch),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=args.worker_script.resolve().parent,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    active[gpu] = {
                        "trial": trial,
                        "parameters": parameters,
                        "trial_dir": trial_dir,
                        "process": process,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                    log_event(
                        "trial_started",
                        gpu=gpu,
                        trial_number=trial.number,
                        pid=process.pid,
                        parameters=parameters,
                    )

            status = controller_status(
                study=study,
                active=active,
                stop_new_at=stop_new_at,
                started_at=started_at,
            )
            write_json(study_dir / "controller_status.json", status)
            save_study(study, study_dir)
            if (
                datetime.now().astimezone() >= stop_new_at
                or finished >= args.target_total_trials
            ) and not active:
                break
            time.sleep(args.poll_seconds)
    finally:
        log_stream.close()

    write_json(
        study_dir / "controller_complete.json",
        {
            **controller_status(
                study=study,
                active=active,
                stop_new_at=stop_new_at,
                started_at=started_at,
            ),
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )


if __name__ == "__main__":
    main()
