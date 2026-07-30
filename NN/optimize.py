from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import PROCESSED_DIR, RANDOM_SEED
from hpo_utils import (
    MODEL_PROFILES,
    configure_tf32,
    create_list_loader,
    create_model,
    create_streaming_loader,
    environment_information,
    evaluate_model,
    json_ready,
    learning_rate_for_step,
    load_or_create_validation_manifest,
    make_checkpoint,
    set_optimizer_learning_rate,
    sha256_bytes,
    sha256_file,
    shutdown_loader_workers,
    strip_evaluation_arrays,
    write_json,
)
from train import (
    choose_device,
    move_batch,
    require_finite,
    require_finite_gradients,
    set_random_seed,
)


DEFAULT_STUDY_NAME = "old-sample-joint-hpo-smoke-v1"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "hpo"

BATCH_SIZE = 128
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
MAX_STEPS = 1000
SUBSET_EVAL_EVERY = 100
FULL_EVAL_EVERY = 500
PRUNING_WARMUP_STEPS = 600
VALIDATION_PER_CLASS = 2048

SEARCH_SPACE = {
    "model_profile": list(MODEL_PROFILES),
    "learning_rate": {
        "low": 2.0e-5,
        "high": 5.0e-4,
        "log": True,
    },
    "dropout": {"low": 0.0, "high": 0.35},
    "weight_decay": {
        "low": 1.0e-6,
        "high": 3.0e-3,
        "log": True,
    },
    "warmup_ratio": {"low": 0.0, "high": 0.20},
    "scheduler": ["constant", "cosine"],
}

BASELINE = {
    "model_profile": "current",
    "learning_rate": 1.0e-4,
    "dropout": 0.1,
    "weight_decay": 1.0e-4,
    "warmup_ratio": 0.0,
    "scheduler": "constant",
}

INITIAL_TRIALS = [
    BASELINE,
    {**BASELINE, "model_profile": "small"},
    {**BASELINE, "model_profile": "deep"},
    {**BASELINE, "model_profile": "wide"},
    {**BASELINE, "model_profile": "large"},
    {**BASELINE, "learning_rate": 2.0e-5},
    {**BASELINE, "learning_rate": 5.0e-4},
    {**BASELINE, "dropout": 0.0},
    {**BASELINE, "dropout": 0.3},
    {
        **BASELINE,
        "warmup_ratio": 0.1,
        "scheduler": "cosine",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Joint Optuna HPO for TauSpin Transformer model size and "
            "training hyperparameters."
        )
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR
    )
    parser.add_argument(
        "--study-name", default=DEFAULT_STUDY_NAME
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--target-total-trials", type=int, required=True
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--subset-eval-every-steps",
        type=int,
        default=SUBSET_EVAL_EVERY,
    )
    parser.add_argument(
        "--full-eval-every-steps",
        type=int,
        default=FULL_EVAL_EVERY,
    )
    parser.add_argument(
        "--pruning-warmup-steps",
        type=int,
        default=PRUNING_WARMUP_STEPS,
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=None,
        help="Physical GPU exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
    )
    return parser.parse_args()


def setup_logging(study_dir: Path) -> logging.Logger:
    log_dir = study_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tauspin_hpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "study.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def trial_file_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"tauspin_hpo.trial.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )
    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = True
    return logger


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "target_total_trials": arguments.target_total_trials,
        "max_steps": arguments.max_steps,
        "subset_eval_every_steps": (
            arguments.subset_eval_every_steps
        ),
        "full_eval_every_steps": arguments.full_eval_every_steps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(
            f"These arguments must be positive: {', '.join(invalid)}"
        )
    if arguments.pruning_warmup_steps < 0:
        raise ValueError("Pruning warmup must be non-negative")
    metadata_path = arguments.processed_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)


def immutable_study_config(
    arguments: argparse.Namespace,
    processed_dir: Path,
) -> dict[str, Any]:
    return {
        "purpose": (
            "Joint HPO mechanism smoke test on old sample with known "
            "jet-pT generation issue; not a physics result."
        ),
        "study_name": arguments.study_name,
        "direction": "maximize",
        "objective": "best full-validation ROC AUC during a trial",
        "test_split_policy": "never instantiate or load",
        "processed_dir": str(processed_dir.resolve()),
        "processed_metadata_sha256": sha256_file(
            processed_dir / "metadata.json"
        ),
        "search_space": SEARCH_SPACE,
        "model_profiles": MODEL_PROFILES,
        "sampler": {
            "name": "TPESampler",
            "seed": RANDOM_SEED,
            "n_startup_trials": 10,
        },
        "pruner": {
            "name": "MedianPruner",
            "n_startup_trials": 5,
            "n_warmup_steps": arguments.pruning_warmup_steps,
            "interval_steps": arguments.subset_eval_every_steps,
            "n_min_trials": 3,
            "reported_metric": (
                "mean of the three most recent fixed-subset AUC values"
            ),
        },
        "budget": {
            "max_optimizer_steps": arguments.max_steps,
            "subset_eval_every_steps": (
                arguments.subset_eval_every_steps
            ),
            "full_eval_every_steps": arguments.full_eval_every_steps,
        },
        "runtime": {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": PREFETCH_FACTOR,
            "precision": "tf32",
            "torch_compile": False,
            "execution": "eager",
            "balanced_sampling": True,
            "seed": RANDOM_SEED,
            "single_visible_gpu_required": True,
        },
        "validation_subset": {
            "split": "validation",
            "events_per_class": VALIDATION_PER_CLASS,
            "total_events": 2 * VALIDATION_PER_CLASS,
        },
        "initial_trials": INITIAL_TRIALS,
    }


def ensure_study_config(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != json_ready(expected):
            raise RuntimeError(
                "Existing study_config.json differs from requested setup"
            )
    else:
        write_json(path, expected)


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "model_profile": trial.suggest_categorical(
            "model_profile", SEARCH_SPACE["model_profile"]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate"]["low"],
            SEARCH_SPACE["learning_rate"]["high"],
            log=True,
        ),
        "dropout": trial.suggest_float(
            "dropout",
            SEARCH_SPACE["dropout"]["low"],
            SEARCH_SPACE["dropout"]["high"],
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            SEARCH_SPACE["weight_decay"]["low"],
            SEARCH_SPACE["weight_decay"]["high"],
            log=True,
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            SEARCH_SPACE["warmup_ratio"]["low"],
            SEARCH_SPACE["warmup_ratio"]["high"],
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler", SEARCH_SPACE["scheduler"]
        ),
    }


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for item in history:
        for key in item:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(json_ready(value))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in item.items()
            }
            for item in history
        )


def completed_or_pruned_count(study: optuna.Study) -> int:
    return sum(
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
        for trial in study.trials
    )


def enqueue_initial_trials_if_new(
    study: optuna.Study,
    logger: logging.Logger,
) -> bool:
    if study.trials:
        return False
    for parameters in INITIAL_TRIALS:
        study.enqueue_trial(
            parameters,
            user_attrs={"source": "required_initial_trial"},
            skip_if_exists=True,
        )
    logger.info("Enqueued %d required initial trials", len(INITIAL_TRIALS))
    return True


def training_checkpoint_state(
    *,
    global_step: int,
    events_seen: int,
    valid_tokens_seen: int,
    class_counts: Counter,
    epoch_equivalent: float,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "optimizer_step": global_step,
        "events_seen": events_seen,
        "valid_tokens_seen": valid_tokens_seen,
        "class_counts": dict(class_counts),
        "epoch_equivalent": epoch_equivalent,
        "learning_rate": learning_rate,
    }


def build_objective(
    *,
    arguments: argparse.Namespace,
    study_dir: Path,
    metadata: Mapping[str, Any],
    metadata_hash: str,
    manifest_hash: str,
    subset_loader: torch.utils.data.DataLoader,
    full_validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    runtime: Mapping[str, Any],
    study_logger: logging.Logger,
):
    def objective(trial: optuna.Trial) -> float:
        trial_started = time.perf_counter()
        trial_dir = study_dir / "trials" / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        logger = trial_file_logger(trial_dir / "run.log")
        parameters = suggest_parameters(trial)
        warmup_steps = round(
            float(parameters["warmup_ratio"]) * arguments.max_steps
        )
        parameters["warmup_steps"] = warmup_steps
        parameters["max_steps"] = arguments.max_steps

        set_random_seed(RANDOM_SEED)
        model, parameter_counts = create_model(
            metadata,
            parameters["model_profile"],
            float(parameters["dropout"]),
            device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(parameters["learning_rate"]),
            weight_decay=float(parameters["weight_decay"]),
        )
        loss_function = nn.BCEWithLogitsLoss()
        train_dataset, train_loader = create_streaming_loader(
            arguments.processed_dir,
            split="train",
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            shuffle=True,
            balanced=True,
            seed=RANDOM_SEED,
        )
        train_dataset.set_epoch(0)
        train_iterator = iter(train_loader)
        workers = [
            worker.pid
            for worker in getattr(train_iterator, "_workers", [])
        ]

        config = {
            "trial_number": trial.number,
            "parameters": parameters,
            "model_profile": MODEL_PROFILES[
                parameters["model_profile"]
            ],
            "parameter_counts": parameter_counts,
            "runtime": runtime,
            "processed_metadata_sha256": metadata_hash,
            "validation_manifest_sha256": manifest_hash,
            "train_worker_pids": workers,
            "test_split_loaded": False,
        }
        write_json(trial_dir / "config.json", config)
        logger.info("Starting trial %d: %s", trial.number, parameters)
        logger.info("Parameter counts: %s", parameter_counts)

        history: list[dict[str, Any]] = []
        subset_history: list[dict[str, Any]] = []
        full_history: list[dict[str, Any]] = []
        rolling_subset_auc: deque[float] = deque(maxlen=3)
        class_counts: Counter = Counter()
        events_seen = 0
        valid_tokens_seen = 0
        running_loss_sum = 0.0
        global_step = 0
        best_full_auc = -math.inf
        best_full_metrics: dict[str, Any] = {}
        best_step = None
        validation_parameter_checks = []
        state = "RUNNING"
        prune_reason = None
        worker_shutdown: dict[str, Any] = {}

        try:
            model.train()
            while global_step < arguments.max_steps:
                cpu_batch = next(train_iterator)
                batch = move_batch(cpu_batch, device)
                global_step += 1
                current_learning_rate = learning_rate_for_step(
                    base_learning_rate=float(
                        parameters["learning_rate"]
                    ),
                    step=global_step,
                    max_steps=arguments.max_steps,
                    warmup_steps=warmup_steps,
                    scheduler=str(parameters["scheduler"]),
                )
                set_optimizer_learning_rate(
                    optimizer, current_learning_rate
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                require_finite(logits, "train logits")
                loss = loss_function(logits, batch["labels"])
                require_finite(loss, "train loss")
                loss.backward()
                require_finite_gradients(model)
                optimizer.step()

                batch_events = int(batch["labels"].shape[0])
                batch_valid_tokens = (
                    int((~batch["padding_mask"]).sum().item())
                    + batch_events
                )
                events_seen += batch_events
                valid_tokens_seen += batch_valid_tokens
                labels = batch["labels"].detach().to(torch.int64).cpu()
                class_counts.update(labels.tolist())
                batch_loss = float(loss.detach().cpu())
                running_loss_sum += batch_loss * batch_events
                elapsed = time.perf_counter() - trial_started
                epoch_equivalent = events_seen / len(train_dataset)
                history.append(
                    {
                        "optimizer_step": global_step,
                        "epoch_equivalent": epoch_equivalent,
                        "events_seen": events_seen,
                        "h_seen": class_counts[1],
                        "z_seen": class_counts[0],
                        "valid_tokens_seen": valid_tokens_seen,
                        "batch_train_loss": batch_loss,
                        "mean_train_loss": (
                            running_loss_sum / events_seen
                        ),
                        "learning_rate": current_learning_rate,
                        "elapsed_seconds": elapsed,
                        "events_per_second": events_seen / elapsed,
                        "tokens_per_second": (
                            valid_tokens_seen / elapsed
                        ),
                    }
                )

                subset_due = (
                    global_step % arguments.subset_eval_every_steps == 0
                    or global_step == arguments.max_steps
                )
                full_due = (
                    global_step % arguments.full_eval_every_steps == 0
                    or global_step == arguments.max_steps
                )

                if subset_due:
                    subset_metrics = evaluate_model(
                        model,
                        subset_loader,
                        loss_function,
                        device,
                        "validation subset",
                        verify_parameters_unchanged=(
                            len(subset_history) == 0
                        ),
                    )
                    subset_record = {
                        "optimizer_step": global_step,
                        "epoch_equivalent": epoch_equivalent,
                        **strip_evaluation_arrays(subset_metrics),
                    }
                    subset_history.append(subset_record)
                    rolling_subset_auc.append(
                        float(subset_metrics["auc"])
                    )
                    rolling_mean = float(
                        np.mean(rolling_subset_auc)
                    )
                    subset_record["rolling_auc_3"] = rolling_mean
                    if subset_metrics["parameters_unchanged"] is not None:
                        validation_parameter_checks.append(
                            {
                                "kind": "subset",
                                "step": global_step,
                                "unchanged": subset_metrics[
                                    "parameters_unchanged"
                                ],
                            }
                        )
                    trial.report(rolling_mean, step=global_step)
                    logger.info(
                        "step=%d subset_auc=%.6f subset_loss=%.6f "
                        "rolling_auc_3=%.6f",
                        global_step,
                        subset_metrics["auc"],
                        subset_metrics["loss"],
                        rolling_mean,
                    )

                if full_due:
                    full_metrics = evaluate_model(
                        model,
                        full_validation_loader,
                        loss_function,
                        device,
                        "full validation",
                        verify_parameters_unchanged=(
                            len(full_history) == 0
                        ),
                    )
                    full_record = {
                        "optimizer_step": global_step,
                        "epoch_equivalent": epoch_equivalent,
                        **strip_evaluation_arrays(full_metrics),
                    }
                    full_history.append(full_record)
                    if full_metrics["parameters_unchanged"] is not None:
                        validation_parameter_checks.append(
                            {
                                "kind": "full",
                                "step": global_step,
                                "unchanged": full_metrics[
                                    "parameters_unchanged"
                                ],
                            }
                        )
                    logger.info(
                        "step=%d full_auc=%.6f full_loss=%.6f",
                        global_step,
                        full_metrics["auc"],
                        full_metrics["loss"],
                    )
                    if float(full_metrics["auc"]) > best_full_auc:
                        best_full_auc = float(full_metrics["auc"])
                        best_step = global_step
                        best_full_metrics = strip_evaluation_arrays(
                            full_metrics
                        )
                        checkpoint = make_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            metadata=metadata,
                            trial_number=trial.number,
                            trial_parameters=parameters,
                            model_profile=MODEL_PROFILES[
                                parameters["model_profile"]
                            ],
                            parameter_counts=parameter_counts,
                            training_state=training_checkpoint_state(
                                global_step=global_step,
                                events_seen=events_seen,
                                valid_tokens_seen=valid_tokens_seen,
                                class_counts=class_counts,
                                epoch_equivalent=epoch_equivalent,
                                learning_rate=current_learning_rate,
                            ),
                            best_metrics=best_full_metrics,
                            runtime=runtime,
                            manifest_hash=manifest_hash,
                            metadata_hash=metadata_hash,
                            batch_size=BATCH_SIZE,
                            seed=RANDOM_SEED,
                        )
                        torch.save(
                            checkpoint, trial_dir / "best_model.pt"
                        )

                if subset_due:
                    prune_decision = trial.should_prune()
                    logger.info(
                        "step=%d pruner_checked=true decision=%s",
                        global_step,
                        prune_decision,
                    )
                    if prune_decision:
                        prune_reason = (
                            "MedianPruner requested pruning at optimizer "
                            f"step {global_step}; rolling subset AUC="
                            f"{rolling_mean:.6f}"
                        )
                        state = "PRUNED"
                        raise optuna.TrialPruned(prune_reason)

            state = "COMPLETE"
            final_full = full_history[-1]
            trial.set_user_attr("best_full_validation_auc", best_full_auc)
            trial.set_user_attr("best_step", best_step)
            trial.set_user_attr(
                "final_full_validation_auc", final_full["auc"]
            )
            trial.set_user_attr(
                "final_full_validation_loss", final_full["loss"]
            )
            minimum_loss_record = min(
                full_history, key=lambda item: float(item["loss"])
            )
            trial.set_user_attr(
                "best_full_validation_loss",
                float(minimum_loss_record["loss"]),
            )
            trial.set_user_attr(
                "best_full_validation_loss_step",
                int(minimum_loss_record["optimizer_step"]),
            )
            trial.set_user_attr(
                "parameter_count", parameter_counts["total"]
            )
            trial.set_user_attr(
                "elapsed_seconds", time.perf_counter() - trial_started
            )
            last_checkpoint = make_checkpoint(
                model=model,
                optimizer=optimizer,
                metadata=metadata,
                trial_number=trial.number,
                trial_parameters=parameters,
                model_profile=MODEL_PROFILES[
                    parameters["model_profile"]
                ],
                parameter_counts=parameter_counts,
                training_state=training_checkpoint_state(
                    global_step=global_step,
                    events_seen=events_seen,
                    valid_tokens_seen=valid_tokens_seen,
                    class_counts=class_counts,
                    epoch_equivalent=events_seen / len(train_dataset),
                    learning_rate=optimizer.param_groups[0]["lr"],
                ),
                best_metrics=best_full_metrics,
                runtime=runtime,
                manifest_hash=manifest_hash,
                metadata_hash=metadata_hash,
                batch_size=BATCH_SIZE,
                seed=RANDOM_SEED,
            )
            torch.save(last_checkpoint, trial_dir / "last_model.pt")
            return best_full_auc
        except optuna.TrialPruned:
            trial.set_user_attr("prune_reason", prune_reason)
            trial.set_user_attr("pruned_at_step", global_step)
            trial.set_user_attr(
                "elapsed_seconds", time.perf_counter() - trial_started
            )
            raise
        finally:
            worker_shutdown = shutdown_loader_workers(
                train_loader, train_iterator
            )
            elapsed = time.perf_counter() - trial_started
            metrics = {
                "trial_number": trial.number,
                "state": state,
                "objective_best_full_validation_auc": (
                    None
                    if best_full_auc == -math.inf
                    else best_full_auc
                ),
                "best_step": best_step,
                "best_full_validation_metrics": best_full_metrics,
                "minimum_full_validation_loss": (
                    min(
                        (
                            {
                                "loss": float(item["loss"]),
                                "optimizer_step": int(
                                    item["optimizer_step"]
                                ),
                            }
                            for item in full_history
                        ),
                        key=lambda item: item["loss"],
                        default=None,
                    )
                ),
                "final_full_validation_metrics": (
                    full_history[-1] if full_history else None
                ),
                "optimizer_steps": global_step,
                "events_seen": events_seen,
                "valid_tokens_seen": valid_tokens_seen,
                "class_counts": dict(class_counts),
                "elapsed_seconds": elapsed,
                "prune_reason": prune_reason,
                "validation_parameter_checks": (
                    validation_parameter_checks
                ),
                "worker_shutdown": worker_shutdown,
                "test_split_loaded": False,
                "finite_training": True,
            }
            write_json(
                trial_dir / "history.json",
                {
                    "train_steps": history,
                    "subset_validation": subset_history,
                    "full_validation": full_history,
                },
            )
            save_history_csv(
                trial_dir / "history.csv", history
            )
            write_json(trial_dir / "metrics.json", metrics)
            logger.info(
                "Finished trial %d state=%s steps=%d elapsed=%.2fs",
                trial.number,
                state,
                global_step,
                elapsed,
            )
            study_logger.info(
                "trial=%d state=%s best_auc=%s steps=%d elapsed=%.2fs",
                trial.number,
                state,
                (
                    "none"
                    if best_full_auc == -math.inf
                    else f"{best_full_auc:.6f}"
                ),
                global_step,
                elapsed,
            )

    return objective


def trial_state_counts(study: optuna.Study) -> dict[str, int]:
    counts = Counter(trial.state.name for trial in study.trials)
    return dict(sorted(counts.items()))


def save_optimization_plot(
    study: optuna.Study,
    path: Path,
) -> None:
    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained"
    )
    if completed:
        numbers = [trial.number for trial in completed]
        values = [float(trial.value) for trial in completed]
        running_best = np.maximum.accumulate(values)
        axes[0].scatter(numbers, values, label="Completed trial")
        axes[0].plot(
            numbers,
            running_best,
            color="tab:red",
            label="Running best",
        )
        profiles = sorted(MODEL_PROFILES)
        data = [
            [
                float(trial.value)
                for trial in completed
                if trial.params.get("model_profile") == profile
            ]
            for profile in profiles
        ]
        nonempty = [
            (profile, values)
            for profile, values in zip(profiles, data)
            if values
        ]
        axes[1].boxplot(
            [values for _, values in nonempty],
            tick_labels=[profile for profile, _ in nonempty],
        )
    else:
        axes[0].text(0.5, 0.5, "No completed trials", ha="center")
        axes[1].text(0.5, 0.5, "No completed trials", ha="center")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Best full-validation AUC")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Model profile")
    axes[1].set_ylabel("Best full-validation AUC")
    axes[1].grid(alpha=0.3)
    figure.suptitle(
        "Old-sample joint HPO smoke test\n"
        "Known generation issue; mechanism validation only"
    )
    figure.savefig(path)
    plt.close(figure)


def reload_best_checkpoint(
    *,
    study: optuna.Study,
    study_dir: Path,
    metadata: Mapping[str, Any],
    full_validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    best = study.best_trial
    checkpoint_path = (
        study_dir
        / "trials"
        / f"trial_{best.number:03d}"
        / "best_model.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    hyperparameters = checkpoint["hyperparameters"]
    model, parameter_counts = create_model(
        metadata,
        hyperparameters["model_profile"],
        float(hyperparameters["dropout"]),
        device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(
        model,
        full_validation_loader,
        nn.BCEWithLogitsLoss(),
        device,
        "reloaded best full validation",
        verify_parameters_unchanged=True,
    )
    expected_auc = float(
        checkpoint["best_full_validation_metrics"]["auc"]
    )
    difference = float(metrics["auc"]) - expected_auc
    if abs(difference) > 1.0e-10:
        raise RuntimeError(
            "Reloaded best checkpoint did not reproduce stored AUC: "
            f"difference={difference}"
        )
    return {
        "trial_number": best.number,
        "checkpoint_path": str(checkpoint_path),
        "expected_auc": expected_auc,
        "reloaded_metrics": strip_evaluation_arrays(metrics),
        "auc_difference": difference,
        "parameter_counts": parameter_counts,
        "passed": True,
    }


def save_study_artifacts(
    *,
    study: optuna.Study,
    study_dir: Path,
    reload_audit: Mapping[str, Any] | None,
    command_started: float,
) -> None:
    trial_rows = []
    for trial in study.trials:
        trial_rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": (
                    trial.datetime_start.isoformat()
                    if trial.datetime_start is not None
                    else None
                ),
                "datetime_complete": (
                    trial.datetime_complete.isoformat()
                    if trial.datetime_complete is not None
                    else None
                ),
                "duration_seconds": (
                    trial.duration.total_seconds()
                    if trial.duration is not None
                    else None
                ),
                "parameters": json.dumps(
                    json_ready(trial.params), sort_keys=True
                ),
                "user_attributes": json.dumps(
                    json_ready(trial.user_attrs), sort_keys=True
                ),
                "intermediate_values": json.dumps(
                    json_ready(trial.intermediate_values),
                    sort_keys=True,
                ),
            }
        )
    with (study_dir / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(trial_rows[0]) if trial_rows else ["number"],
        )
        writer.writeheader()
        writer.writerows(trial_rows)
    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    best_trial = None
    if completed:
        best = study.best_trial
        best_trial = {
            "number": best.number,
            "value": best.value,
            "parameters": best.params,
            "user_attributes": best.user_attrs,
        }
        write_json(study_dir / "best_trial.json", best_trial)
    importance: dict[str, Any]
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as error:
        importance = {"error": repr(error)}
    write_json(study_dir / "parameter_importance.json", importance)
    save_optimization_plot(
        study, study_dir / "optimization_history.pdf"
    )
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "trial_state_counts": trial_state_counts(study),
        "completed_or_pruned_trials": completed_or_pruned_count(study),
        "queued_trials": sum(
            trial.state == TrialState.WAITING for trial in study.trials
        ),
        "best_trial": best_trial,
        "reload_audit": reload_audit,
        "test_split_loaded": False,
        "command_elapsed_seconds": time.perf_counter() - command_started,
    }
    write_json(study_dir / "study_summary.json", summary)


def main() -> None:
    arguments = parse_args()
    validate_arguments(arguments)
    command_started = time.perf_counter()
    study_dir = arguments.output_root / arguments.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(exist_ok=True)
    logger = setup_logging(study_dir)
    logger.info("Starting/resuming study %s", arguments.study_name)

    set_random_seed(RANDOM_SEED)
    device = choose_device(arguments.device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("HPO requires exactly one visible GPU")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if arguments.physical_gpu_index is not None and (
        visible_devices != str(arguments.physical_gpu_index)
    ):
        raise RuntimeError(
            "--physical-gpu-index must match CUDA_VISIBLE_DEVICES; "
            f"got {arguments.physical_gpu_index} and {visible_devices!r}"
        )
    precision_info = configure_tf32()
    runtime = {
        "device": str(device),
        "physical_gpu_index": arguments.physical_gpu_index,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": PREFETCH_FACTOR,
        "precision": precision_info,
        "torch_compile": False,
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
    }

    metadata = json.loads(
        (arguments.processed_dir / "metadata.json").read_text()
    )
    metadata_hash = sha256_file(
        arguments.processed_dir / "metadata.json"
    )
    study_config = immutable_study_config(
        arguments, arguments.processed_dir
    )
    ensure_study_config(
        study_dir / "study_config.json", study_config
    )
    environment_path = study_dir / "environment.json"
    if not environment_path.exists():
        write_json(
            environment_path,
            environment_information(
                PROJECT_DIR.parent, arguments.physical_gpu_index
            ),
        )

    manifest, subset_events, manifest_created = (
        load_or_create_validation_manifest(
            study_dir / "validation_subset_manifest.json",
            arguments.processed_dir,
            metadata,
            per_class=VALIDATION_PER_CLASS,
        )
    )
    manifest_hash = sha256_bytes(
        json.dumps(manifest, sort_keys=True).encode()
    )
    logger.info(
        "Validation manifest %s; fingerprint=%s",
        "created" if manifest_created else "reused",
        manifest["event_tensor_fingerprint_sha256"],
    )

    subset_loader = create_list_loader(
        subset_events,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )
    _, full_validation_loader = create_streaming_loader(
        arguments.processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=RANDOM_SEED,
    )

    storage_path = study_dir / "optuna.db"
    storage_url = f"sqlite:///{storage_path.resolve()}"
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED, n_startup_trials=10
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=arguments.pruning_warmup_steps,
        interval_steps=arguments.subset_eval_every_steps,
        n_min_trials=3,
    )
    study = optuna.create_study(
        study_name=arguments.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    enqueue_initial_trials_if_new(study, logger)
    finished_before = completed_or_pruned_count(study)
    remaining = max(
        0, arguments.target_total_trials - finished_before
    )
    logger.info(
        "Finished before command=%d target=%d remaining=%d states=%s",
        finished_before,
        arguments.target_total_trials,
        remaining,
        trial_state_counts(study),
    )

    objective = build_objective(
        arguments=arguments,
        study_dir=study_dir,
        metadata=metadata,
        metadata_hash=metadata_hash,
        manifest_hash=manifest_hash,
        subset_loader=subset_loader,
        full_validation_loader=full_validation_loader,
        device=device,
        runtime=runtime,
        study_logger=logger,
    )
    try:
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                gc_after_trial=True,
                show_progress_bar=False,
            )
        reload_audit = (
            reload_best_checkpoint(
                study=study,
                study_dir=study_dir,
                metadata=metadata,
                full_validation_loader=full_validation_loader,
                device=device,
            )
            if any(
                trial.state == TrialState.COMPLETE
                for trial in study.trials
            )
            else None
        )
        save_study_artifacts(
            study=study,
            study_dir=study_dir,
            reload_audit=reload_audit,
            command_started=command_started,
        )
    finally:
        subset_shutdown = shutdown_loader_workers(subset_loader)
        full_shutdown = shutdown_loader_workers(
            full_validation_loader
        )
        write_json(
            study_dir / "loader_shutdown.json",
            {
                "subset": subset_shutdown,
                "full_validation": full_shutdown,
            },
        )

    logger.info(
        "Study command complete: states=%s best=%s elapsed=%.2fs",
        trial_state_counts(study),
        (
            f"trial {study.best_trial.number} AUC={study.best_value:.6f}"
            if any(
                trial.state == TrialState.COMPLETE
                for trial in study.trials
            )
            else "none"
        ),
        time.perf_counter() - command_started,
    )


if __name__ == "__main__":
    main()
