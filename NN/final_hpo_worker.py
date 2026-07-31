from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset import event_partition_count
from hpo_utils import (
    MODEL_PROFILES,
    configure_tf32,
    create_model,
    create_streaming_loader,
    evaluate_model,
    learning_rate_for_step,
    portable_model_state_dict,
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


BATCH_SIZE = 512
NUM_WORKERS = 12
PREFETCH_FACTOR = 2
WORKER_PARTITION = "event"
SEED = 42
WEIGHT_DECAY = 1.0e-4
EXPECTED_FEATURE_SET = "absolute-plus-parent-relative-v3"
RUNTIME_PROFILE_ID = "high-throughput-v1"
FINITE_CHECK_INTERVAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one hash-bound validation-only HPO trial on one GPU."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--parameters-json", type=Path, required=True)
    parser.add_argument(
        "--event-selection-manifest", type=Path, required=True
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-snapshot-sha256", required=True)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--objective-start-epoch", type=int, default=8)
    parser.add_argument("--early-stop-start-epoch", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=5.0e-4)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--overfit-auc-drop", type=float, default=3.0e-3)
    parser.add_argument("--overfit-patience", type=int, default=3)
    return parser.parse_args()


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_steps_per_epoch(metadata: dict[str, Any]) -> int:
    """Mirror event-stride partitioning and worker-local balancing."""
    total_batches = 0
    for worker in range(NUM_WORKERS):
        counts = {}
        for sample in ("H", "Z"):
            event_offset = 0
            counts[sample] = 0
            for record in metadata["shards"]["train"][sample]:
                n_events = int(record["events"])
                counts[sample] += event_partition_count(
                    n_events,
                    worker,
                    NUM_WORKERS,
                    event_offset,
                )
                event_offset += n_events
        worker_events = 2 * max(counts.values())
        total_batches += math.ceil(worker_events / BATCH_SIZE)
    return total_batches


def checkpoint_document(
    *,
    model: nn.Module,
    metadata: dict[str, Any],
    parameters: dict[str, Any],
    parameter_counts: dict[str, int],
    trial_number: int,
    epoch: int,
    optimizer_step: int,
    validation_record: dict[str, Any],
    rolling_auc: float | None,
    rolling_loss: float | None,
    metadata_hash: str,
    selection_hash: str,
    snapshot_hash: str,
    precision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "purpose": (
            "Hash-bound high-throughput validation-only HPO; "
            "test split was not loaded."
        ),
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "trial_number": trial_number,
        "model_state_dict": portable_model_state_dict(model),
        "feature_set": metadata["feature_set"],
        "feature_dimensions": metadata["feature_dimensions"],
        "feature_names": metadata["feature_names"],
        "tau_decay_num_embeddings": metadata[
            "tau_decay_num_embeddings"
        ],
        "tau_decay_mode_to_id": metadata["tau_decay_mode_to_id"],
        "model_profile_name": parameters["model_profile"],
        "model_profile": MODEL_PROFILES[parameters["model_profile"]],
        "parameter_counts": parameter_counts,
        "hyperparameters": parameters,
        "training_state": {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
        },
        "validation_metrics": validation_record,
        "rolling_validation_auc_3": rolling_auc,
        "rolling_validation_loss_3": rolling_loss,
        "data": {
            "processed_metadata_sha256": metadata_hash,
            "event_selection_manifest_sha256": selection_hash,
            "snapshot_manifest_sha256": snapshot_hash,
            "event_selection": metadata["event_selection"],
            "batch_size": BATCH_SIZE,
            "balanced_sampling": True,
            "worker_partition": WORKER_PARTITION,
            "seed": SEED,
        },
        "runtime": {
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "worker_partition": WORKER_PARTITION,
            "input_projection": "dense_masked",
            "precision": precision,
            "fused_adamw": True,
            "torch_compile": True,
            "torch_compile_dynamic": True,
            "finite_check_interval_steps": FINITE_CHECK_INTERVAL,
        },
        "test_split_loaded": False,
    }


def main() -> None:
    args = parse_args()
    parameter_document = json.loads(args.parameters_json.read_text())
    parameters = parameter_document.get("parameters", parameter_document)
    metadata_path = args.processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_hash = sha256_file(metadata_path)
    selection_hash = sha256_file(args.event_selection_manifest)
    snapshot_hash = sha256_file(args.snapshot_manifest)
    actual_hashes = {
        "metadata": metadata_hash,
        "selection": selection_hash,
        "snapshot": snapshot_hash,
    }
    expected_hashes = {
        "metadata": args.expected_metadata_sha256,
        "selection": args.expected_selection_sha256,
        "snapshot": args.expected_snapshot_sha256,
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"Data binding hash mismatch: actual={actual_hashes}, "
            f"expected={expected_hashes}"
        )
    args.trial_dir.mkdir(parents=True, exist_ok=False)
    if metadata["feature_set"] != EXPECTED_FEATURE_SET:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_SET}, got {metadata['feature_set']}"
        )
    if metadata.get("event_selection") is None:
        raise ValueError("The final HPO requires a pT-matching manifest")
    if parameters["model_profile"] not in MODEL_PROFILES:
        raise ValueError(parameters["model_profile"])
    if parameters["schedule_profile"] not in (
        "constant",
        "cosine_warmup5",
    ):
        raise ValueError(parameters["schedule_profile"])

    set_random_seed(SEED)
    device = choose_device("cuda")
    precision = configure_tf32()
    model, parameter_counts = create_model(
        metadata,
        parameters["model_profile"],
        float(parameters["dropout"]),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    model = torch.compile(model, dynamic=True)
    loss_function = nn.BCEWithLogitsLoss()
    _, validation_loader = create_streaming_loader(
        args.processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
        worker_partition=WORKER_PARTITION,
    )
    steps_per_epoch = exact_steps_per_epoch(metadata)
    maximum_steps = steps_per_epoch * args.max_epochs
    warmup_steps = (
        round(0.05 * maximum_steps)
        if parameters["schedule_profile"] == "cosine_warmup5"
        else 0
    )
    scheduler_name = (
        "cosine"
        if parameters["schedule_profile"] == "cosine_warmup5"
        else "constant"
    )

    config = {
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "trial_number": args.trial_number,
        "parameters": parameters,
        "model_profile": MODEL_PROFILES[parameters["model_profile"]],
        "parameter_counts": parameter_counts,
        "processed_dir": str(args.processed_dir.resolve()),
        "processed_metadata_sha256": metadata_hash,
        "event_selection_manifest_path": str(
            args.event_selection_manifest.resolve()
        ),
        "event_selection_manifest_sha256": selection_hash,
        "snapshot_manifest_path": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": snapshot_hash,
        "feature_set": metadata["feature_set"],
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": args.max_epochs,
        "maximum_optimizer_steps_for_schedule": maximum_steps,
        "warmup_steps": warmup_steps,
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "worker_partition": WORKER_PARTITION,
            "input_projection": "dense_masked",
            "precision": precision,
            "fused_adamw": True,
            "torch_compile": True,
            "torch_compile_dynamic": True,
            "finite_check_interval_steps": FINITE_CHECK_INTERVAL,
            "seed": SEED,
        },
        "early_stopping": {
            "start_after_epoch": args.early_stop_start_epoch,
            "min_delta": args.early_stop_min_delta,
            "patience": args.early_stop_patience,
            "overfit_auc_drop": args.overfit_auc_drop,
            "overfit_patience": args.overfit_patience,
        },
        "objective": {
            "metric": "maximum three-epoch mean validation AUC",
            "central_epoch_at_least": args.objective_start_epoch,
        },
        "test_split_loaded": False,
    }
    write_json(args.trial_dir / "config.json", config)

    history: list[dict[str, Any]] = []
    recent_auc: deque[float] = deque(maxlen=3)
    recent_loss: deque[float] = deque(maxlen=3)
    recent_checkpoint_paths: deque[Path] = deque(maxlen=3)
    best_rolling_auc = -math.inf
    best_rolling_loss: float | None = None
    best_window: list[int] | None = None
    best_center_epoch: int | None = None
    substantial_best = -math.inf
    no_improvement_epochs = 0
    overfit_epochs = 0
    stopped_early = False
    stop_reason: str | None = None
    global_step = 0
    events_seen = 0
    class_counts: Counter[int] = Counter()
    epoch_worker_shutdown: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for epoch_index in range(args.max_epochs):
            epoch_number = epoch_index + 1
            train_dataset, train_loader = create_streaming_loader(
                args.processed_dir,
                split="train",
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                shuffle=True,
                balanced=True,
                seed=SEED,
                worker_partition=WORKER_PARTITION,
            )
            train_dataset.set_epoch(epoch_index)
            train_iterator = iter(train_loader)
            epoch_loss_terms: list[torch.Tensor] = []
            epoch_events = 0
            epoch_steps = 0
            last_logits: torch.Tensor | None = None
            last_loss: torch.Tensor | None = None
            try:
                while True:
                    try:
                        cpu_batch = next(train_iterator)
                    except StopIteration:
                        break
                    class_counts.update(
                        cpu_batch["labels"].to(torch.int64).tolist()
                    )
                    batch = move_batch(cpu_batch, device)
                    global_step += 1
                    epoch_steps += 1
                    current_lr = learning_rate_for_step(
                        base_learning_rate=float(
                            parameters["learning_rate"]
                        ),
                        step=global_step,
                        max_steps=maximum_steps,
                        warmup_steps=warmup_steps,
                        scheduler=scheduler_name,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch)
                    loss = loss_function(logits, batch["labels"])
                    loss.backward()
                    if (
                        epoch_steps == 1
                        or epoch_steps % FINITE_CHECK_INTERVAL == 0
                    ):
                        require_finite(logits, "train logits")
                        require_finite(loss, "train loss")
                        require_finite_gradients(model)
                    optimizer.step()
                    batch_events = int(batch["labels"].shape[0])
                    epoch_loss_terms.append(
                        loss.detach() * batch_events
                    )
                    epoch_events += batch_events
                    events_seen += batch_events
                    last_logits = logits
                    last_loss = loss
            finally:
                shutdown = shutdown_loader_workers(
                    train_loader, train_iterator
                )
                shutdown["epoch"] = epoch_number
                shutdown["optimizer_steps"] = epoch_steps
                epoch_worker_shutdown.append(shutdown)

            if epoch_steps == 0 or epoch_events == 0:
                raise RuntimeError("Training epoch produced no batches")
            assert last_logits is not None and last_loss is not None
            require_finite(last_logits, "final train logits")
            require_finite(last_loss, "final train loss")
            require_finite_gradients(model)
            epoch_loss_sum = float(
                torch.stack(epoch_loss_terms).sum().cpu()
            )
            metrics = evaluate_model(
                model,
                validation_loader,
                loss_function,
                device,
                f"trial {args.trial_number} validation epoch {epoch_number}",
                verify_parameters_unchanged=True,
            )
            recent_auc.append(float(metrics["auc"]))
            recent_loss.append(float(metrics["loss"]))
            record = {
                "epoch": epoch_number,
                "optimizer_step": global_step,
                "epoch_optimizer_steps": epoch_steps,
                "events_seen": events_seen,
                "epoch_train_loss": epoch_loss_sum / epoch_events,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_auc": float(metrics["auc"]),
                "validation_loss": float(metrics["loss"]),
                "validation_events": int(metrics["event_count"]),
                "validation_parameters_unchanged": metrics[
                    "parameters_unchanged"
                ],
                "rolling_auc_3": (
                    float(np.mean(recent_auc))
                    if len(recent_auc) == 3
                    else None
                ),
                "rolling_loss_3": (
                    float(np.mean(recent_loss))
                    if len(recent_loss) == 3
                    else None
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)

            recent_path = (
                args.trial_dir / f"recent_epoch_{epoch_number:03d}.pt"
            )
            torch.save(
                checkpoint_document(
                    model=model,
                    metadata=metadata,
                    parameters=parameters,
                    parameter_counts=parameter_counts,
                    trial_number=args.trial_number,
                    epoch=epoch_number,
                    optimizer_step=global_step,
                    validation_record={
                        **strip_evaluation_arrays(metrics),
                        "epoch": epoch_number,
                    },
                    rolling_auc=record["rolling_auc_3"],
                    rolling_loss=record["rolling_loss_3"],
                    metadata_hash=metadata_hash,
                    selection_hash=selection_hash,
                    snapshot_hash=snapshot_hash,
                    precision=precision,
                ),
                recent_path,
            )
            recent_checkpoint_paths.append(recent_path)
            keep = set(recent_checkpoint_paths)
            for path in args.trial_dir.glob("recent_epoch_*.pt"):
                if path not in keep:
                    path.unlink()

            central_epoch = epoch_number - 1
            rolling_auc = record["rolling_auc_3"]
            rolling_loss = record["rolling_loss_3"]
            if (
                rolling_auc is not None
                and central_epoch >= args.objective_start_epoch
                and rolling_auc > best_rolling_auc
            ):
                best_rolling_auc = float(rolling_auc)
                best_rolling_loss = float(rolling_loss)
                best_window = [
                    epoch_number - 2,
                    epoch_number - 1,
                    epoch_number,
                ]
                best_center_epoch = central_epoch
                shutil.copy2(
                    list(recent_checkpoint_paths)[1],
                    args.trial_dir / "best_rolling_auc_model.pt",
                )

            if (
                rolling_auc is not None
                and epoch_number > args.early_stop_start_epoch
            ):
                if rolling_auc >= (
                    substantial_best + args.early_stop_min_delta
                ):
                    substantial_best = float(rolling_auc)
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                worsened_loss = (
                    best_rolling_loss is not None
                    and rolling_loss is not None
                    and rolling_loss > best_rolling_loss
                )
                if (
                    rolling_auc
                    <= best_rolling_auc - args.overfit_auc_drop
                    and worsened_loss
                ):
                    overfit_epochs += 1
                else:
                    overfit_epochs = 0

                if overfit_epochs >= args.overfit_patience:
                    stopped_early = True
                    stop_reason = (
                        "clear_overfit: rolling AUC below best by at least "
                        f"{args.overfit_auc_drop} and rolling loss worsened "
                        f"for {overfit_epochs} epochs"
                    )
                elif no_improvement_epochs >= args.early_stop_patience:
                    stopped_early = True
                    stop_reason = (
                        "plateau: rolling AUC failed to improve by "
                        f"{args.early_stop_min_delta} for "
                        f"{no_improvement_epochs} epochs"
                    )
            save_csv(args.trial_dir / "history.csv", history)
            write_json(args.trial_dir / "history.json", history)
            write_json(
                args.trial_dir / "progress.json",
                {
                    "epoch": epoch_number,
                    "optimizer_step": global_step,
                    "best_rolling_auc": (
                        None
                        if best_rolling_auc == -math.inf
                        else best_rolling_auc
                    ),
                    "best_center_epoch": best_center_epoch,
                    "stopped_early": stopped_early,
                    "stop_reason": stop_reason,
                    "test_split_loaded": False,
                },
            )
            print(
                f"trial={args.trial_number} epoch={epoch_number} "
                f"steps={global_step} train_loss={record['epoch_train_loss']:.6f} "
                f"val_auc={record['validation_auc']:.6f} "
                f"val_loss={record['validation_loss']:.6f} "
                f"rolling_auc_3={rolling_auc}",
                flush=True,
            )
            if stopped_early:
                break
    finally:
        validation_shutdown = shutdown_loader_workers(validation_loader)

    if best_center_epoch is None:
        raise RuntimeError("No eligible three-epoch objective window")

    best_checkpoint_path = args.trial_dir / "best_rolling_auc_model.pt"
    saved = torch.load(
        best_checkpoint_path, map_location=device, weights_only=True
    )
    reloaded_model, _ = create_model(
        metadata,
        parameters["model_profile"],
        float(parameters["dropout"]),
        device,
    )
    reloaded_model.load_state_dict(saved["model_state_dict"])
    reloaded_evaluation_model = torch.compile(
        reloaded_model,
        dynamic=True,
    )
    _, reload_loader = create_streaming_loader(
        args.processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
        worker_partition=WORKER_PARTITION,
    )
    reload_metrics = evaluate_model(
        reloaded_evaluation_model,
        reload_loader,
        loss_function,
        device,
        "reloaded best rolling center checkpoint",
        verify_parameters_unchanged=True,
    )
    reload_shutdown = shutdown_loader_workers(reload_loader)
    expected_center_auc = float(
        next(
            record["validation_auc"]
            for record in history
            if record["epoch"] == best_center_epoch
        )
    )
    reload_difference = float(reload_metrics["auc"]) - expected_center_auc
    if abs(reload_difference) > 1.0e-12:
        raise RuntimeError(
            "Reloaded checkpoint AUC differs from recorded center epoch: "
            f"{reload_difference}"
        )

    minimum_loss = min(history, key=lambda item: item["validation_loss"])
    single_best = max(history, key=lambda item: item["validation_auc"])
    result = {
        "state": "COMPLETE",
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "trial_number": args.trial_number,
        "objective": best_rolling_auc,
        "best_rolling_auc_3": best_rolling_auc,
        "best_rolling_loss_3": best_rolling_loss,
        "best_window_epochs": best_window,
        "best_center_epoch": best_center_epoch,
        "best_single_validation_auc": single_best["validation_auc"],
        "best_single_validation_auc_epoch": single_best["epoch"],
        "minimum_validation_loss": minimum_loss["validation_loss"],
        "minimum_validation_loss_epoch": minimum_loss["epoch"],
        "epochs_completed": len(history),
        "optimizer_steps": global_step,
        "events_seen": events_seen,
        "class_counts": dict(class_counts),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": parameters,
        "parameter_counts": parameter_counts,
        "steps_per_epoch_expected": steps_per_epoch,
        "epoch_step_counts": [
            int(record["epoch_optimizer_steps"]) for record in history
        ],
        "checkpoint_sha256": sha256_file(best_checkpoint_path),
        "data_binding": {
            "processed_metadata_sha256": metadata_hash,
            "event_selection_manifest_sha256": selection_hash,
            "snapshot_manifest_sha256": snapshot_hash,
        },
        "reloaded_center_auc": float(reload_metrics["auc"]),
        "reloaded_center_auc_difference": reload_difference,
        "validation_worker_shutdown": validation_shutdown,
        "reload_worker_shutdown": reload_shutdown,
        "finite_training": True,
        "test_split_loaded": False,
    }
    write_json(args.trial_dir / "result.json", result)
    for path in args.trial_dir.glob("recent_epoch_*.pt"):
        path.unlink()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
