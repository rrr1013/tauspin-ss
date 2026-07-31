from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import TauSpinDataset, collate_events
from model import TauSpinTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TauSpin Transformer."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--runtime-profile",
        choices=("compatibility", "high-throughput"),
        default="high-throughput",
        help=(
            "high-throughput uses the verified GPU settings (default); "
            "compatibility explicitly preserves the original runtime"
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument(
        "--worker-partition",
        choices=("shard", "event"),
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
    )
    arguments = parser.parse_args()
    defaults = {
        "compatibility": {
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "prefetch_factor": 2,
            "worker_partition": "shard",
            "tf32": False,
            "fused_adamw": False,
            "compile_model": False,
        },
        "high-throughput": {
            "batch_size": 512,
            "num_workers": 12,
            "prefetch_factor": 2,
            "worker_partition": "event",
            "tf32": True,
            "fused_adamw": True,
            "compile_model": True,
        },
    }[arguments.runtime_profile]
    for name, value in defaults.items():
        if getattr(arguments, name) is None:
            setattr(arguments, name, value)
    return arguments


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Exactly one visible CUDA device is required. "
                "Set CUDA_VISIBLE_DEVICES to one free GPU."
            )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def configure_tf32(enabled: bool) -> dict[str, bool | str]:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return {
        "enabled": enabled,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
        ),
    }


def portable_model_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    original = getattr(model, "_orig_mod", None)
    return (model if original is None else original).state_dict()


def shutdown_loader_workers(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"NaN or inf detected in {name}")


def require_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(
                f"NaN or inf detected in gradient: {name}"
            )


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("Labels and scores must have the same shape")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("ROC AUC labels must be Z=0 or H=1")
    if not np.isfinite(scores).all():
        raise FloatingPointError("NaN or inf detected in AUC scores")

    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC AUC requires both H and Z events")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        stop = start + 1
        while (
            stop < labels.size
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop

    positive_rank_sum = float(ranks[labels == 1].sum())
    auc = (
        positive_rank_sum
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    if not math.isfinite(auc):
        raise FloatingPointError("Non-finite validation ROC AUC")
    return auc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_terms: list[torch.Tensor] = []
    event_count = 0
    last_logits: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None

    for step, cpu_batch in enumerate(loader, start=1):
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_function(logits, batch["labels"])
        loss.backward()
        if step == 1 or step % 100 == 0:
            require_finite(logits, "train logits")
            require_finite(loss, "train loss")
            require_finite_gradients(model)
        optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        loss_terms.append(loss.detach() * batch_size)
        event_count += batch_size
        last_logits = logits
        last_loss = loss

    if event_count == 0:
        raise RuntimeError("Training loader produced no events")
    assert last_logits is not None and last_loss is not None
    require_finite(last_logits, "final train logits")
    require_finite(last_loss, "final train loss")
    require_finite_gradients(model)
    loss_sum = float(torch.stack(loss_terms).sum().cpu())
    return loss_sum / event_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    name: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    event_count = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            require_finite(logits, f"{name} logits")
            loss = loss_function(logits, batch["labels"])
            require_finite(loss, f"{name} loss")

            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            event_count += batch_size
            all_labels.append(batch["labels"].detach().cpu())
            all_scores.append(torch.sigmoid(logits).detach().cpu())

    if event_count == 0:
        raise RuntimeError(f"{name.capitalize()} loader produced no events")
    labels = torch.cat(all_labels).numpy()
    scores = torch.cat(all_scores).numpy()
    return (
        loss_sum / event_count,
        binary_roc_auc(labels, scores),
        labels,
        scores,
    )


def roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("Labels and scores must have the same shape")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    thresholds = np.r_[distinct, labels.size - 1]

    true_positive = np.cumsum(sorted_labels)[thresholds]
    false_positive = 1 + thresholds - true_positive
    true_positive = np.r_[0, true_positive]
    false_positive = np.r_[0, false_positive]

    return (
        false_positive / false_positive[-1],
        true_positive / true_positive[-1],
    )


def checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int],
    metadata: dict,
    arguments: argparse.Namespace,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": portable_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "feature_dimensions": metadata["feature_dimensions"],
        "tau_decay_num_embeddings": metadata[
            "tau_decay_num_embeddings"
        ],
        "training_arguments": {
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "seed": RANDOM_SEED,
            "runtime_profile": arguments.runtime_profile,
            "num_workers": arguments.num_workers,
            "prefetch_factor": arguments.prefetch_factor,
            "worker_partition": arguments.worker_partition,
            "input_projection": "dense_masked",
            "tf32": arguments.tf32,
            "tf32_configuration": arguments.tf32_configuration,
            "fused_adamw": arguments.fused_adamw,
            "torch_compile": arguments.compile_model,
            "torch_compile_dynamic": arguments.compile_model,
        },
    }


def save_history(
    path: Path,
    history: list[dict[str, float | int]],
    best_epoch: int,
    arguments: argparse.Namespace,
    device: torch.device,
) -> None:
    document = {
        "purpose": (
            "Pipeline test only; the sample has a known jet pT generation issue."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "configuration": {
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "runtime_profile": arguments.runtime_profile,
            "num_workers": arguments.num_workers,
            "prefetch_factor": arguments.prefetch_factor,
            "worker_partition": arguments.worker_partition,
            "input_projection": "dense_masked",
            "tf32": arguments.tf32,
            "tf32_configuration": arguments.tf32_configuration,
            "fused_adamw": arguments.fused_adamw,
            "torch_compile": arguments.compile_model,
            "torch_compile_dynamic": arguments.compile_model,
        },
        "epochs": history,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")


def plot_summary(
    history: list[dict[str, float | int]],
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_auc: float,
    output_path: Path,
) -> None:
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    validation_loss = [
        float(item["validation_loss"]) for item in history
    ]
    validation_auc = [
        float(item["validation_auc"]) for item in history
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
        layout="constrained",
    )
    axes[0, 0].plot(epochs, train_loss, label="Train")
    axes[0, 0].plot(
        epochs, validation_loss, marker="o", label="Validation"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("BCE loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        validation_auc,
        color="tab:green",
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation ROC AUC")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.3)

    false_positive_rate, true_positive_rate = roc_curve(
        test_labels, test_scores
    )
    axes[1, 0].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Test AUC = {test_auc:.4f}",
    )
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="0.5")
    axes[1, 0].set_xlabel("False positive rate")
    axes[1, 0].set_ylabel("True positive rate")
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    bins = np.linspace(0.0, 1.0, 41)
    axes[1, 1].hist(
        test_scores[test_labels == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Z",
    )
    axes[1, 1].hist(
        test_scores[test_labels == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="H",
    )
    axes[1, 1].set_xlabel("Model score P(H)")
    axes[1, 1].set_ylabel("Normalized density")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle(
        "TauSpin old-sample full training\n"
        "Known jet pT generation-condition issue - "
        "pipeline and scaling study only; not a physics result."
    )
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive")
    if arguments.num_workers < 0:
        raise ValueError("Number of workers cannot be negative")
    if arguments.prefetch_factor <= 0:
        raise ValueError("Prefetch factor must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(RANDOM_SEED)
    device = choose_device(arguments.device)
    if arguments.fused_adamw and device.type != "cuda":
        raise ValueError("Fused AdamW requires --device cuda")
    if arguments.compile_model and not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")
    arguments.tf32_configuration = configure_tf32(
        arguments.tf32 and device.type == "cuda"
    )
    metadata = json.loads(
        (arguments.processed_dir / "metadata.json").read_text()
    )

    train_dataset = TauSpinDataset(
        arguments.processed_dir,
        split="train",
        shuffle=True,
        balanced=True,
        seed=RANDOM_SEED,
        worker_partition=arguments.worker_partition,
    )
    validation_dataset = TauSpinDataset(
        arguments.processed_dir,
        split="validation",
        shuffle=False,
        balanced=False,
        seed=RANDOM_SEED,
        worker_partition=arguments.worker_partition,
    )
    test_dataset = TauSpinDataset(
        arguments.processed_dir,
        split="test",
        shuffle=False,
        balanced=False,
        seed=RANDOM_SEED,
        worker_partition=arguments.worker_partition,
    )
    loader_arguments = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_events,
    }
    if arguments.num_workers > 0:
        loader_arguments.update(
            {
                "persistent_workers": True,
                "prefetch_factor": arguments.prefetch_factor,
            }
        )
    validation_loader = DataLoader(
        validation_dataset, **loader_arguments
    )
    test_loader = DataLoader(test_dataset, **loader_arguments)

    model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
    ).to(device)
    optimizer_arguments = {
        "lr": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
    }
    if arguments.fused_adamw:
        optimizer_arguments["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_arguments,
    )
    if arguments.compile_model:
        model = torch.compile(model, dynamic=True)
    loss_function = nn.BCEWithLogitsLoss()

    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        "Runtime: "
        f"profile={arguments.runtime_profile}, "
        f"batch={arguments.batch_size}, "
        f"workers={arguments.num_workers}, "
        f"partition={arguments.worker_partition}, "
        f"TF32={arguments.tf32_configuration['enabled']}, "
        f"fused_AdamW={arguments.fused_adamw}, "
        f"compile_dynamic={arguments.compile_model}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"CUDA runtime: {torch.version.cuda}", flush=True)
        print(
            f"Visible CUDA devices: {torch.cuda.device_count()}",
            flush=True,
        )
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    print(
        f"Events: train={len(train_dataset)} balanced, "
        f"validation={len(validation_dataset)} unbalanced, "
        f"test={len(test_dataset)} unbalanced",
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_validation_loss = math.inf

    for epoch in range(1, arguments.epochs + 1):
        start_time = time.perf_counter()
        train_dataset.set_epoch(epoch - 1)
        train_loader = DataLoader(train_dataset, **loader_arguments)
        try:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
            )
        finally:
            shutdown_loader_workers(train_loader)
        (
            validation_loss,
            validation_auc,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            "validation",
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed_seconds = time.perf_counter() - start_time
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_auc": validation_auc,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds,
            "events_per_second": len(train_dataset) / elapsed_seconds,
        }
        history.append(metrics)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    metadata,
                    arguments,
                ),
                arguments.output_dir / "best_model.pt",
            )

        save_history(
            arguments.output_dir / "history.json",
            history,
            best_epoch,
            arguments,
            device,
        )
        print(
            f"Epoch {epoch:02d}/{arguments.epochs}: "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_auc={validation_auc:.6f} "
            f"lr={learning_rate:.2e} "
            f"rate={metrics['events_per_second']:.1f} events/s "
            f"time={elapsed_seconds:.1f}s",
            flush=True,
        )

    torch.save(
        checkpoint(
            model,
            optimizer,
            arguments.epochs,
            history[-1],
            metadata,
            arguments,
        ),
        arguments.output_dir / "last_model.pt",
    )
    best_checkpoint = torch.load(
        arguments.output_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    best_model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
    ).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    if arguments.compile_model:
        best_model = torch.compile(best_model, dynamic=True)

    (
        reloaded_validation_loss,
        reloaded_validation_auc,
        _,
        _,
    ) = evaluate(
        best_model,
        validation_loader,
        loss_function,
        device,
        "reloaded validation",
    )
    saved_best = best_checkpoint["metrics"]
    if not math.isclose(
        reloaded_validation_loss,
        float(saved_best["validation_loss"]),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ) or not math.isclose(
        reloaded_validation_auc,
        float(saved_best["validation_auc"]),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise RuntimeError(
            "Reloaded checkpoint does not reproduce validation metrics"
        )

    test_loss, test_auc, test_labels, test_scores = evaluate(
        best_model,
        test_loader,
        loss_function,
        device,
        "test",
    )
    maximum_auc_record = max(
        history, key=lambda item: float(item["validation_auc"])
    )
    test_metrics = {
        "purpose": (
            "Old sample with known jet pT generation-condition issue; "
            "pipeline and scaling study only, not a physics result."
        ),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": reloaded_validation_loss,
        "best_epoch_validation_auc": reloaded_validation_auc,
        "maximum_validation_auc": float(
            maximum_auc_record["validation_auc"]
        ),
        "maximum_validation_auc_epoch": int(
            maximum_auc_record["epoch"]
        ),
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_events": int(test_labels.size),
        "test_h_events": int(test_labels.sum()),
        "test_z_events": int(test_labels.size - test_labels.sum()),
    }
    (arguments.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n"
    )
    test_fpr, test_tpr = roc_curve(test_labels, test_scores)
    np.savez_compressed(
        arguments.output_dir / "test_predictions.npz",
        labels=test_labels,
        scores=test_scores,
        false_positive_rate=test_fpr,
        true_positive_rate=test_tpr,
    )
    plot_summary(
        history,
        test_labels,
        test_scores,
        test_auc,
        arguments.output_dir / "training_summary.pdf",
    )
    shutdown_loader_workers(validation_loader)
    shutdown_loader_workers(test_loader)
    print(
        f"Training complete. Best epoch: {best_epoch} "
        f"(validation_loss={best_validation_loss:.6f}). "
        f"Test loss={test_loss:.6f}, test AUC={test_auc:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
