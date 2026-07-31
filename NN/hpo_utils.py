from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    TauSpinDataset,
    WorkerPartition,
    collate_events,
    extract_event,
)
from model import TauSpinTransformer
from train import binary_roc_auc, move_batch, require_finite, roc_curve


MODEL_PROFILES: dict[str, dict[str, int]] = {
    "small": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 256,
    },
    "current": {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 4,
        "dim_feedforward": 512,
    },
    "deep": {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 6,
        "dim_feedforward": 512,
    },
    "wide": {
        "d_model": 192,
        "n_heads": 12,
        "n_layers": 4,
        "dim_feedforward": 768,
    },
    "large": {
        "d_model": 192,
        "n_heads": 12,
        "n_layers": 6,
        "dim_feedforward": 768,
    },
}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Counter):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {
            str(key): json_ready(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(json_ready(document), indent=2) + "\n")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_information(repository: Path) -> dict[str, Any]:
    head = run_command(
        ["git", "-C", str(repository), "rev-parse", "HEAD"]
    )
    status = run_command(
        ["git", "-C", str(repository), "status", "--short"]
    )
    diff_stat = run_command(
        ["git", "-C", str(repository), "diff", "--stat"]
    )
    diff = run_command(["git", "-C", str(repository), "diff"])
    if head["returncode"] != 0 or status["returncode"] != 0:
        raise RuntimeError("Could not inspect Git state")
    return {
        "head": head["stdout"],
        "working_tree_clean": status["stdout"] == "",
        "status_short": status["stdout"],
        "diff_stat": diff_stat["stdout"],
        "diff_sha256": sha256_bytes(diff["stdout"].encode()),
    }


def environment_information(
    repository: Path,
    physical_gpu_index: int | None,
) -> dict[str, Any]:
    gpu = {}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_name": properties.name,
            "visible_device_total_memory": properties.total_memory,
            "physical_gpu_index": physical_gpu_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    driver = run_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "created_at_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": (
            driver["stdout"].splitlines()[0]
            if driver["returncode"] == 0 and driver["stdout"]
            else None
        ),
        "gpu": gpu,
        "git": git_information(repository),
    }


def configure_tf32() -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {
        "precision": "tf32",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
        ),
    }


def create_streaming_loader(
    processed_dir: Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    balanced: bool,
    seed: int,
    worker_partition: WorkerPartition = "shard",
) -> tuple[TauSpinDataset, DataLoader]:
    dataset = TauSpinDataset(
        processed_dir,
        split=split,
        shuffle=shuffle,
        balanced=balanced,
        seed=seed,
        worker_partition=worker_partition,
    )
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_events,
        "drop_last": False,
    }
    if num_workers > 0:
        options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
        )
    return dataset, DataLoader(dataset, **options)


def create_list_loader(
    events: list[dict[str, torch.Tensor]],
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_events,
        "drop_last": False,
    }
    if num_workers > 0:
        options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
        )
    return DataLoader(events, **options)


def shutdown_loader_workers(
    loader: DataLoader,
    iterator: Any | None = None,
) -> dict[str, Any]:
    active_iterator = iterator
    if active_iterator is None:
        active_iterator = getattr(loader, "_iterator", None)
    workers_before = []
    workers_after = []
    if active_iterator is not None and hasattr(active_iterator, "_workers"):
        workers_before = [
            {"pid": worker.pid, "alive": worker.is_alive()}
            for worker in active_iterator._workers
        ]
    if (
        active_iterator is not None
        and hasattr(active_iterator, "_shutdown_workers")
    ):
        active_iterator._shutdown_workers()
    if active_iterator is not None and hasattr(active_iterator, "_workers"):
        workers_after = [
            {"pid": worker.pid, "alive": worker.is_alive()}
            for worker in active_iterator._workers
        ]
    if hasattr(loader, "_iterator"):
        loader._iterator = None
    return {
        "workers_before_shutdown": workers_before,
        "workers_after_shutdown": workers_after,
        "workers_alive_after_shutdown": sum(
            int(worker["alive"]) for worker in workers_after
        ),
    }


def event_fingerprint(events: Iterable[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        for name in sorted(event):
            tensor = event[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_validation_manifest(
    processed_dir: Path,
    metadata: Mapping[str, Any],
    *,
    per_class: int,
) -> dict[str, Any]:
    entries_by_sample: dict[str, list[dict[str, Any]]] = {}
    for sample, label in (("H", 1), ("Z", 0)):
        selected: list[dict[str, Any]] = []
        remaining = per_class
        for record in metadata["shards"]["validation"][sample]:
            take = min(remaining, int(record["events"]))
            selected.extend(
                {
                    "sample": sample,
                    "shard_path": record["path"],
                    "local_index": index,
                    "label": label,
                }
                for index in range(take)
            )
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"Validation {sample} has fewer than {per_class} events"
            )
        entries_by_sample[sample] = selected

    entries = []
    for index in range(per_class):
        entries.append(entries_by_sample["H"][index])
        entries.append(entries_by_sample["Z"][index])
    manifest = {
        "version": 1,
        "split": "validation",
        "selection": "first events per class, interleaved H/Z",
        "per_class": per_class,
        "total_events": len(entries),
        "processed_dir": str(processed_dir.resolve()),
        "metadata_sha256": sha256_file(processed_dir / "metadata.json"),
        "entries": entries,
    }
    events = load_manifest_events(processed_dir, manifest)
    manifest["event_tensor_fingerprint_sha256"] = event_fingerprint(events)
    manifest["manifest_entries_sha256"] = sha256_bytes(
        json.dumps(entries, sort_keys=True).encode()
    )
    return manifest


def load_manifest_events(
    processed_dir: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, torch.Tensor]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    entries = list(manifest["entries"])
    for output_index, entry in enumerate(entries):
        if entry["sample"] not in ("H", "Z"):
            raise ValueError("Manifest contains an unknown sample")
        grouped[str(entry["shard_path"])].append((output_index, entry))

    events: list[dict[str, torch.Tensor] | None] = [None] * len(entries)
    for relative_path, indexed_entries in grouped.items():
        shard = torch.load(
            processed_dir / relative_path,
            map_location="cpu",
            weights_only=True,
        )
        n_events = int(shard["labels"].shape[0])
        for output_index, entry in indexed_entries:
            local_index = int(entry["local_index"])
            if not 0 <= local_index < n_events:
                raise IndexError(
                    f"Manifest index {local_index} is outside {relative_path}"
                )
            event = extract_event(shard, local_index)
            actual_label = int(event["label"].item())
            if actual_label != int(entry["label"]):
                raise ValueError(
                    f"Manifest label mismatch in {relative_path}:{local_index}"
                )
            events[output_index] = event
    if any(event is None for event in events):
        raise RuntimeError("Manifest reconstruction left missing events")
    return [event for event in events if event is not None]


def load_or_create_validation_manifest(
    path: Path,
    processed_dir: Path,
    metadata: Mapping[str, Any],
    *,
    per_class: int,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]], bool]:
    created = False
    if path.exists():
        manifest = json.loads(path.read_text())
        if int(manifest["per_class"]) != per_class:
            raise ValueError("Existing manifest has a different class size")
        if manifest["metadata_sha256"] != sha256_file(
            processed_dir / "metadata.json"
        ):
            raise ValueError("Existing manifest metadata hash does not match")
    else:
        manifest = build_validation_manifest(
            processed_dir, metadata, per_class=per_class
        )
        write_json(path, manifest)
        created = True

    events = load_manifest_events(processed_dir, manifest)
    fingerprint = event_fingerprint(events)
    if fingerprint != manifest["event_tensor_fingerprint_sha256"]:
        raise RuntimeError("Validation subset fingerprint mismatch")
    label_counts = Counter(
        int(event["label"].item()) for event in events
    )
    if label_counts != Counter({0: per_class, 1: per_class}):
        raise RuntimeError(
            f"Unexpected validation subset labels: {label_counts}"
        )
    return manifest, events, created


def create_model(
    metadata: Mapping[str, Any],
    profile_name: str,
    dropout: float,
    device: torch.device,
) -> tuple[TauSpinTransformer, dict[str, int]]:
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {profile_name}")
    profile = MODEL_PROFILES[profile_name]
    model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
        d_model=profile["d_model"],
        n_head=profile["n_heads"],
        n_layers=profile["n_layers"],
        dim_feedforward=profile["dim_feedforward"],
        dropout=dropout,
    ).to(device)
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    return model, parameter_counts


def learning_rate_for_step(
    *,
    base_learning_rate: float,
    step: int,
    max_steps: int,
    warmup_steps: int,
    scheduler: str,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    if scheduler == "constant":
        return base_learning_rate
    if scheduler != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler}")
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(
        1.0, max(0.0, (step - warmup_steps) / decay_steps)
    )
    return 0.5 * base_learning_rate * (
        1.0 + math.cos(math.pi * progress)
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in portable_model_state_dict(model).items():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def unwrapped_model(model: nn.Module) -> nn.Module:
    """Return the eager module behind torch.compile, when present."""
    original = getattr(model, "_orig_mod", None)
    return model if original is None else original


def portable_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save keys accepted by the ordinary, uncompiled model."""
    return unwrapped_model(model).state_dict()


def background_rejection_at_signal_efficiency(
    labels: np.ndarray,
    scores: np.ndarray,
    target_efficiency: float = 0.7,
) -> dict[str, float | None]:
    false_positive_rate, true_positive_rate = roc_curve(labels, scores)
    candidates = np.flatnonzero(true_positive_rate >= target_efficiency)
    if candidates.size == 0:
        return {
            "target_signal_efficiency": target_efficiency,
            "achieved_signal_efficiency": None,
            "false_positive_rate": None,
            "background_rejection": None,
        }
    index = int(candidates[0])
    fpr = float(false_positive_rate[index])
    return {
        "target_signal_efficiency": target_efficiency,
        "achieved_signal_efficiency": float(true_positive_rate[index]),
        "false_positive_rate": fpr,
        "background_rejection": None if fpr == 0.0 else 1.0 / fpr,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    name: str,
    *,
    verify_parameters_unchanged: bool = True,
) -> dict[str, Any]:
    was_training = model.training
    fingerprint_before = (
        parameter_fingerprint(model)
        if verify_parameters_unchanged
        else None
    )
    model.eval()
    loss_sum = 0.0
    event_count = 0
    labels = []
    scores = []
    started = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            logits = model(batch)
            require_finite(logits, f"{name} logits")
            loss = loss_function(logits, batch["labels"])
            require_finite(loss, f"{name} loss")
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            event_count += batch_size
            labels.append(batch["labels"].detach().cpu())
            scores.append(torch.sigmoid(logits).detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if was_training:
        model.train()
    if event_count == 0:
        raise RuntimeError(f"{name} loader produced no events")
    label_array = torch.cat(labels).numpy()
    score_array = torch.cat(scores).numpy()
    fingerprint_after = (
        parameter_fingerprint(model)
        if verify_parameters_unchanged
        else None
    )
    parameters_unchanged = (
        fingerprint_before == fingerprint_after
        if verify_parameters_unchanged
        else None
    )
    if verify_parameters_unchanged and not parameters_unchanged:
        raise RuntimeError(f"{name} validation modified model parameters")
    return {
        "loss": loss_sum / event_count,
        "auc": binary_roc_auc(label_array, score_array),
        "event_count": event_count,
        "label_counts": dict(Counter(label_array.astype(int).tolist())),
        "elapsed_seconds": elapsed,
        "background_rejection_at_signal_efficiency_0p7": (
            background_rejection_at_signal_efficiency(
                label_array, score_array
            )
        ),
        "parameters_unchanged": parameters_unchanged,
        "labels": label_array,
        "scores": score_array,
    }


def strip_evaluation_arrays(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("labels", "scores")
    }


def make_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    trial_number: int,
    trial_parameters: Mapping[str, Any],
    model_profile: Mapping[str, int],
    parameter_counts: Mapping[str, int],
    training_state: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    manifest_hash: str,
    metadata_hash: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "purpose": (
            "Old-sample HPO smoke test only; not a physics result."
        ),
        "trial_number": trial_number,
        "model_state_dict": portable_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dimensions": metadata["feature_dimensions"],
        "tau_decay_num_embeddings": metadata[
            "tau_decay_num_embeddings"
        ],
        "tau_decay_mode_to_id": metadata["tau_decay_mode_to_id"],
        "model_profile": dict(model_profile),
        "parameter_counts": dict(parameter_counts),
        "hyperparameters": dict(trial_parameters),
        "training_state": dict(training_state),
        "best_full_validation_metrics": dict(best_metrics),
        "runtime": dict(runtime),
        "data": {
            "processed_metadata_sha256": metadata_hash,
            "validation_manifest_sha256": manifest_hash,
            "sampling": "balanced_oversampling",
            "batch_size": batch_size,
            "seed": seed,
        },
    }
