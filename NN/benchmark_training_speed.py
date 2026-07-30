from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

MODULE_START = time.perf_counter()

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    LEARNING_RATE,
    N_HEAD,
    N_LAYERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import TauSpinDataset, collate_events
from model import TauSpinTransformer
from train import (
    binary_roc_auc,
    choose_device,
    move_batch,
    set_random_seed,
)


BASELINE_FULL_VALIDATION_LOSS = 0.5680814058294793
BASELINE_FULL_VALIDATION_AUC = 0.7379231497434271
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "benchmarks"
    / "training-speed-v1"
)
WARMUP_STEPS = 100
MEASUREMENT_STEPS = 300
DETAIL_INTERVAL = 10
BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-GPU TauSpin Transformer training speed."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        required=False,
        help="Physical GPU index exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--one-epoch-spec", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--one-epoch-result", type=Path, help=argparse.SUPPRESS
    )
    return parser.parse_args()


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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


def run_command(
    command: list[str],
    *,
    check: bool = False,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_information(repository: Path) -> dict[str, Any]:
    head = run_command(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
    )["stdout"]
    status = run_command(
        ["git", "-C", str(repository), "status", "--short"],
        check=True,
    )["stdout"]
    diff_stat = run_command(
        ["git", "-C", str(repository), "diff", "--stat"],
        check=True,
    )["stdout"]
    return {
        "head": head,
        "working_tree_clean": status == "",
        "status_short": status,
        "diff_stat": diff_stat,
    }


def nvidia_query(physical_gpu_index: int) -> dict[str, Any]:
    fields = (
        "index,uuid,name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw"
    )
    result = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "-i",
            str(physical_gpu_index),
        ],
        check=True,
    )
    values = [item.strip() for item in result["stdout"].split(",")]
    return {
        "physical_index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": float(values[4]),
        "memory_used_mib": float(values[5]),
        "utilization_percent": float(values[6]),
        "temperature_c": float(values[7]),
        "power_w": float(values[8]),
    }


def gpu_processes() -> list[dict[str, Any]]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result["returncode"] != 0 or not result["stdout"]:
        return []
    processes = []
    for line in result["stdout"].splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 4:
            processes.append(
                {
                    "gpu_uuid": values[0],
                    "pid": int(values[1]),
                    "process_name": values[2],
                    "used_memory_mib": float(values[3]),
                }
            )
    return processes


class GPUMonitor:
    def __init__(self, physical_gpu_index: int, interval: float = 0.5):
        self.physical_gpu_index = physical_gpu_index
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            result = run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.physical_gpu_index),
                ]
            )
            if result["returncode"] == 0 and result["stdout"]:
                values = [
                    item.strip()
                    for item in result["stdout"].splitlines()[0].split(",")
                ]
                if len(values) == 3:
                    self.samples.append(
                        {
                            "utilization_percent": float(values[0]),
                            "memory_used_mib": float(values[1]),
                            "power_w": float(values[2]),
                        }
                    )
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        if not self.samples:
            return {
                "samples": 0,
                "mean_utilization_percent": None,
                "max_utilization_percent": None,
                "mean_memory_used_mib": None,
                "max_memory_used_mib": None,
                "mean_power_w": None,
            }
        return {
            "samples": len(self.samples),
            "mean_utilization_percent": statistics.fmean(
                item["utilization_percent"] for item in self.samples
            ),
            "max_utilization_percent": max(
                item["utilization_percent"] for item in self.samples
            ),
            "mean_memory_used_mib": statistics.fmean(
                item["memory_used_mib"] for item in self.samples
            ),
            "max_memory_used_mib": max(
                item["memory_used_mib"] for item in self.samples
            ),
            "mean_power_w": statistics.fmean(
                item["power_w"] for item in self.samples
            ),
        }


def audit_worker_partition(
    metadata: dict[str, Any],
    num_workers: int,
) -> dict[str, Any]:
    worker_count = max(1, num_workers)
    records = metadata["shards"]["train"]
    assignments: dict[str, list[list[dict[str, Any]]]] = {}
    for sample in ("H", "Z"):
        assignments[sample] = [
            list(records[sample][worker_id::worker_count])
            for worker_id in range(worker_count)
        ]

    overlap_paths: dict[str, list[str]] = {}
    union_matches = {}
    for sample in ("H", "Z"):
        paths = [
            record["path"]
            for worker_records in assignments[sample]
            for record in worker_records
        ]
        counts = Counter(paths)
        overlap_paths[sample] = sorted(
            path for path, count in counts.items() if count > 1
        )
        union_matches[sample] = set(paths) == {
            record["path"] for record in records[sample]
        }

    effective_paths = {"H": set(), "Z": set()}
    inactive_workers = []
    worker_details = []
    for worker_id in range(worker_count):
        h_records = assignments["H"][worker_id]
        z_records = assignments["Z"][worker_id]
        active = bool(h_records and z_records)
        if active:
            effective_paths["H"].update(
                record["path"] for record in h_records
            )
            effective_paths["Z"].update(
                record["path"] for record in z_records
            )
        else:
            inactive_workers.append(worker_id)
        worker_details.append(
            {
                "worker_id": worker_id,
                "active_for_balanced_iteration": active,
                "H_paths": [record["path"] for record in h_records],
                "Z_paths": [record["path"] for record in z_records],
                "H_events": sum(
                    int(record["events"]) for record in h_records
                ),
                "Z_events": sum(
                    int(record["events"]) for record in z_records
                ),
            }
        )

    missing_records = {}
    missing_events = {}
    for sample in ("H", "Z"):
        missing = [
            record
            for record in records[sample]
            if record["path"] not in effective_paths[sample]
        ]
        missing_records[sample] = [
            record["path"] for record in missing
        ]
        missing_events[sample] = sum(
            int(record["events"]) for record in missing
        )

    valid = (
        all(union_matches.values())
        and not any(overlap_paths.values())
        and sum(missing_events.values()) == 0
    )
    return {
        "valid_for_balanced_epoch": valid,
        "worker_count": worker_count,
        "worker_details": worker_details,
        "inactive_workers": inactive_workers,
        "union_matches_all_shards": union_matches,
        "overlap_paths": overlap_paths,
        "missing_records_from_balanced_iteration": missing_records,
        "missing_events_from_balanced_iteration": missing_events,
        "note": (
            "Minority-class oversampling is intentional. This audit tests "
            "unintended worker overlap and shards that cannot participate "
            "because a worker received only one class."
        ),
    }


def expected_balanced_batches(
    partition_audit: dict[str, Any],
    batch_size: int,
) -> int:
    return sum(
        math.ceil(
            2
            * max(
                int(worker["H_events"]),
                int(worker["Z_events"]),
            )
            / batch_size
        )
        for worker in partition_audit["worker_details"]
        if worker["active_for_balanced_iteration"]
    )


def precision_context(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cast_continuous_features_for_amp(
    batch: dict[str, torch.Tensor],
    precision: str,
) -> dict[str, torch.Tensor]:
    if precision not in ("bf16", "fp16"):
        return batch
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    converted = dict(batch)
    for name in (
        "event_features",
        "tau_features",
        "track_features",
        "pfo_features",
    ):
        converted[name] = batch[name].to(dtype=dtype)
    return converted


def configure_precision(precision: str) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    return {
        "precision": precision,
        "autocast_enabled": precision in ("bf16", "fp16"),
        "autocast_dtype": (
            str(torch.bfloat16)
            if precision == "bf16"
            else str(torch.float16)
            if precision == "fp16"
            else None
        ),
        "grad_scaler_enabled": precision == "fp16",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def create_loader(
    processed_dir: Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    balanced: bool,
) -> tuple[TauSpinDataset, DataLoader]:
    dataset = TauSpinDataset(
        processed_dir,
        split=split,
        shuffle=shuffle,
        balanced=balanced,
        seed=RANDOM_SEED,
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


def create_model_and_optimizer(
    metadata: dict[str, Any],
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    use_compile: bool,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    base_model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
        dropout=DROPOUT,
    ).to(device)
    model: nn.Module = base_model
    if use_compile:
        torch._dynamo.reset()
        model = torch.compile(base_model, dynamic=True)
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return base_model, model, optimizer


def optimizer_step(
    model: nn.Module,
    base_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    batch: dict[str, torch.Tensor],
    precision: str,
    scaler: torch.amp.GradScaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with precision_context(precision):
        logits = model(batch)
        loss = loss_function(logits, batch["labels"])
    if precision == "fp16":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss, logits


def tensor_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def gradients_finite(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or tensor_finite(parameter.grad)
        for parameter in model.parameters()
    )


def shutdown_loader_workers(
    loader: DataLoader,
    iterator: Any,
) -> dict[str, Any]:
    workers_before = []
    workers_after = []
    if hasattr(iterator, "_workers"):
        workers_before = [
            {
                "pid": worker.pid,
                "alive": worker.is_alive(),
            }
            for worker in iterator._workers
        ]
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    if hasattr(iterator, "_workers"):
        workers_after = [
            {
                "pid": worker.pid,
                "alive": worker.is_alive(),
            }
            for worker in iterator._workers
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


def compile_counters() -> dict[str, Any]:
    try:
        from torch._dynamo.utils import counters

        return {
            group: {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()
            if values
        }
    except Exception as error:
        return {"error": repr(error)}


def run_benchmark_worker(
    spec: dict[str, Any],
) -> dict[str, Any]:
    processed_dir = Path(spec["processed_dir"])
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    device = choose_device("cuda")
    set_random_seed(int(spec["seed"]))
    precision_info = configure_precision(spec["precision"])
    partition_audit = audit_worker_partition(
        metadata, int(spec["num_workers"])
    )
    dataset, loader = create_loader(
        processed_dir,
        split="train",
        batch_size=int(spec["batch_size"]),
        num_workers=int(spec["num_workers"]),
        prefetch_factor=int(spec["prefetch_factor"]),
        shuffle=True,
        balanced=True,
    )
    dataset.set_epoch(0)
    base_model, model, optimizer = create_model_and_optimizer(
        metadata,
        device=device,
        learning_rate=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
        use_compile=bool(spec["compile"]),
    )
    loss_function = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=spec["precision"] == "fp16"
    )
    first_parameter = next(
        parameter for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    parameter_before = first_parameter.detach().clone()

    iterator = iter(loader)
    startup_seconds = time.perf_counter() - MODULE_START
    worker_pids = [
        worker.pid for worker in getattr(iterator, "_workers", [])
    ]

    warmup_start = time.perf_counter()
    warmup_losses: list[torch.Tensor] = []
    observed_output_dtypes = set()
    for _ in range(int(spec["warmup_steps"])):
        batch = move_batch(next(iterator), device)
        batch = cast_continuous_features_for_amp(
            batch, spec["precision"]
        )
        loss, logits = optimizer_step(
            model,
            base_model,
            optimizer,
            loss_function,
            batch,
            spec["precision"],
            scaler,
        )
        warmup_losses.append(loss.detach())
        observed_output_dtypes.add(str(logits.dtype))
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    torch.cuda.reset_peak_memory_stats()
    loader_wait_seconds = 0.0
    processed_events = 0
    valid_tokens = 0
    padded_tokens = 0
    losses: list[torch.Tensor] = []
    sampled_step_seconds = []
    sampled_loader_seconds = []
    sampled_transfer_ms = []
    sampled_compute_ms = []
    label_counts = Counter()
    observed_event_numbers: list[tuple[int, int]] = []
    monitor = GPUMonitor(int(spec["physical_gpu_index"]))

    torch.cuda.synchronize()
    measurement_start = time.perf_counter()
    monitor.start()
    for measurement_index in range(int(spec["measurement_steps"])):
        sample_detail = measurement_index % DETAIL_INTERVAL == 0
        if sample_detail:
            torch.cuda.synchronize()
            sampled_step_start = time.perf_counter()

        wait_start = time.perf_counter()
        cpu_batch = next(iterator)
        wait_seconds = time.perf_counter() - wait_start
        loader_wait_seconds += wait_seconds

        batch_events = int(cpu_batch["labels"].shape[0])
        batch_valid_tokens = (
            int((~cpu_batch["padding_mask"]).sum().item()) + batch_events
        )
        batch_padded_tokens = (
            int(cpu_batch["padding_mask"].numel()) + batch_events
        )
        processed_events += batch_events
        valid_tokens += batch_valid_tokens
        padded_tokens += batch_padded_tokens
        label_values = cpu_batch["labels"].to(torch.int64).tolist()
        label_counts.update(label_values)
        observed_event_numbers.extend(
            zip(
                label_values,
                cpu_batch["event_numbers"].to(torch.int64).tolist(),
            )
        )

        if sample_detail:
            transfer_start = torch.cuda.Event(enable_timing=True)
            transfer_end = torch.cuda.Event(enable_timing=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            transfer_start.record()
        batch = move_batch(cpu_batch, device)
        batch = cast_continuous_features_for_amp(
            batch, spec["precision"]
        )
        if sample_detail:
            transfer_end.record()
            compute_start.record()
        loss, logits = optimizer_step(
            model,
            base_model,
            optimizer,
            loss_function,
            batch,
            spec["precision"],
            scaler,
        )
        losses.append(loss.detach())
        observed_output_dtypes.add(str(logits.dtype))
        if sample_detail:
            compute_end.record()
            torch.cuda.synchronize()
            sampled_step_seconds.append(
                time.perf_counter() - sampled_step_start
            )
            sampled_loader_seconds.append(wait_seconds)
            sampled_transfer_ms.append(
                transfer_start.elapsed_time(transfer_end)
            )
            sampled_compute_ms.append(
                compute_start.elapsed_time(compute_end)
            )

    torch.cuda.synchronize()
    measured_wall_seconds = time.perf_counter() - measurement_start
    gpu_monitor = monitor.stop()
    loss_values = torch.stack(losses).float().cpu().numpy()
    warmup_loss_values = (
        torch.stack(warmup_losses).float().cpu().numpy()
    )
    finite_gradients = gradients_finite(base_model)
    parameter_delta_max = float(
        (first_parameter.detach() - parameter_before).abs().max().cpu()
    )
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated() / 1024.0**2
    )
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024.0**2
    workers = shutdown_loader_workers(loader, iterator)

    duplicate_event_number_pairs = (
        len(observed_event_numbers) - len(set(observed_event_numbers))
    )
    finite_losses = bool(np.isfinite(loss_values).all())
    loss_range_reasonable = bool(
        finite_losses
        and loss_values.min() >= 0.0
        and loss_values.max() < 10.0
    )
    valid = (
        finite_losses
        and finite_gradients
        and parameter_delta_max > 0.0
        and loss_range_reasonable
        and workers["workers_alive_after_shutdown"] == 0
        and partition_audit["valid_for_balanced_epoch"]
        and processed_events
        == int(spec["measurement_steps"]) * int(spec["batch_size"])
        and set(label_counts) == {0, 1}
    )
    step_samples = sampled_step_seconds
    result = {
        "run_id": spec["run_id"],
        "stage": spec["stage"],
        "configuration": spec,
        "valid": valid,
        "startup_seconds": startup_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_wall_seconds": measured_wall_seconds,
        "measured_optimizer_steps": int(spec["measurement_steps"]),
        "processed_events": processed_events,
        "valid_tokens": valid_tokens,
        "padded_tokens": padded_tokens,
        "padding_fraction": (
            1.0 - valid_tokens / padded_tokens
            if padded_tokens
            else None
        ),
        "events_per_second": processed_events / measured_wall_seconds,
        "tokens_per_second": valid_tokens / measured_wall_seconds,
        "padded_tokens_per_second": (
            padded_tokens / measured_wall_seconds
        ),
        "mean_step_seconds": (
            measured_wall_seconds / int(spec["measurement_steps"])
        ),
        "sampled_median_step_seconds": statistics.median(step_samples),
        "sampled_p95_step_seconds": float(
            np.percentile(step_samples, 95)
        ),
        "detail_sample_count": len(step_samples),
        "loader_wait_seconds_total": loader_wait_seconds,
        "loader_wait_fraction_of_wall": (
            loader_wait_seconds / measured_wall_seconds
        ),
        "sampled_loader_wait_ms_mean": (
            1000.0 * statistics.fmean(sampled_loader_seconds)
        ),
        "sampled_transfer_ms_mean": statistics.fmean(
            sampled_transfer_ms
        ),
        "sampled_gpu_compute_ms_mean": statistics.fmean(
            sampled_compute_ms
        ),
        "sampled_gpu_total_ms_mean": statistics.fmean(
            transfer + compute
            for transfer, compute in zip(
                sampled_transfer_ms, sampled_compute_ms
            )
        ),
        "peak_gpu_memory_allocated_mib": peak_allocated_mib,
        "peak_gpu_memory_reserved_mib": peak_reserved_mib,
        "gpu_monitor": gpu_monitor,
        "loss_mean": float(loss_values.mean()),
        "loss_min": float(loss_values.min()),
        "loss_max": float(loss_values.max()),
        "warmup_loss_mean": float(warmup_loss_values.mean()),
        "losses_finite": finite_losses,
        "gradients_finite": finite_gradients,
        "loss_range_reasonable": loss_range_reasonable,
        "parameter_delta_max": parameter_delta_max,
        "optimizer_update_performed": parameter_delta_max > 0.0,
        "label_counts": {
            "Z": int(label_counts[0]),
            "H": int(label_counts[1]),
        },
        "observed_label_event_number_duplicate_count": (
            duplicate_event_number_pairs
        ),
        "event_number_duplicate_note": (
            "eventNumber is known to be non-unique. Worker correctness is "
            "therefore decided by disjoint shard assignment, complete shard "
            "coverage, and expected balanced labels rather than this count."
        ),
        "precision": precision_info,
        "observed_logits_dtypes": sorted(observed_output_dtypes),
        "grad_scaler_scale_final": (
            float(scaler.get_scale())
            if spec["precision"] == "fp16"
            else None
        ),
        "worker_partition_audit": partition_audit,
        "worker_pids": worker_pids,
        "worker_shutdown": workers,
        "compile_counters": (
            compile_counters() if spec["compile"] else {}
        ),
        "cuda": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
    }
    return result


def evaluate_with_precision(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    precision: str,
) -> tuple[float, float, int]:
    model.eval()
    loss_sum = 0.0
    event_count = 0
    labels = []
    scores = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            batch = cast_continuous_features_for_amp(batch, precision)
            with precision_context(precision):
                logits = model(batch)
                loss = loss_function(logits, batch["labels"])
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(loss.float().cpu()) * batch_size
            event_count += batch_size
            labels.append(batch["labels"].cpu())
            scores.append(torch.sigmoid(logits.float()).cpu())
    label_array = torch.cat(labels).numpy()
    score_array = torch.cat(scores).numpy()
    return (
        loss_sum / event_count,
        binary_roc_auc(label_array, score_array),
        event_count,
    )


def run_one_epoch_worker(spec: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(spec["processed_dir"])
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    device = choose_device("cuda")
    set_random_seed(int(spec["seed"]))
    precision_info = configure_precision(spec["precision"])
    partition_audit = audit_worker_partition(
        metadata, int(spec["num_workers"])
    )
    if not partition_audit["valid_for_balanced_epoch"]:
        raise RuntimeError(
            "Selected runtime has invalid balanced multi-worker partition"
        )
    train_dataset, train_loader = create_loader(
        processed_dir,
        split="train",
        batch_size=int(spec["batch_size"]),
        num_workers=int(spec["num_workers"]),
        prefetch_factor=int(spec["prefetch_factor"]),
        shuffle=True,
        balanced=True,
    )
    validation_dataset, validation_loader = create_loader(
        processed_dir,
        split="validation",
        batch_size=int(spec["batch_size"]),
        num_workers=int(spec["num_workers"]),
        prefetch_factor=int(spec["prefetch_factor"]),
        shuffle=False,
        balanced=False,
    )
    train_dataset.set_epoch(0)
    base_model, model, optimizer = create_model_and_optimizer(
        metadata,
        device=device,
        learning_rate=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
        use_compile=bool(spec["compile"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=spec["precision"] == "fp16"
    )
    loss_function = nn.BCEWithLogitsLoss()
    iterator = iter(train_loader)
    expected_steps = expected_balanced_batches(
        partition_audit, int(spec["batch_size"])
    )
    losses = []
    event_count = 0
    steps = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    for cpu_batch in iterator:
        batch = move_batch(cpu_batch, device)
        batch = cast_continuous_features_for_amp(
            batch, spec["precision"]
        )
        loss, _ = optimizer_step(
            model,
            base_model,
            optimizer,
            loss_function,
            batch,
            spec["precision"],
            scaler,
        )
        losses.append(loss.detach())
        event_count += int(batch["labels"].shape[0])
        steps += 1
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    finite_gradients = gradients_finite(base_model)
    loss_values = torch.stack(losses).float().cpu().numpy()
    train_workers = shutdown_loader_workers(
        train_loader, iterator
    )

    validation_iterator = iter(validation_loader)
    validation_loss, validation_auc, validation_events = (
        evaluate_with_precision(
            model,
            validation_loader,
            loss_function,
            device,
            spec["precision"],
        )
    )
    validation_workers = shutdown_loader_workers(
        validation_loader, validation_iterator
    )
    auc_difference = validation_auc - BASELINE_FULL_VALIDATION_AUC
    return {
        "configuration": spec,
        "optimizer_steps": steps,
        "expected_optimizer_steps": expected_steps,
        "train_events": event_count,
        "train_seconds": train_seconds,
        "events_per_second": event_count / train_seconds,
        "train_loss": float(
            np.average(
                loss_values,
                weights=[
                    int(spec["batch_size"])
                ] * (len(loss_values) - 1)
                + [
                    event_count
                    - int(spec["batch_size"]) * (len(loss_values) - 1)
                ],
            )
        ),
        "losses_finite": bool(np.isfinite(loss_values).all()),
        "gradients_finite": finite_gradients,
        "validation_events": validation_events,
        "full_validation_loss": validation_loss,
        "full_validation_auc": validation_auc,
        "baseline_full_validation_loss": (
            BASELINE_FULL_VALIDATION_LOSS
        ),
        "baseline_full_validation_auc": BASELINE_FULL_VALIDATION_AUC,
        "validation_auc_difference": auc_difference,
        "validation_auc_within_0p002": abs(auc_difference) <= 0.002,
        "peak_gpu_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024.0**2
        ),
        "peak_gpu_memory_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1024.0**2
        ),
        "precision": precision_info,
        "worker_partition_audit": partition_audit,
        "train_worker_shutdown": train_workers,
        "validation_worker_shutdown": validation_workers,
        "compile_counters": (
            compile_counters() if spec["compile"] else {}
        ),
        "valid": (
            steps == expected_steps
            and event_count == len(train_dataset)
            and bool(np.isfinite(loss_values).all())
            and finite_gradients
            and math.isfinite(validation_loss)
            and math.isfinite(validation_auc)
            and train_workers["workers_alive_after_shutdown"] == 0
            and validation_workers["workers_alive_after_shutdown"] == 0
        ),
    }


def worker_entry(spec_path: Path, result_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    try:
        result = run_benchmark_worker(spec)
        result["worker_exception"] = None
    except Exception as error:
        result = {
            "run_id": spec.get("run_id"),
            "stage": spec.get("stage"),
            "configuration": spec,
            "valid": False,
            "worker_exception": repr(error),
        }
        write_json(result_path, result)
        raise
    write_json(result_path, result)


def one_epoch_entry(spec_path: Path, result_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    try:
        result = run_one_epoch_worker(spec)
        result["worker_exception"] = None
    except Exception as error:
        result = {
            "configuration": spec,
            "valid": False,
            "worker_exception": repr(error),
        }
        write_json(result_path, result)
        raise
    write_json(result_path, result)


def configuration_key(spec: dict[str, Any]) -> str:
    return (
        f"nw{spec['num_workers']}_pf{spec['prefetch_factor']}_"
        f"{spec['precision']}_"
        f"{'compile' if spec['compile'] else 'eager'}"
    )


def result_group_summary(
    results: list[dict[str, Any]],
    group_key: str,
) -> dict[str, Any]:
    group = [
        item
        for item in results
        if configuration_key(item["configuration"]) == group_key
        and item.get("valid", False)
    ]
    if not group:
        return {
            "configuration_key": group_key,
            "valid_runs": 0,
            "valid": False,
        }
    fields = (
        "measured_wall_seconds",
        "events_per_second",
        "tokens_per_second",
        "mean_step_seconds",
        "loader_wait_seconds_total",
        "loader_wait_fraction_of_wall",
        "sampled_transfer_ms_mean",
        "sampled_gpu_compute_ms_mean",
        "peak_gpu_memory_allocated_mib",
        "peak_gpu_memory_reserved_mib",
        "loss_mean",
    )
    summary = {
        "configuration_key": group_key,
        "configuration": group[0]["configuration"],
        "valid_runs": len(group),
        "valid": True,
        "run_ids": [item["run_id"] for item in group],
    }
    for field in fields:
        summary[f"median_{field}"] = statistics.median(
            float(item[field]) for item in group
        )
    summary["median_gpu_utilization_percent"] = statistics.median(
        float(item["gpu_monitor"]["mean_utilization_percent"])
        for item in group
        if item["gpu_monitor"]["mean_utilization_percent"] is not None
    )
    return summary


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    configuration = result.get("configuration", {})
    monitor = result.get("gpu_monitor", {})
    partition = result.get("worker_partition_audit", {})
    return {
        "run_id": result.get("run_id"),
        "stage": result.get("stage"),
        "configuration_key": (
            configuration_key(configuration)
            if configuration
            else None
        ),
        "repeat": configuration.get("repeat"),
        "num_workers": configuration.get("num_workers"),
        "prefetch_factor": configuration.get("prefetch_factor"),
        "precision": configuration.get("precision"),
        "compile": configuration.get("compile"),
        "valid": result.get("valid"),
        "worker_exception": result.get("worker_exception"),
        "startup_seconds": result.get("startup_seconds"),
        "warmup_seconds": result.get("warmup_seconds"),
        "subprocess_wall_seconds": result.get(
            "subprocess_wall_seconds"
        ),
        "measured_wall_seconds": result.get(
            "measured_wall_seconds"
        ),
        "processed_events": result.get("processed_events"),
        "valid_tokens": result.get("valid_tokens"),
        "padded_tokens": result.get("padded_tokens"),
        "events_per_second": result.get("events_per_second"),
        "tokens_per_second": result.get("tokens_per_second"),
        "mean_step_seconds": result.get("mean_step_seconds"),
        "sampled_median_step_seconds": result.get(
            "sampled_median_step_seconds"
        ),
        "sampled_p95_step_seconds": result.get(
            "sampled_p95_step_seconds"
        ),
        "loader_wait_seconds_total": result.get(
            "loader_wait_seconds_total"
        ),
        "loader_wait_fraction_of_wall": result.get(
            "loader_wait_fraction_of_wall"
        ),
        "sampled_transfer_ms_mean": result.get(
            "sampled_transfer_ms_mean"
        ),
        "sampled_gpu_compute_ms_mean": result.get(
            "sampled_gpu_compute_ms_mean"
        ),
        "peak_gpu_memory_allocated_mib": result.get(
            "peak_gpu_memory_allocated_mib"
        ),
        "mean_gpu_utilization_percent": monitor.get(
            "mean_utilization_percent"
        ),
        "max_gpu_utilization_percent": monitor.get(
            "max_utilization_percent"
        ),
        "loss_mean": result.get("loss_mean"),
        "loss_min": result.get("loss_min"),
        "loss_max": result.get("loss_max"),
        "missing_worker_partition_events": (
            sum(
                partition.get(
                    "missing_events_from_balanced_iteration", {}
                ).values()
            )
            if partition
            else None
        ),
    }


def save_results(
    output_dir: Path,
    results: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    write_json(
        output_dir / "benchmark_results.json",
        {
            "runs": results,
            "aggregates": aggregates,
        },
    )
    rows = [flatten_result(result) for result in results]
    if rows:
        with (output_dir / "benchmark_results.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def aggregate_all(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = []
    for result in results:
        if "configuration" not in result:
            continue
        key = configuration_key(result["configuration"])
        if key not in keys:
            keys.append(key)
    return [
        result_group_summary(results, key)
        for key in keys
    ]


def ranked_valid_groups(
    results: list[dict[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    stage_results = [
        result for result in results if result.get("stage") == stage
    ]
    summaries = aggregate_all(stage_results)
    return sorted(
        [summary for summary in summaries if summary["valid"]],
        key=lambda item: (
            float(item["median_measured_wall_seconds"]),
            -float(item["median_tokens_per_second"]),
        ),
    )


def make_spec(
    *,
    run_id: str,
    stage: str,
    repeat: int,
    processed_dir: Path,
    physical_gpu_index: int,
    num_workers: int,
    prefetch_factor: int,
    precision: str,
    compile_model: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "repeat": repeat,
        "processed_dir": str(processed_dir),
        "physical_gpu_index": physical_gpu_index,
        "warmup_steps": WARMUP_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "detail_interval": DETAIL_INTERVAL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor,
        "precision": precision,
        "compile": compile_model,
        "d_model": D_MODEL,
        "n_heads": N_HEAD,
        "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
    }


def run_worker_subprocess(
    spec: dict[str, Any],
    *,
    script_path: Path,
    temporary_dir: Path,
    log_prefix: str,
) -> dict[str, Any]:
    spec_path = temporary_dir / f"{spec['run_id']}.spec.json"
    result_path = temporary_dir / f"{spec['run_id']}.result.json"
    write_json(spec_path, spec)
    command = [
        sys.executable,
        str(script_path),
        "--worker-spec",
        str(spec_path),
        "--worker-result",
        str(result_path),
    ]
    environment = os.environ.copy()
    if spec["compile"]:
        environment["TORCH_LOGS"] = "recompiles"
    print(
        f"\n[{log_prefix}] START {spec['run_id']} "
        f"{configuration_key(spec)}",
        flush=True,
    )
    subprocess_start = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    subprocess_wall_seconds = time.perf_counter() - subprocess_start
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {
            "run_id": spec["run_id"],
            "stage": spec["stage"],
            "configuration": spec,
            "valid": False,
            "worker_exception": (
                f"Worker exited {completed.returncode} without result"
            ),
        }
    result["subprocess_returncode"] = completed.returncode
    result["subprocess_wall_seconds"] = subprocess_wall_seconds
    result["stderr_recompile_lines"] = [
        line
        for line in completed.stderr.splitlines()
        if "recompil" in line.lower()
    ]
    print(
        f"[{log_prefix}] END {spec['run_id']} "
        f"return={completed.returncode} "
        f"valid={result.get('valid')} "
        f"wall={result.get('measured_wall_seconds')} "
        f"tokens/s={result.get('tokens_per_second')}",
        flush=True,
    )
    return result


def run_one_epoch_subprocess(
    spec: dict[str, Any],
    *,
    script_path: Path,
    temporary_dir: Path,
) -> dict[str, Any]:
    spec_path = temporary_dir / "one_epoch.spec.json"
    result_path = temporary_dir / "one_epoch.result.json"
    write_json(spec_path, spec)
    command = [
        sys.executable,
        str(script_path),
        "--one-epoch-spec",
        str(spec_path),
        "--one-epoch-result",
        str(result_path),
    ]
    environment = os.environ.copy()
    if spec["compile"]:
        environment["TORCH_LOGS"] = "recompiles"
    print("\n[one-epoch] START", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if not result_path.exists():
        raise RuntimeError(
            f"One-epoch verification produced no result: "
            f"{completed.stderr}"
        )
    result = json.loads(result_path.read_text())
    result["subprocess_returncode"] = completed.returncode
    result["subprocess_wall_seconds"] = elapsed
    print(
        f"[one-epoch] END return={completed.returncode} "
        f"valid={result.get('valid')} "
        f"AUC={result.get('full_validation_auc')} "
        f"time={result.get('train_seconds')}",
        flush=True,
    )
    return result


def plot_benchmark(
    aggregates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [item for item in aggregates if item.get("valid")]
    rows.sort(key=lambda item: item["median_tokens_per_second"])
    labels = [item["configuration_key"] for item in rows]
    y = np.arange(len(rows))
    tokens = [item["median_tokens_per_second"] for item in rows]
    events = [item["median_events_per_second"] for item in rows]
    wall = [item["median_measured_wall_seconds"] for item in rows]
    loader_ms = [
        1000.0
        * item["median_loader_wait_seconds_total"]
        / MEASUREMENT_STEPS
        for item in rows
    ]
    compute_ms = [
        item["median_sampled_gpu_compute_ms_mean"] for item in rows
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
        layout="constrained",
    )
    colors = {
        "blue": "#3B6FB6",
        "orange": "#D9822B",
        "gold": "#B58B1B",
        "gray": "#7A8288",
    }
    axes[0, 0].barh(y, tokens, color=colors["blue"])
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].set_xlabel("Valid tokens / second")
    axes[0, 0].set_title("Training throughput by runtime configuration")
    axes[0, 0].grid(axis="x", alpha=0.25)

    axes[0, 1].barh(y, events, color=colors["orange"])
    axes[0, 1].set_yticks(y, labels)
    axes[0, 1].set_xlabel("Events / second")
    axes[0, 1].set_title("Event throughput")
    axes[0, 1].grid(axis="x", alpha=0.25)

    axes[1, 0].barh(y, wall, color=colors["gold"])
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].set_xlabel(
        f"Wall time for {MEASUREMENT_STEPS} optimizer steps [s]"
    )
    axes[1, 0].set_title("Measured wall time (lower is better)")
    axes[1, 0].grid(axis="x", alpha=0.25)

    height = 0.38
    axes[1, 1].barh(
        y - height / 2,
        loader_ms,
        height=height,
        label="DataLoader wait / step",
        color=colors["gray"],
    )
    axes[1, 1].barh(
        y + height / 2,
        compute_ms,
        height=height,
        label="GPU compute / sampled step",
        color=colors["blue"],
    )
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].set_xlabel("Milliseconds / step")
    axes[1, 1].set_title("Input wait and GPU compute timing")
    axes[1, 1].grid(axis="x", alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle(
        "TauSpin single-GPU training speed benchmark\n"
        "Old sample; runtime study only, not a physics result."
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def orchestrate(arguments: argparse.Namespace) -> None:
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_stream = (output_dir / "benchmark.log").open("w")
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    benchmark_start = time.perf_counter()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if arguments.physical_gpu_index is None:
        if not visible or "," in visible:
            raise RuntimeError(
                "Set CUDA_VISIBLE_DEVICES to exactly one physical GPU "
                "and pass --physical-gpu-index."
            )
        arguments.physical_gpu_index = int(visible)
    if visible != str(arguments.physical_gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must match --physical-gpu-index"
        )
    device = choose_device("cuda")
    del device

    repository = Path(__file__).resolve().parents[1]
    processed_dir = arguments.processed_dir.resolve()
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    git_info = git_information(repository)
    gpu_info = nvidia_query(arguments.physical_gpu_index)
    foreign_processes = [
        process
        for process in gpu_processes()
        if process["gpu_uuid"] == gpu_info["uuid"]
        and process["pid"] != os.getpid()
    ]
    if foreign_processes:
        raise RuntimeError(
            f"Selected GPU is not free: {foreign_processes}"
        )

    system_info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "visible_cuda_devices": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices_environment": visible,
        "physical_gpu": gpu_info,
        "gpu_processes_before": gpu_processes(),
        "git": git_info,
    }
    write_json(output_dir / "system_info.json", system_info)
    benchmark_config = {
        "purpose": (
            "Single-GPU runtime benchmark for short Optuna trials; "
            "no hyperparameter optimization and no physics conclusion."
        ),
        "processed_dir": str(processed_dir),
        "dataset_metadata": metadata,
        "warmup_steps": WARMUP_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "detail_interval": DETAIL_INTERVAL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
        "d_model": D_MODEL,
        "n_heads": N_HEAD,
        "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "validation_during_benchmark": False,
        "test_loaded": False,
        "checkpoint_saved": False,
        "git": git_info,
    }
    write_json(output_dir / "benchmark_config.json", benchmark_config)
    temporary_dir = output_dir / ".tmp"
    temporary_dir.mkdir()
    script_path = Path(__file__).resolve()
    results: list[dict[str, Any]] = []

    def run(spec: dict[str, Any], prefix: str) -> dict[str, Any]:
        result = run_worker_subprocess(
            spec,
            script_path=script_path,
            temporary_dir=temporary_dir,
            log_prefix=prefix,
        )
        results.append(result)
        save_results(output_dir, results, aggregate_all(results))
        return result

    print("Stage 1: DataLoader workers", flush=True)
    for num_workers in (0, 2, 4, 8):
        run(
            make_spec(
                run_id=f"stage1_nw{num_workers}_r1",
                stage="stage1_dataloader",
                repeat=1,
                processed_dir=processed_dir,
                physical_gpu_index=arguments.physical_gpu_index,
                num_workers=num_workers,
                prefetch_factor=2,
                precision="fp32",
                compile_model=False,
            ),
            "stage1",
        )
    stage1_ranked = ranked_valid_groups(
        results, "stage1_dataloader"
    )
    if len(stage1_ranked) < 2:
        raise RuntimeError(
            "Fewer than two valid Stage 1 configurations"
        )
    stage1_top2 = stage1_ranked[:2]
    for summary in stage1_top2:
        base = summary["configuration"]
        for repeat in (2, 3):
            run(
                make_spec(
                    run_id=(
                        f"stage1_nw{base['num_workers']}_r{repeat}"
                    ),
                    stage="stage1_dataloader",
                    repeat=repeat,
                    processed_dir=processed_dir,
                    physical_gpu_index=arguments.physical_gpu_index,
                    num_workers=int(base["num_workers"]),
                    prefetch_factor=2,
                    precision="fp32",
                    compile_model=False,
                ),
                "stage1-repeat",
            )
    stage1_best = ranked_valid_groups(
        results, "stage1_dataloader"
    )[0]
    best_num_workers = int(
        stage1_best["configuration"]["num_workers"]
    )
    print(
        f"Stage 1 best num_workers={best_num_workers}",
        flush=True,
    )

    print("\nStage 2: numerical precision", flush=True)
    for precision in ("fp32", "tf32", "bf16", "fp16"):
        run(
            make_spec(
                run_id=f"stage2_{precision}_r1",
                stage="stage2_precision",
                repeat=1,
                processed_dir=processed_dir,
                physical_gpu_index=arguments.physical_gpu_index,
                num_workers=best_num_workers,
                prefetch_factor=2,
                precision=precision,
                compile_model=False,
            ),
            "stage2",
        )
    stage2_initial = [
        result
        for result in results
        if result.get("stage") == "stage2_precision"
        and result.get("valid")
    ]
    fp32_reference = next(
        result
        for result in stage2_initial
        if result["configuration"]["precision"] == "fp32"
    )
    for result in stage2_initial:
        result["loss_mean_difference_from_fp32"] = (
            float(result["loss_mean"])
            - float(fp32_reference["loss_mean"])
        )
        result["numerically_stable_vs_fp32"] = (
            abs(result["loss_mean_difference_from_fp32"]) <= 0.05
        )
        if not result["numerically_stable_vs_fp32"]:
            result["valid"] = False
    save_results(output_dir, results, aggregate_all(results))
    stage2_ranked = ranked_valid_groups(
        results, "stage2_precision"
    )
    if len(stage2_ranked) < 2:
        raise RuntimeError(
            "Fewer than two valid Stage 2 configurations"
        )
    for summary in stage2_ranked[:2]:
        base = summary["configuration"]
        for repeat in (2, 3):
            run(
                make_spec(
                    run_id=(
                        f"stage2_{base['precision']}_r{repeat}"
                    ),
                    stage="stage2_precision",
                    repeat=repeat,
                    processed_dir=processed_dir,
                    physical_gpu_index=arguments.physical_gpu_index,
                    num_workers=best_num_workers,
                    prefetch_factor=2,
                    precision=str(base["precision"]),
                    compile_model=False,
                ),
                "stage2-repeat",
            )
    stage2_best = ranked_valid_groups(
        results, "stage2_precision"
    )[0]
    best_precision = str(stage2_best["configuration"]["precision"])
    print(
        f"Stage 2 best precision={best_precision}",
        flush=True,
    )

    print("\nStage 3: torch.compile", flush=True)
    for compile_model in (False, True):
        run(
            make_spec(
                run_id=(
                    "stage3_compile"
                    if compile_model
                    else "stage3_eager"
                ),
                stage="stage3_compile",
                repeat=1,
                processed_dir=processed_dir,
                physical_gpu_index=arguments.physical_gpu_index,
                num_workers=best_num_workers,
                prefetch_factor=2,
                precision=best_precision,
                compile_model=compile_model,
            ),
            "stage3",
        )
    stage3 = [
        result
        for result in results
        if result.get("stage") == "stage3_compile"
        and result.get("valid")
    ]
    eager = next(
        result
        for result in stage3
        if not result["configuration"]["compile"]
    )
    compiled = next(
        (
            result
            for result in stage3
            if result["configuration"]["compile"]
        ),
        None,
    )
    use_compile = False
    compile_decision = {
        "adopt": False,
        "reason": "compile run was invalid or failed",
    }
    if compiled is not None:
        eager_short_total = (
            float(eager["startup_seconds"])
            + float(eager["warmup_seconds"])
            + float(eager["measured_wall_seconds"])
        )
        compiled_short_total = (
            float(compiled["startup_seconds"])
            + float(compiled["warmup_seconds"])
            + float(compiled["measured_wall_seconds"])
        )
        recompiles = len(compiled.get("stderr_recompile_lines", []))
        use_compile = (
            compiled_short_total < eager_short_total
            and recompiles == 0
        )
        compile_decision = {
            "adopt": use_compile,
            "eager_short_trial_total_seconds": eager_short_total,
            "compiled_short_trial_total_seconds": compiled_short_total,
            "steady_state_speedup": (
                float(eager["measured_wall_seconds"])
                / float(compiled["measured_wall_seconds"])
            ),
            "recompile_warning_lines": recompiles,
            "reason": (
                "compile improves total short-trial time without "
                "recompile warnings"
                if use_compile
                else "compile overhead is not recovered by a short trial "
                "or dynamic-shape recompiles were observed"
            ),
        }
    print(f"torch.compile decision: {compile_decision}", flush=True)

    selected_prefetch = 2
    best_stage_result = (
        compiled if use_compile and compiled is not None else eager
    )
    stage4_ran = (
        best_num_workers > 0
        and float(
            best_stage_result["loader_wait_fraction_of_wall"]
        )
        >= 0.05
    )
    stage4_decision = {
        "ran": stage4_ran,
        "selected_prefetch_factor": 2,
        "reason": (
            "DataLoader wait was material"
            if stage4_ran
            else "DataLoader wait was below 5% or num_workers=0"
        ),
    }
    if stage4_ran:
        print("\nStage 4: prefetch factor", flush=True)
        stage4_results = []
        for prefetch_factor in (2, 4):
            stage4_results.append(
                run(
                    make_spec(
                        run_id=f"stage4_pf{prefetch_factor}",
                        stage="stage4_prefetch",
                        repeat=1,
                        processed_dir=processed_dir,
                        physical_gpu_index=arguments.physical_gpu_index,
                        num_workers=best_num_workers,
                        prefetch_factor=prefetch_factor,
                        precision=best_precision,
                        compile_model=use_compile,
                    ),
                    "stage4",
                )
            )
        valid_stage4 = [
            result for result in stage4_results if result.get("valid")
        ]
        pf2 = next(
            result
            for result in valid_stage4
            if result["configuration"]["prefetch_factor"] == 2
        )
        pf4 = next(
            (
                result
                for result in valid_stage4
                if result["configuration"]["prefetch_factor"] == 4
            ),
            None,
        )
        if (
            pf4 is not None
            and float(pf4["measured_wall_seconds"])
            < 0.98 * float(pf2["measured_wall_seconds"])
        ):
            selected_prefetch = 4
        stage4_decision = {
            "ran": True,
            "selected_prefetch_factor": selected_prefetch,
            "pf2_wall_seconds": float(pf2["measured_wall_seconds"]),
            "pf4_wall_seconds": (
                float(pf4["measured_wall_seconds"])
                if pf4 is not None
                else None
            ),
            "reason": (
                "prefetch 4 improved wall time by more than 2%"
                if selected_prefetch == 4
                else "prefetch 4 improvement was below 2% or invalid"
            ),
        }

    final_spec = make_spec(
        run_id="selected_runtime",
        stage="selected",
        repeat=1,
        processed_dir=processed_dir,
        physical_gpu_index=arguments.physical_gpu_index,
        num_workers=best_num_workers,
        prefetch_factor=selected_prefetch,
        precision=best_precision,
        compile_model=use_compile,
    )
    write_json(
        output_dir / "best_runtime_config.json",
        {
            "selected_configuration": final_spec,
            "compile_decision": compile_decision,
            "prefetch_decision": stage4_decision,
            "selection_priority": [
                "shorter measured wall time",
                "higher valid tokens/s",
                "higher events/s",
                "finite stable loss and gradients",
                "complete non-overlapping worker shard coverage",
                "reproducibility",
                "short-trial total time including compile overhead",
            ],
        },
    )

    print("\nFinal one-epoch verification", flush=True)
    one_epoch_result = run_one_epoch_subprocess(
        final_spec,
        script_path=script_path,
        temporary_dir=temporary_dir,
    )
    write_json(
        output_dir / "one_epoch_verification.json",
        one_epoch_result,
    )
    if not one_epoch_result.get("valid"):
        raise RuntimeError("One-epoch verification failed")
    if not one_epoch_result["validation_auc_within_0p002"]:
        raise RuntimeError(
            "Selected runtime changes full validation AUC by more than 0.002"
        )

    aggregates = aggregate_all(results)
    save_results(output_dir, results, aggregates)
    plot_benchmark(
        aggregates, output_dir / "benchmark_summary.png"
    )
    baseline_key = "nw0_pf2_fp32_eager"
    baseline = result_group_summary(results, baseline_key)
    selected_key = configuration_key(final_spec)
    selected = result_group_summary(results, selected_key)
    if not selected.get("valid"):
        run(final_spec, "selected")
        selected = result_group_summary(results, selected_key)
    final_summary = {
        "benchmark_total_seconds": (
            time.perf_counter() - benchmark_start
        ),
        "physical_gpu": gpu_info,
        "baseline": baseline,
        "selected_runtime_configuration": final_spec,
        "selected_aggregate": selected,
        "speedup_vs_baseline": (
            float(baseline["median_measured_wall_seconds"])
            / float(selected["median_measured_wall_seconds"])
        ),
        "measurement_seconds_saved_vs_baseline": (
            float(baseline["median_measured_wall_seconds"])
            - float(selected["median_measured_wall_seconds"])
        ),
        "compile_decision": compile_decision,
        "prefetch_decision": stage4_decision,
        "one_epoch_verification": one_epoch_result,
        "gpu_processes_after": gpu_processes(),
    }
    write_json(output_dir / "summary.json", final_summary)
    shutil.rmtree(temporary_dir)
    print(
        "\nBenchmark complete: "
        f"speedup={final_summary['speedup_vs_baseline']:.3f} "
        f"selected={configuration_key(final_spec)}",
        flush=True,
    )


def main() -> None:
    arguments = parse_args()
    if arguments.worker_spec:
        if arguments.worker_result is None:
            raise ValueError("--worker-result is required")
        worker_entry(arguments.worker_spec, arguments.worker_result)
        return
    if arguments.one_epoch_spec:
        if arguments.one_epoch_result is None:
            raise ValueError("--one-epoch-result is required")
        one_epoch_entry(
            arguments.one_epoch_spec,
            arguments.one_epoch_result,
        )
        return
    orchestrate(arguments)


if __name__ == "__main__":
    main()
