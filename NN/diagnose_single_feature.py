from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import LEARNING_RATE, RANDOM_SEED, WEIGHT_DECAY
from diagnose_relative_v3 import (
    LOGGER,
    configure_logging,
    load_json,
    run_command,
    sha256_file,
    summarize,
    train_feature_set,
    write_json,
)
from hpo_utils import configure_tf32
from train import choose_device, set_random_seed


FEATURE_SETS = (
    "absolute-v1",
    "absolute-plus-parent-relative-v3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TauSpin feature set for a long-step diagnostic."
    )
    parser.add_argument("--feature-set", choices=FEATURE_SETS, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def validate_input(
    feature_set: str,
    processed_dir: Path,
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    actual_feature_set = metadata.get("feature_set", "absolute-v1")
    if actual_feature_set != feature_set:
        raise ValueError(
            f"Feature set mismatch: requested={feature_set}, "
            f"metadata={actual_feature_set}"
        )
    for group in ("event", "tau", "track", "pfo"):
        if stats[group]["names"] != metadata["feature_names"][group]:
            raise ValueError(f"{group} stats/metadata feature mismatch")
        mean = np.asarray(stats[group]["mean"], dtype=np.float64)
        std = np.asarray(stats[group]["std"], dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"Non-finite {group} statistics")
        if np.any(std <= 0):
            raise ValueError(f"Non-positive {group} standard deviation")
    selection = metadata.get("event_selection")
    if not selection:
        raise ValueError("The pT-matched selection manifest is missing")
    return {
        "processed_dir": str(processed_dir.resolve()),
        "metadata_sha256": sha256_file(processed_dir / "metadata.json"),
        "stats_sha256": sha256_file(processed_dir / "stats.json"),
        "matching_manifest_sha256": selection["manifest_sha256"],
        "test_split_loaded": False,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if arguments.eval_every_steps <= 0:
        raise ValueError("--eval-every-steps must be positive")
    if arguments.max_steps % arguments.eval_every_steps:
        raise ValueError(
            "--max-steps must be divisible by --eval-every-steps"
        )
    configure_logging(arguments.output_dir)
    processed_dir = arguments.processed_dir.resolve()
    metadata = load_json(processed_dir / "metadata.json")
    stats = load_json(processed_dir / "stats.json")
    input_audit = validate_input(
        arguments.feature_set,
        processed_dir,
        metadata,
        stats,
    )

    set_random_seed(RANDOM_SEED)
    device = choose_device(arguments.device)
    runtime = configure_tf32()
    runtime.update(
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_cuda_devices": torch.cuda.device_count(),
        }
    )
    repository = Path(__file__).resolve().parent.parent
    config = {
        "purpose": (
            "Long-step diagnostic for one feature set on the fixed partial "
            "20 GeV pT-matched sample."
        ),
        "feature_set": arguments.feature_set,
        "max_optimizer_steps": arguments.max_steps,
        "full_validation_every_steps": arguments.eval_every_steps,
        "model_profile": "current",
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "dropout": arguments.dropout,
        "weight_decay": arguments.weight_decay,
        "scheduler": "constant",
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
        "num_workers": arguments.num_workers,
        "prefetch_factor": arguments.prefetch_factor,
        "runtime": runtime,
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"], repository),
            "status_short": run_command(
                ["git", "status", "--short"], repository
            ).splitlines(),
            "diff_stat": run_command(
                ["git", "diff", "--stat"], repository
            ).splitlines(),
        },
        "input_audit": input_audit,
        "test_split_loaded": False,
    }
    write_json(arguments.output_dir / "config.json", config)
    LOGGER.info("Feature set: %s", arguments.feature_set)
    LOGGER.info("GPU: %s", runtime["gpu_name"])

    result = train_feature_set(
        feature_set=arguments.feature_set,
        processed_dir=processed_dir,
        metadata=metadata,
        arguments=arguments,
        device=device,
        output_dir=arguments.output_dir,
        input_audit=input_audit,
    )
    summary = {
        "feature_set": arguments.feature_set,
        "series": summarize(
            result["validation_history"], arguments.max_steps
        ),
        "training_run": result["metrics"],
        "test_split_loaded": False,
    }
    write_json(arguments.output_dir / "summary.json", summary)
    LOGGER.info(
        "Finished feature_set=%s best_auc=%.6f step=%d",
        arguments.feature_set,
        summary["series"]["best_auc"],
        summary["series"]["best_auc_step"],
    )


if __name__ == "__main__":
    main()
