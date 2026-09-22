"""Train fixed downstream H/Z readouts for SV-weighted representations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core import auc, h_metric_block, jsonable


SEED = 20260916
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4


class FixedMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class WeightedDeepSets(nn.Module):
    """Same phi/rho capacity as the historical DeepSets, with weighted pooling."""
    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 6:
            raise ValueError(f"Expected [batch,draw,6], got {tuple(samples.shape)}")
        if weights.shape != samples.shape[:2]:
            raise ValueError("Sample/weight shape mismatch")
        pooled = torch.sum(self.phi(samples) * weights[..., None], dim=1)
        return self.rho(pooled).squeeze(-1)


def load_npz(path: Path, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        selected = data.files if names is None else names
        return {name: np.asarray(data[name]) for name in selected}


def setup_model(model: nn.Module, device: torch.device, seed: int) -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    return model, optimizer


def predict_fixed(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    with torch.inference_mode():
        for start in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[start : start + BATCH_SIZE].to(device, non_blocking=device.type == "cuda")
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def fit_fixed(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    device: torch.device,
    model: nn.Module,
    seed: int,
) -> tuple[nn.Module, np.ndarray, list[dict[str, float]]]:
    train_x = np.ascontiguousarray(train_x, dtype=np.float32)
    validation_x = np.ascontiguousarray(validation_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    validation_y = np.asarray(validation_y, dtype=np.float32)
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise RuntimeError("Non-finite fixed-readout input")
    model, optimizer = setup_model(model, device, seed)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    history = []
    for epoch in range(EPOCHS):
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train(); loss_sum = 0.0; count = 0
        for values, labels in loader:
            values = values.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            loss = nn.functional.binary_cross_entropy_with_logits(model(values), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite fixed-readout loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels); count += len(labels)
        model.eval(); val_loss_sum = 0.0; val_count = 0
        with torch.inference_mode():
            for start in range(0, len(validation_x), BATCH_SIZE):
                values = torch.from_numpy(validation_x[start : start + BATCH_SIZE]).to(device)
                labels = torch.from_numpy(validation_y[start : start + BATCH_SIZE]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(values), labels)
                val_loss_sum += float(loss.detach().cpu()) * len(labels); val_count += len(labels)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                        "validation_loss": val_loss_sum / val_count})
    return model, predict_fixed(model, validation_x, device), history


def fit_weighted_deepsets(
    train_h: np.ndarray,
    train_w: np.ndarray,
    train_y: np.ndarray,
    validation_h: np.ndarray,
    validation_w: np.ndarray,
    validation_y: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, np.ndarray, list[dict[str, float]], float]:
    train_h = np.ascontiguousarray(train_h.reshape(len(train_h), train_h.shape[1], 6), dtype=np.float32)
    validation_h = np.ascontiguousarray(
        validation_h.reshape(len(validation_h), validation_h.shape[1], 6), dtype=np.float32
    )
    train_w = np.ascontiguousarray(train_w, dtype=np.float32)
    validation_w = np.ascontiguousarray(validation_w, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    validation_y = np.asarray(validation_y, dtype=np.float32)
    if not all(np.isfinite(values).all() for values in (train_h, train_w, validation_h, validation_w)):
        raise RuntimeError("Non-finite weighted DeepSets input")
    model, optimizer = setup_model(WeightedDeepSets(), device, seed)
    dataset = TensorDataset(
        torch.from_numpy(train_h), torch.from_numpy(train_w), torch.from_numpy(train_y)
    )
    history = []
    for epoch in range(EPOCHS):
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train(); loss_sum = 0.0; count = 0
        for samples, weights, labels in loader:
            samples = samples.to(device); weights = weights.to(device); labels = labels.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(samples, weights), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite weighted DeepSets loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels); count += len(labels)
        model.eval(); val_loss_sum = 0.0; val_count = 0
        with torch.inference_mode():
            for start in range(0, len(validation_h), BATCH_SIZE):
                samples = torch.from_numpy(validation_h[start : start + BATCH_SIZE]).to(device)
                weights = torch.from_numpy(validation_w[start : start + BATCH_SIZE]).to(device)
                labels = torch.from_numpy(validation_y[start : start + BATCH_SIZE]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(samples, weights), labels)
                val_loss_sum += float(loss.detach().cpu()) * len(labels); val_count += len(labels)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                        "validation_loss": val_loss_sum / val_count})
    model.eval(); outputs = []
    with torch.inference_mode():
        for start in range(0, len(validation_h), BATCH_SIZE):
            samples = torch.from_numpy(validation_h[start : start + BATCH_SIZE]).to(device)
            weights = torch.from_numpy(validation_w[start : start + BATCH_SIZE]).to(device)
            outputs.append(torch.sigmoid(model(samples, weights)).cpu().numpy())
        count = min(1024, len(validation_h))
        original = torch.sigmoid(model(
            torch.from_numpy(validation_h[:count]).to(device),
            torch.from_numpy(validation_w[:count]).to(device),
        ))
        reversed_score = torch.sigmoid(model(
            torch.from_numpy(validation_h[:count, ::-1].copy()).to(device),
            torch.from_numpy(validation_w[:count, ::-1].copy()).to(device),
        ))
        permutation_delta = float(torch.max(torch.abs(original - reversed_score)).cpu())
    return model, np.concatenate(outputs).astype(np.float64), history, permutation_delta


def save_model(path: Path, model: nn.Module, description: str) -> None:
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "description": description, "seed": SEED, "epochs": EPOCHS,
    }, path)


def full_scores_from_strict(
    score: np.ndarray, strict_rows: np.ndarray, total: int
) -> np.ndarray:
    result = np.full(total, np.nan, dtype=np.float64)
    result[strict_rows] = score
    return result


def load_existing_score(path: Path, ids: np.ndarray, strict_rows: np.ndarray) -> np.ndarray:
    data = load_npz(path)
    position = {int(value): index for index, value in enumerate(data["global_indices"])}
    selected_ids = ids[strict_rows]
    try:
        order = np.asarray([position[int(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"Existing score missing global index {error.args[0]}") from error
    if not np.array_equal(data["global_indices"][order], selected_ids):
        raise RuntimeError("Existing score alignment failed")
    return full_scores_from_strict(data["scores"][order], strict_rows, len(ids))


def auc_block(
    labels: np.ndarray, scores: np.ndarray, event_weights: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = mask & np.isfinite(scores)
    if not selected.any() or len(np.unique(labels[selected])) < 2:
        return {"events": int(selected.sum()), "weighted_auc": None, "unweighted_auc": None}
    return {
        "events": int(selected.sum()),
        "weighted_auc": auc(labels[selected], scores[selected], event_weights[selected]),
        "unweighted_auc": auc(labels[selected], scores[selected], np.ones(int(selected.sum()))),
    }


def prepare_auc(
    labels: np.ndarray, scores: np.ndarray, event_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1]
    return order, labels[order], event_weights[order], distinct


def auc_with_counts(
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], counts: np.ndarray
) -> float:
    order, labels, base_weight, distinct = prepared
    weight = base_weight * counts[order]
    positive = float(weight[labels == 1].sum())
    negative = float(weight[labels == 0].sum())
    if positive <= 0 or negative <= 0:
        return np.nan
    tp = np.cumsum(weight * (labels == 1))
    fp = np.cumsum(weight * (labels == 0))
    tpr = np.r_[0.0, tp[distinct] / positive]
    fpr = np.r_[0.0, fp[distinct] / negative]
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(tpr, fpr))


def paired_auc_bootstrap(
    score_a: np.ndarray,
    score_b: np.ndarray,
    labels: np.ndarray,
    event_weights: np.ndarray,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = mask & np.isfinite(score_a) & np.isfinite(score_b)
    y = labels[selected]; w = event_weights[selected]
    a = score_a[selected]; b = score_b[selected]
    prep_a = prepare_auc(y, a, w); prep_b = prepare_auc(y, b, w)
    rng = np.random.default_rng(seed)
    difference = np.empty(draws, dtype=np.float64)
    probability = np.full(len(y), 1.0 / len(y))
    for index in range(draws):
        counts = rng.multinomial(len(y), probability)
        difference[index] = auc_with_counts(prep_a, counts) - auc_with_counts(prep_b, counts)
    difference = difference[np.isfinite(difference)]
    point = auc(y, a, w) - auc(y, b, w)
    return {
        "events": int(len(y)), "difference": float(point),
        "ci_low": float(np.quantile(difference, 0.025)),
        "ci_high": float(np.quantile(difference, 0.975)),
        "draws": draws, "seed": seed,
    }


def weighted_roc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]; s = scores[order]; w = weights[order]
    positive = w[y == 1].sum(); negative = w[y == 0].sum()
    tp = np.cumsum(w * (y == 1)); fp = np.cumsum(w * (y == 0))
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    return np.r_[0.0, fp[distinct] / negative], np.r_[0.0, tp[distinct] / positive]


def plot_histories(output: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, history in histories.items():
        epoch = [row["epoch"] for row in history]
        axes[0].plot(epoch, [row["train_loss"] for row in history], label=name)
        axes[1].plot(epoch, [row["validation_loss"] for row in history], label=name)
    axes[0].set(xlabel="epoch", ylabel="BCE", title="Fixed-readout training loss")
    axes[1].set(xlabel="epoch", ylabel="BCE", title="Development validation loss (not selected)")
    axes[0].legend(fontsize=6, ncol=2); axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(output / "readout_training.png", dpi=180)
    plt.close(fig)


def plot_spin(
    output: Path,
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
    auc_metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    methods = (
        ("baseline_flow_mean", "baseline flow mean", "0.35"),
        ("sv_weighted_mean", "SV-weighted mean", "#4C78A8"),
        ("sv_weighted_full_posterior", "SV-weighted posterior", "#59A14F"),
        ("truth_nu_functional", "truth-nu functional", "#B279A2"),
        ("exact_h", "exact h", "#E45756"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, cohort in enumerate(("inclusive", "threeprong_x_threeprong")):
        mask = masks[cohort]
        for name, label, color in methods:
            selected = mask & np.isfinite(scores[name])
            fpr, tpr = weighted_roc(labels[selected], scores[name][selected], weights[selected])
            value = auc_metrics[cohort][name]["weighted_auc"]
            axes[0, column].plot(fpr, tpr, lw=2, color=color, label=f"{label} ({value:.4f})")
        axes[0, column].plot([0, 1], [0, 1], color="0.6", ls="--")
        axes[0, column].set(xlabel="weighted false-positive rate", ylabel="weighted true-positive rate",
                            title="Overall" if cohort == "inclusive" else "3p x 3p")
        axes[0, column].legend(fontsize=7)
    for column, cohort in enumerate(("inclusive", "threeprong_x_threeprong")):
        mask = masks[cohort]
        for name, label, color in methods[:4]:
            selected = mask & np.isfinite(scores[name])
            axes[1, column].hist(scores[name][selected & (labels == 0)], bins=np.linspace(0, 1, 51),
                                 density=True, histtype="step", lw=1.5, color=color, ls="--")
            axes[1, column].hist(scores[name][selected & (labels == 1)], bins=np.linspace(0, 1, 51),
                                 density=True, histtype="step", lw=1.5, color=color, label=label)
        axes[1, column].set(xlabel="H/Z readout score", ylabel="density",
                            title=("Overall" if cohort == "inclusive" else "3p x 3p") + ": H solid, Z dashed")
        axes[1, column].legend(fontsize=7)
    fig.suptitle("Fixed-readout spin discrimination", fontsize=14)
    fig.savefig(output / "spin_discrimination.png", dpi=180)
    plt.close(fig)


def plot_summary(
    output: Path,
    h_report: dict[str, Any],
    auc_metrics: dict[str, dict[str, dict[str, Any]]],
    bootstrap: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.arange(3)
    labels = ("baseline flow", "SV weighted", "direction oracle")
    colors = ("0.4", "#4C78A8", "#B279A2")
    for row, cohort in enumerate(("inclusive", "threeprong_x_threeprong")):
        functional = h_report["h_metrics_reco_visible_truth_nu_functional"][cohort]
        mse = [
            functional["uniform"]["mse"],
            functional["sv"]["mse"],
            functional["oracle_direction"]["mse"],
        ]
        cosine = [
            functional["uniform"]["cosine"],
            functional["sv"]["cosine"],
            functional["oracle_direction"]["cosine"],
        ]
        spin = [
            auc_metrics[cohort]["baseline_flow_mean"]["weighted_auc"],
            auc_metrics[cohort]["sv_weighted_mean"]["weighted_auc"],
            auc_metrics[cohort]["oracle_direction_mean"]["weighted_auc"],
        ]
        for column, values in enumerate((mse, cosine, spin)):
            axes[row, column].plot(x, values, color="0.5", lw=1.5, zorder=1)
            axes[row, column].scatter(x, values, s=70, color=colors, zorder=2)
            axes[row, column].set_xticks(x, labels, rotation=18, ha="right")
            for position, value in zip(x, values):
                axes[row, column].annotate(f"{value:.6f}", (position, value), xytext=(0, 7),
                                           textcoords="offset points", ha="center", fontsize=8)
            span = max(values) - min(values)
            margin = max(0.18 * span, 1.0e-4)
            axes[row, column].set_ylim(min(values) - margin, max(values) + margin)
        axes[row, 0].set_ylabel("Overall" if row == 0 else "3p x 3p")
        axes[row, 0].set_title("h MSE (technical closure target)")
        axes[row, 1].set_title("h cosine (technical closure target)")
        axes[row, 2].set_title("weighted H/Z AUC")
        point_auc = auc_metrics[cohort]["point_h"]["weighted_auc"]
        exact_auc = auc_metrics[cohort]["exact_h"]["weighted_auc"]
        axes[row, 2].text(
            0.02, 0.98,
            f"fixed references (off scale)\npoint h {point_auc:.4f}; exact h {exact_auc:.4f}",
            transform=axes[row, 2].transAxes, fontsize=8, va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        ci = bootstrap[cohort]["sv_mean_minus_baseline_mean"]
        axes[row, 2].text(
            0.98, 0.04, f"paired delta AUC = {ci['difference']:+.6f}\n95% CI [{ci['ci_low']:+.6f}, {ci['ci_high']:+.6f}]",
            transform=axes[row, 2].transAxes, fontsize=8, va="bottom", ha="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
    fig.suptitle(
        "Baseline -> SV likelihood -> direction oracle on the same development cohorts",
        fontsize=14,
    )
    fig.savefig(output / "summary_ladder.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--train-posterior", type=Path, required=True)
    parser.add_argument("--validation-posterior", type=Path, required=True)
    parser.add_argument("--train-generated", type=Path, required=True)
    parser.add_argument("--train-representations", type=Path, required=True)
    parser.add_argument("--validation-representations", type=Path, required=True)
    parser.add_argument("--existing-summary", type=Path, required=True)
    parser.add_argument("--point-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True); (args.output / "models").mkdir()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    input_names = ("global_indices", "labels", "weights", "modes", "h")
    train_source = load_npz(args.train_inputs, input_names)
    val_source = load_npz(args.validation_inputs, input_names)
    posterior_names = ("global_indices", "h_truth_reco", "truth_h_reco_valid", "all_draws_valid")
    train_post = load_npz(args.train_posterior, posterior_names)
    val_post = load_npz(args.validation_posterior, posterior_names + ("h_samples",))
    train_cohorts = load_npz(args.train_representations / "cohorts.npz")
    val_cohorts = load_npz(args.validation_representations / "cohorts.npz")
    if not np.array_equal(train_source["global_indices"], train_cohorts["global_indices"]):
        raise RuntimeError("Train representation identity mismatch")
    if not np.array_equal(val_source["global_indices"], val_cohorts["global_indices"]):
        raise RuntimeError("Validation representation identity mismatch")
    train_strict = train_cohorts["strict"].astype(bool)
    val_strict = val_cohorts["strict"].astype(bool)
    train_rows = np.flatnonzero(train_strict); val_rows = np.flatnonzero(val_strict)
    if len(train_rows) < 100_000 or len(val_rows) < 10_000:
        raise RuntimeError("Too few strict events")
    train_labels = train_source["labels"][train_rows]
    val_labels = val_source["labels"][val_rows]

    arm_specs = {
        "sv_weighted_mean": ("mean_sv.npy", 6),
        "sv_raw_mean": ("mean_sv_raw.npy", 6),
        "sv_weighted_moments": ("moments_sv.npy", 15),
        "global_gaussian_mean": ("mean_global_gaussian.npy", 6),
        "matched_shuffle_mean": ("mean_matched_shuffle.npy", 6),
        "oracle_direction_mean": ("mean_oracle_direction.npy", 6),
    }
    scores: dict[str, np.ndarray] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    metadata: dict[str, Any] = {}
    started = time.time()
    for index, (name, (filename, width)) in enumerate(arm_specs.items()):
        train_values = np.load(args.train_representations / filename, mmap_mode="r")[train_rows]
        val_values = np.load(args.validation_representations / filename, mmap_mode="r")[val_rows]
        model, strict_score, history = fit_fixed(
            train_values.reshape(len(train_rows), width), train_labels,
            val_values.reshape(len(val_rows), width), val_labels,
            device, FixedMLP(width), SEED,
        )
        scores[name] = full_scores_from_strict(strict_score, val_rows, len(val_source["labels"]))
        histories[name] = history
        save_model(args.output / "models" / f"{name}.pt", model, name)
        metadata[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                          "train_events": int(len(train_rows)), "validation_events": int(len(val_rows))}
        print(json.dumps({"completed_readout": name, "index": index + 1,
                          "total": len(arm_specs) + 2}), flush=True)

    model, strict_score, history = fit_fixed(
        train_post["h_truth_reco"][train_rows].reshape(len(train_rows), 6), train_labels,
        val_post["h_truth_reco"][val_rows].reshape(len(val_rows), 6), val_labels,
        device, FixedMLP(6), SEED,
    )
    scores["truth_nu_functional"] = full_scores_from_strict(
        strict_score, val_rows, len(val_source["labels"])
    )
    histories["truth_nu_functional"] = history
    save_model(args.output / "models" / "truth_nu_functional.pt", model, "truth_nu_functional")
    metadata["truth_nu_functional"] = {"parameters": sum(p.numel() for p in model.parameters()),
                                        "train_events": int(len(train_rows)),
                                        "validation_events": int(len(val_rows))}
    print(json.dumps({"completed_readout": "truth_nu_functional",
                      "index": len(arm_specs) + 1, "total": len(arm_specs) + 2}), flush=True)

    train_h = np.load(args.train_generated / "train_h_samples.npy", mmap_mode="r")[train_rows]
    train_w = np.load(args.train_representations / "weights_sv.npy", mmap_mode="r")[train_rows]
    val_h = val_post["h_samples"][val_rows]
    val_w = np.load(args.validation_representations / "weights_sv.npy", mmap_mode="r")[val_rows]
    model, strict_score, history, permutation_delta = fit_weighted_deepsets(
        train_h, train_w, train_labels, val_h, val_w, val_labels, device, SEED
    )
    if permutation_delta > 1.0e-6:
        raise RuntimeError(f"Weighted DeepSets permutation failure: {permutation_delta}")
    scores["sv_weighted_full_posterior"] = full_scores_from_strict(
        strict_score, val_rows, len(val_source["labels"])
    )
    histories["sv_weighted_full_posterior"] = history
    save_model(args.output / "models" / "sv_weighted_full_posterior.pt", model,
               "sv_weighted_full_posterior")
    metadata["sv_weighted_full_posterior"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_events": int(len(train_rows)), "validation_events": int(len(val_rows)),
        "train_draws": int(train_h.shape[1]), "validation_draws": int(val_h.shape[1]),
        "permutation_max_abs_delta": permutation_delta,
    }
    print(json.dumps({"completed_readout": "sv_weighted_full_posterior",
                      "index": len(arm_specs) + 2, "total": len(arm_specs) + 2}), flush=True)

    existing_files = {
        "point_h": "full_reco_point_h_validation_scores.npz",
        "baseline_flow_mean": "full_reco_flow_mean_h_validation_scores.npz",
        "baseline_full_posterior": "full_reco_full_h_posterior_validation_scores.npz",
        "exact_h": "exact_truth_h_validation_scores.npz",
    }
    for name, filename in existing_files.items():
        scores[name] = load_existing_score(
            args.existing_summary / filename, val_source["global_indices"], val_rows
        )

    masks = {
        name.removeprefix("mask_"): value.astype(bool)
        for name, value in val_cohorts.items() if name.startswith("mask_")
    }
    labels = val_source["labels"].astype(np.int8)
    event_weights = val_source["weights"].astype(np.float64)
    auc_metrics = {
        cohort: {
            name: auc_block(labels, score, event_weights, mask)
            for name, score in scores.items()
        }
        for cohort, mask in masks.items()
    }
    bootstrap: dict[str, Any] = {}
    for cohort_index, cohort in enumerate(("inclusive", "any_sv", "threeprong_x_threeprong", "exact_surface_invalid")):
        if cohort not in masks or not masks[cohort].any():
            continue
        mask = masks[cohort]
        bootstrap[cohort] = {
            "sv_mean_minus_baseline_mean": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["baseline_flow_mean"], labels,
                event_weights, mask, args.bootstrap_draws, 20260922 + cohort_index * 20,
            ),
            "sv_full_minus_baseline_full": paired_auc_bootstrap(
                scores["sv_weighted_full_posterior"], scores["baseline_full_posterior"], labels,
                event_weights, mask, args.bootstrap_draws, 20260923 + cohort_index * 20,
            ),
            "sv_mean_minus_shuffle_mean": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["matched_shuffle_mean"], labels,
                event_weights, mask, args.bootstrap_draws, 20260924 + cohort_index * 20,
            ),
            "global_gaussian_minus_baseline_mean": paired_auc_bootstrap(
                scores["global_gaussian_mean"], scores["baseline_flow_mean"], labels,
                event_weights, mask, args.bootstrap_draws, 20260925 + cohort_index * 20,
            ),
        }

    h_report = json.loads((args.validation_representations / "report.json").read_text())
    point = load_npz(args.point_predictions)
    point_position = {int(value): index for index, value in enumerate(point["global_indices"])}
    point_order = np.asarray([point_position[int(value)] for value in val_source["global_indices"]])
    point_h = point["h_pred"][point_order]
    h_references = {
        "point_h_vs_canonical": {
            cohort: h_metric_block(point_h, val_source["h"], mask)
            for cohort, mask in masks.items()
        },
        "exact_h_vs_canonical": {
            cohort: h_metric_block(val_source["h"], val_source["h"], mask)
            for cohort, mask in masks.items()
        },
    }
    gap_recovery: dict[str, Any] = {}
    for cohort in masks:
        if cohort not in auc_metrics:
            continue
        functional = h_report["h_metrics_reco_visible_truth_nu_functional"][cohort]
        base_mse = functional["uniform"].get("mse")
        sv_mse = functional["sv"].get("mse")
        base_cos = functional["uniform"].get("cosine")
        sv_cos = functional["sv"].get("cosine")
        base_auc = auc_metrics[cohort]["baseline_flow_mean"]["weighted_auc"]
        sv_auc = auc_metrics[cohort]["sv_weighted_mean"]["weighted_auc"]
        oracle_auc = auc_metrics[cohort]["truth_nu_functional"]["weighted_auc"]
        gap_recovery[cohort] = {
            "h_mse_truth_nu_gap_fraction": (
                (base_mse - sv_mse) / base_mse if base_mse not in (None, 0.0) else None
            ),
            "h_cosine_truth_nu_gap_fraction": (
                (sv_cos - base_cos) / (1.0 - base_cos)
                if base_cos is not None and base_cos != 1.0 else None
            ),
            "auc_truth_nu_functional_gap_fraction": (
                (sv_auc - base_auc) / (oracle_auc - base_auc)
                if None not in (base_auc, sv_auc, oracle_auc) and oracle_auc != base_auc else None
            ),
        }

    for name, score in scores.items():
        np.savez_compressed(
            args.output / f"{name}_validation_scores.npz",
            scores=score[val_rows], labels=labels[val_rows], overlap_weights=event_weights[val_rows],
            modes=val_source["modes"][val_rows], global_indices=val_source["global_indices"][val_rows],
        )
    report = {
        "contract": {
            "readout_recipe": {
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "scheduler": "none", "selection": "fixed endpoint; validation not used for selection",
                "seed": SEED,
            },
            "weighted_deepsets": "historical phi/rho capacity with normalized posterior-weight pooling",
            "test_loaded": False,
        },
        "counts": {"train_strict": int(len(train_rows)), "validation_strict": int(len(val_rows))},
        "metadata": metadata,
        "auc_metrics": auc_metrics,
        "paired_bootstrap_auc": bootstrap,
        "oracle_gap_recovery": gap_recovery,
        "h_references": h_references,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
    (args.output / "readout_histories.json").write_text(
        json.dumps(jsonable(histories), indent=2, sort_keys=True) + "\n"
    )
    plot_histories(args.output, histories)
    plot_spin(args.output, scores, labels, event_weights, masks, auc_metrics)
    plot_summary(args.output, h_report, auc_metrics, bootstrap)
    print(json.dumps({"counts": report["counts"],
                      "overall_auc": auc_metrics["inclusive"],
                      "threeprong_auc": auc_metrics["threeprong_x_threeprong"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
