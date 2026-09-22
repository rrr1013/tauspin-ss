"""Direct H/Z classifier trained on hard labels or on the analytic spin weight.

Soft target ("mining gold", Brehmer et al. PNAS 2020): the truth-level spin
density gives, per event, s(z) = w_H/(w_H+w_Z) with w_X = 1 + h-^T C_X h+,
C_H = diag(1,1,-1), C_Z = diag(0,0,1) in the canonical (n,r,k) basis.  Since
E[s(z)|x] is the spin-only Bayes posterior, regressing it with BCE removes the
Bernoulli label noise while keeping the same minimiser under equal production.
Inputs, encoder, sampler and schedule are those of p4 (local-frame geometry).

Base: Point-h regression with local-frame geometry channels.

Recipe, architecture, sampler, seed, rolling-3 checkpoint selection and the
50-epoch schedule are those of the 2026-09-20 IP/SV ablation
(``NN/ip_sv_ablation/train_geometry.py``).  The only change is the content of
tau-token slots 6..9: baseline zeroes them, every other arm fills them with a
4-vector from ``p3_geometry_features.py`` expressed in the reco (n, r, k) basis.
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

CODE = Path("/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import reco_truth_common_v2 as c
from models import RecoModel
from reco_data import RecoHDataset, collate_reco_h
from train_geometry import evaluate
from train_reco_truth_common_v2 import configure_runtime, state_dict_on_cpu

ARMS = ("baseline", "ip_pv", "ip_legacy", "ip_shuffle", "oracle")
CH = torch.tensor([1.0, 1.0, -1.0])
CZ = torch.tensor([0.0, 0.0, 1.0])


def soft_target(h):
    prod = h[:, 0] * h[:, 1]
    wh = 1 + (prod * CH.to(h)).sum(-1)
    wz = 1 + (prod * CZ.to(h)).sum(-1)
    return wh / (wh + wz)


def target_of(batch, kind):
    return batch["labels"].float() if kind == "hard" else soft_target(batch["h"])


def evaluate_cls(model, loader, device, kind):
    model.eval()
    out = {k: [] for k in ("scores", "labels", "overlap_weights", "global_indices", "truth_modes")}
    tot, n = 0.0, 0
    with torch.inference_mode():
        for cpu_batch in loader:
            b = c.move_batch(cpu_batch, device)
            logit, _ = model(b)
            loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), target_of(b, kind))
            tot += float(loss) * len(logit)
            n += len(logit)
            out["scores"].append(torch.sigmoid(logit.float()).cpu().numpy())
            for k in ("labels", "overlap_weights", "global_indices", "truth_modes"):
                out[k].append(b[k].cpu().numpy())
    return {"validation_loss": tot / n}, {k: np.concatenate(v) for k, v in out.items()}


class GeoDataset(RecoHDataset):
    def __init__(self, split, features, arm):
        super().__init__(split)
        ids = self.global_index.numpy()
        src = features[f"{split}_global_indices"]
        pos = {int(v): i for i, v in enumerate(src)}
        rows = np.asarray([pos[int(v)] for v in ids])
        key = arm if arm != "baseline" else "ip_pv"
        geo = np.asarray(features[f"{split}_{key}"])[rows]
        if arm == "baseline":
            geo = np.zeros_like(geo)
        self.geo = torch.as_tensor(geo, dtype=torch.float32)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["geo"] = self.geo[index]
        return item


def collate(events):
    batch = collate_reco_h(events)
    batch["geo"] = torch.stack([e["geo"] for e in events])
    return batch


class LocalGeoModel(RecoModel):
    def __init__(self):
        super().__init__("baseline")

    def forward(self, batch):
        tau = batch["tau_features"]
        tau_tokens = torch.cat((tau[:, 1:3, :6], batch["geo"].to(tau)), dim=-1)
        batch = dict(batch)
        batch["tau_features"] = torch.cat((tau[:, :1], tau_tokens, tau[:, 3:]), dim=1)
        # baseline arm mask would zero slots 6..9; apply only the track mask.
        masked = {k: batch[k] for k in ("object_type", "tau_side", "decay_mode", "event_features",
                                         "tau_features", "track_features", "pfo_features", "padding_mask")}
        from models import ARM_MASKS
        track_mask = torch.as_tensor(ARM_MASKS["baseline"]["track_features"],
                                     device=tau.device, dtype=tau.dtype)
        masked["track_features"] = masked["track_features"] * track_mask
        z = super(RecoModel, self).forward(masked)
        return self.hz_head(z).squeeze(-1), self.h_head(z).reshape(-1, 2, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--target", choices=("hard", "soft"), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    c.set_random_seed(c.SEED)
    device = torch.device("cuda")
    runtime = configure_runtime(device)
    features = dict(np.load(args.features))
    train = GeoDataset("train", features, args.arm)
    validation = GeoDataset("validation", features, args.arm)
    stream = c.BalancedCommonStreamDataset(train, shuffle=True, balanced=True, seed=c.SEED)
    vloader = DataLoader(validation, batch_size=c.BATCH_SIZE, num_workers=args.workers,
                         collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    model = LocalGeoModel().to(device)
    network = torch.compile(model, dynamic=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.LEARNING_RATE,
                                  weight_decay=c.WEIGHT_DECAY, fused=True)
    steps_per_epoch = stream.steps_per_epoch(c.BATCH_SIZE, max(1, args.workers))
    total_steps = steps_per_epoch * args.epochs
    (args.output_dir / "config.json").write_text(json.dumps({
        "arm": args.arm, "target": args.target, "epochs": args.epochs, "seed": c.SEED, "features": str(args.features),
        "train_events": len(train), "validation_events": len(validation),
        "tau_slots_6_9": "local-frame geometry 4-vector (zeros for baseline)",
        "track_mask": "baseline (no scalar or legacy directional IP)",
        "runtime": runtime, "test_loaded": False}, indent=2))
    history, best, previous_state, selected_state, selection_epoch = [], math.inf, None, None, None
    step, started = 0, time.time()
    for epoch in range(args.epochs):
        stream.set_epoch(epoch)
        loader = DataLoader(stream, batch_size=c.BATCH_SIZE, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True)
        network.train()
        tot_n, tot_l = 0, 0.0
        for cpu_batch in loader:
            batch = c.move_batch(cpu_batch, device)
            step += 1
            for g in optimizer.param_groups:
                g["lr"] = c.learning_rate_for_step(step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            logit, h_pred = network(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), target_of(batch, args.target))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            tot_n += len(h_pred)
            tot_l += float(loss.detach()) * len(h_pred)
            if args.pilot and step >= 3:
                break
        if args.pilot:
            print(json.dumps({"pilot_passed": True, "loss": tot_l / tot_n}), flush=True)
            return
        metrics, _ = evaluate_cls(network, vloader, device, args.target)
        row = {"epoch": epoch + 1, "train_loss": tot_l / tot_n, **metrics}
        history.append(row)
        if len(history) >= 3:
            value = float(np.mean([r["validation_loss"] for r in history[-3:]]))
            row["rolling_selection_value"] = value
            if value < best:
                best, selected_state, selection_epoch = value, previous_state, epoch
        previous_state = state_dict_on_cpu(model)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    model.load_state_dict(selected_state)
    metrics, predictions = evaluate_cls(network, vloader, device, args.target)
    torch.save({"model_state_dict": selected_state, "epoch": selection_epoch, "arm": args.arm},
               args.output_dir / "checkpoint.pt")
    np.savez_compressed(args.output_dir / "validation_predictions.npz", **predictions)
    (args.output_dir / "result.json").write_text(json.dumps({
        "arm": args.arm, "selected_epoch": selection_epoch, "status": "complete",
        "hostname": platform.node(), "elapsed_seconds": time.time() - started,
        **metrics, "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
