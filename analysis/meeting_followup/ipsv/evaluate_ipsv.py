"""Fixed readout + comparison for the simple-vs-ATLAS IP/SV arms.

readout: the unchanged fixed_readout_parallel.py recipe (6 -> 64 -> 64 -> 1 MLP on the train-set
predicted h, AdamW 1e-3, 50 epochs, batch 512, seed 20260916), applied to the validation
predicted h.  Labels / identities come from the prediction files of each run.

metrics per run: validation h MSE, per-component correlation of h_pred with the target (n, r, k,
both taus), H/Z AUC unit weight (ATLAS also parent-overlap weighted), truth mode-pair AUCs.
comparison: each arm minus 'none' of the same sample and seed, paired event bootstrap (95% CI);
seed-mean of the arms that have seeds 42 and 43.

usage: python evaluate_ipsv.py RUNS_DIR OUT.json [--parity PRED_DIR SCORES_NPZ]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PAIRS = {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1),
         'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}


class FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)


def readout(train_x, train_y, val_x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    model = FixedMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4, fused=device.type == 'cuda')
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    for epoch in range(50):
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(20260916 + epoch))
        model.train()
        for v, y in loader:
            v, y = v.to(device), y.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(v), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    out = []
    with torch.inference_mode():
        vals = torch.from_numpy(val_x)
        for s in range(0, len(vals), 512):
            out.append(torch.sigmoid(model(vals[s:s + 512].to(device))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def wauc(score, y, w):
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], y[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def load(run):
    tr = np.load(run / 'train_predictions.npz')
    va = np.load(run / 'validation_predictions.npz')
    return tr, va


def pair_mask(modes, a, b):
    lo, hi = np.minimum(modes[:, 0], modes[:, 1]), np.maximum(modes[:, 0], modes[:, 1])
    return (lo == a) & (hi == b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--parity', nargs=2, type=Path)
    ap.add_argument('--boot', type=int, default=500)
    a = ap.parse_args()
    if a.parity:
        tr, va = load(a.parity[0])
        s = readout(tr['h_pred'].reshape(-1, 6).astype(np.float32), tr['labels'].astype(np.float32),
                    va['h_pred'].reshape(-1, 6).astype(np.float32))
        ref = np.load(a.parity[1])
        if not np.array_equal(ref['global_indices'], va['global_indices']):
            raise RuntimeError('parity identity mismatch')
        print(json.dumps({'parity_max_abs_score_diff': float(np.abs(s - ref['scores']).max()),
                          'auc_new': wauc(s, va['labels'], np.ones(len(s))),
                          'auc_ref': wauc(ref['scores'], va['labels'], np.ones(len(s)))}))
        return
    res, scores, ref_ids = {}, {}, {}
    for run in sorted(p for p in a.runs.iterdir() if (p / 'result.json').exists()):
        name = run.name
        sample = name.split('_')[0]
        tr, va = load(run)
        cache = run / 'readout_scores.npz'
        if cache.exists():
            s = np.load(cache)['scores']
        else:
            s = readout(tr['h_pred'].reshape(-1, 6).astype(np.float32), tr['labels'].astype(np.float32),
                        va['h_pred'].reshape(-1, 6).astype(np.float32))
            np.savez_compressed(cache, scores=s, global_indices=va['global_indices'])
        ids = va['global_indices']
        if sample in ref_ids and not np.array_equal(ref_ids[sample], ids):
            raise RuntimeError(f'{name}: validation identity differs from the other {sample} runs')
        ref_ids[sample] = ids
        y, w, m = va['labels'].astype(int), va['overlap_weights'].astype(np.float64), va['truth_modes']
        h, hp = va['h'].astype(np.float64), va['h_pred'].astype(np.float64)
        corr = [float(np.corrcoef(hp[:, :, k].ravel(), h[:, :, k].ravel())[0, 1]) for k in range(3)]
        r = dict(sample=sample, n=int(len(y)), val_h_mse=float(np.mean((hp - h) ** 2)),
                 corr_nrk=corr, auc_unit=wauc(s, y, np.ones(len(y))),
                 selected_epoch=json.loads((run / 'result.json').read_text())['selected_epoch'])
        if sample == 'atlas':
            r['auc_weighted'] = wauc(s, y, w)
        r['pairs_unit'] = {p: wauc(s[pair_mask(m, *ab)], y[pair_mask(m, *ab)], np.ones(int(pair_mask(m, *ab).sum())))
                           for p, ab in PAIRS.items()}
        res[name] = r
        scores[name] = (s, y, w, m)
    rng = np.random.default_rng(20260930)
    boot_idx = {smp: [rng.integers(0, len(ids), len(ids)) for _ in range(a.boot)] for smp, ids in ref_ids.items()}
    comp = {}
    for name, r in res.items():
        smp, key_seed = r['sample'], name.split('_s')[-1]
        base = f'{smp}_none_s{key_seed}'
        if name == base or base not in scores:
            continue
        s1, y, w, m = scores[name]
        s0 = scores[base][0]
        d = r['auc_unit'] - res[base]['auc_unit']
        bs = [wauc(s1[i], y[i], np.ones(len(i))) - wauc(s0[i], y[i], np.ones(len(i))) for i in boot_idx[smp]]
        comp[name] = dict(vs=base, delta_auc_unit=d, ci95=np.quantile(bs, [0.025, 0.975]).tolist(),
                          delta_pairs_unit={p: r['pairs_unit'][p] - res[base]['pairs_unit'][p] for p in PAIRS},
                          delta_mse=r['val_h_mse'] - res[base]['val_h_mse'])
    # seed means for arms with both seeds (h-averaged readout is not used; mean of AUCs)
    seedmean = {}
    for name in res:
        stem = name.rsplit('_s', 1)[0]
        seeds = [n for n in res if n.rsplit('_s', 1)[0] == stem]
        if len(seeds) > 1:
            seedmean[stem] = dict(seeds=seeds, auc_unit=float(np.mean([res[n]['auc_unit'] for n in seeds])),
                                  spread=float(np.ptp([res[n]['auc_unit'] for n in seeds])))
    a.out.write_text(json.dumps(dict(runs=res, vs_none=comp, seed_mean=seedmean), indent=1))
    for name, r in res.items():
        c = comp.get(name)
        print(f"{name:28s} mse {r['val_h_mse']:.4f} corr {np.round(r['corr_nrk'], 3)} AUC {r['auc_unit']:.4f}"
              + (f" w {r['auc_weighted']:.4f}" if 'auc_weighted' in r else '')
              + (f"  d {c['delta_auc_unit']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]" if c else ''))


if __name__ == '__main__':
    main()
