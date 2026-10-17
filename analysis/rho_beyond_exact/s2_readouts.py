"""H/Z AUC of the reco-visible + truth-nu polarimeter under two readouts, against the main-line learner.

Readouts
  formula   log(1 + h-^T C_H h+) - log(1 + h-^T C_Z h+)  (the exact LR for exact h; the 10/17 reference row)
  mlp       the unchanged fixed readout recipe of ip_sv_ablation/fixed_readout_parallel.py (6 -> 64 -> 64 -> 1,
            AdamW 1e-3, wd 1e-4, batch 512, 50 epochs, seed 20260916, unweighted BCE), trained on the TRAIN
            h of the same representation and applied to VALIDATION.  Invalid sides -> 0.

Rows: exact h (gen3pi targets), reco visible + truth nu ('reco' of s1), the s1 replacement ladder (formula
only; their readout would be trained on a different representation), and the learner (gen3pi p11
full22 s42/s43, base s43): existing fixed-readout scores and the formula applied to h_pred.
AUC: parent-overlap weights, by truth mode pair; deltas by paired event bootstrap.  Test never loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
PAIRS = {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1),
         'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}
CH, CZ = np.array([1, 1, -1.0]), np.array([0, 0, 1.0])
LADDER_ROWS = ('reco', 'pi0dir', 'pi0E', 'pi0', 'pich', 'both')


def wauc(score, y, w):
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], y[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def formula(h):
    h = np.nan_to_num(h, nan=0.0)
    p = h[:, 0] * h[:, 1]
    return np.log(np.maximum(1 + p @ CH, 1e-12)) - np.log(np.maximum(1 + p @ CZ, 1e-12))


class FixedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, v):
        return self.network(v).squeeze(-1)


def mlp_readout(x_tr, y_tr, x_va, device):
    x_tr = np.nan_to_num(x_tr, nan=0.0).reshape(len(x_tr), 6).astype(np.float32)
    x_va = np.nan_to_num(x_va, nan=0.0).reshape(len(x_va), 6).astype(np.float32)
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    model = FixedMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32)))
    for epoch in range(50):
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(20260916 + epoch))
        model.train()
        for v, l in loader:
            v, l = v.to(device), l.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(v), l)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(x_va).to(device)).cpu().numpy().astype(np.float64)


def pair_masks(modes):
    sel = {k: ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))
           for k, (a, b) in PAIRS.items()}
    sel['overall'] = np.ones(len(modes), bool)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=400)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tt = np.load(GEN / 'targets' / 'train.npz')
    vt = np.load(GEN / 'targets' / 'validation.npz')
    tv, vv = tt['h_valid'].astype(bool), vt['h_valid'].astype(bool)
    tr_ids, va_ids = tt['global_indices'][tv], vt['global_indices'][vv]
    y_tr, y_va = tt['labels'][tv].astype(int), vt['labels'][vv].astype(int)
    w_va = vt['weights'][vv].astype(float)
    hx_tr, hx_va = tt['h'][tv].astype(float), vt['h'][vv].astype(float)
    modes = vt['modes'][vv]

    Ltr = np.load(a.ladder_dir / 'ladder_train.npz')
    Lva = np.load(a.ladder_dir / 'ladder_validation.npz')
    pos = {int(g): i for i, g in enumerate(Ltr['global_indices'])}
    ref_tr = Ltr['h_reco'][[pos[int(g)] for g in tr_ids]]
    assert np.array_equal(Lva['global_indices'], va_ids)

    scores = {}
    scores['exact formula'] = formula(hx_va)
    scores['exact mlp'] = mlp_readout(hx_tr, y_tr, hx_va, device)
    scores['reco+truth nu formula'] = formula(Lva['h_reco'])
    scores['reco+truth nu mlp'] = mlp_readout(ref_tr, y_tr, Lva['h_reco'], device)
    for name in LADDER_ROWS[1:]:
        scores[f'ladder {name} formula'] = formula(Lva[f'h_{name}'])
    for arm in ('full22_s42', 'full22_s43', 'base_s43'):
        d = np.load(GEN / 'p11' / arm / 'validation_predictions.npz')
        assert np.array_equal(d['global_indices'], va_ids)
        scores[f'learner {arm} formula'] = formula(d['h_pred'].astype(float))
        r = np.load(GEN / 'p11_readout' / f'readout_scores_{arm}.npz')
        assert np.array_equal(r['global_indices'], va_ids)
        scores[f'learner {arm} readout'] = r['scores'].astype(float)
    r = np.load(GEN / 'p11_readout' / 'readout_scores_ens_full22.npz')
    assert np.array_equal(r['global_indices'], va_ids)
    scores['learner ens_full22 readout'] = r['scores'].astype(float)
    np.savez(a.output.with_suffix('.scores.npz'), global_indices=va_ids, labels=y_va, weights=w_va, modes=modes,
             names=np.array(list(scores)), values=np.stack(list(scores.values())))

    masks = pair_masks(modes)
    res = {'auc': {}, 'delta': {}}
    for name, s in scores.items():
        res['auc'][name] = {k: wauc(s[m], y_va[m], w_va[m]) for k, m in masks.items()}
    comps = [('learner ens_full22 readout', 'reco+truth nu formula'),
             ('learner ens_full22 readout', 'reco+truth nu mlp'),
             ('reco+truth nu mlp', 'reco+truth nu formula'),
             ('learner full22_s42 readout', 'learner full22_s42 formula'),
             ('exact mlp', 'exact formula'),
             ('ladder pi0 formula', 'reco+truth nu formula'),
             ('ladder pi0dir formula', 'reco+truth nu formula'),
             ('ladder pi0E formula', 'reco+truth nu formula'),
             ('ladder pich formula', 'reco+truth nu formula')]
    rng = np.random.default_rng(20261018)
    for key in ('rho x rho', 'pi x rho', 'rho x 3pi', 'overall'):
        m = np.flatnonzero(masks[key])
        boots = {c: [] for c in comps}
        for _ in range(a.boot):
            b = m[rng.integers(0, len(m), len(m))]
            for c in comps:
                boots[c].append(wauc(scores[c[0]][b], y_va[b], w_va[b]) - wauc(scores[c[1]][b], y_va[b], w_va[b]))
        res['delta'][key] = {f'{c[0]} - {c[1]}': [res['auc'][c[0]][key] - res['auc'][c[1]][key],
                                                  float(np.quantile(boots[c], 0.025)),
                                                  float(np.quantile(boots[c], 0.975))] for c in comps}
    a.output.write_text(json.dumps(res, indent=1))
    keys = ('pi x pi', 'pi x rho', 'rho x rho', 'rho x 3pi', '3pi x 3pi', 'overall')
    print(f"{'':36s}" + ''.join(f'{k:>11s}' for k in keys))
    for name in scores:
        print(f'{name:36s}' + ''.join(f"{res['auc'][name][k]:11.4f}" for k in keys))
    for key, d in res['delta'].items():
        print(key)
        for c, (v, lo, hi) in d.items():
            print(f'   {c:62s} {v:+.4f} [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    main()
