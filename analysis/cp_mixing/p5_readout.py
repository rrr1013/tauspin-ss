"""Best CP readout of the existing point-h output, fitted on the train split.

The optimal reco-level statistic for phi_tau is the conditional score
E[S | reco].  Given only the network's six h_pred numbers, that conditional
mean is approximated by an ordinary least-squares fit of S on polynomial
features of h_pred, fitted on the H rows of the *train* split and applied
unchanged to the validation split.  No network is retrained and the
validation score is never used for fitting.
"""
import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import cramer_rao, score, sensitivity, triple  # noqa: E402


def poly_features(h, degree):
    x = h.reshape(len(h), 6).astype(np.float64)
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(6), d):
            cols.append(np.prod(x[:, idx], axis=1))
    return np.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'results')
    ap.add_argument('--n-ref', type=int, default=10000)
    ap.add_argument('--degrees', type=int, nargs='+', default=[1, 2, 3, 4])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    C0 = c_matrix(0.0)
    out = {'n_ref': args.n_ref, 'arms': {}}
    for arm in ('base_s43', 'full22_s42'):
        TR = np.load(args.data / f'gen3pi_{arm}_train.npz')
        VA = np.load(args.data / f'gen3pi_{arm}.npz')
        row = {}
        for tag, F in (('train', TR), ('val', VA)):
            k = F['labels'] > 0.5
            h, hp = F['h'][k].astype(float), F['h_pred'][k].astype(float)
            S, _ = score(h, C0)
            row[tag] = {'n': int(k.sum()), 'S': S, 'hp': hp, 'h': h}
        res = {'n_train_H': row['train']['n'], 'n_val_H': row['val']['n'],
               'plugin_Tp': sensitivity(triple(row['val']['hp'][:, 0], row['val']['hp'][:, 1]),
                                        row['val']['S'], args.n_ref, boot=200),
               'exact_Tp': sensitivity(triple(row['val']['h'][:, 0], row['val']['h'][:, 1]),
                                       row['val']['S'], args.n_ref, boot=200),
               'exact_optimal': cramer_rao(row['val']['S'], args.n_ref)}
        for deg in args.degrees:
            Xtr = poly_features(row['train']['hp'], deg)
            beta, *_ = np.linalg.lstsq(Xtr, row['train']['S'], rcond=None)
            pred_tr = Xtr @ beta
            pred_va = poly_features(row['val']['hp'], deg) @ beta
            res[f'fitted_readout_deg{deg}'] = {
                'n_features': int(Xtr.shape[1]),
                'train_corr': float(np.corrcoef(pred_tr, row['train']['S'])[0, 1]),
                'val_corr': float(np.corrcoef(pred_va, row['val']['S'])[0, 1]),
                **sensitivity(pred_va, row['val']['S'], args.n_ref, boot=200)}
            if deg == 3:
                np.savez_compressed(args.output / f'readout_{arm}.npz',
                                    pred=pred_va, score=row['val']['S'],
                                    plugin=triple(row['val']['hp'][:, 0], row['val']['hp'][:, 1]))
        out['arms'][arm] = res

    (args.output / 'readout.json').write_text(json.dumps(out, indent=1))
    print(f'--- sigma(phi_tau) [deg] per {args.n_ref} H events ---')
    for arm, r in out['arms'].items():
        print(f'{arm}  train H {r["n_train_H"]}  val H {r["n_val_H"]}')
        print(f'  plug-in triple product      {r["plugin_Tp"]["sigma_deg"]:7.3f} +- {r["plugin_Tp"]["sigma_deg_se"]:.3f}')
        for deg in args.degrees:
            v = r[f'fitted_readout_deg{deg}']
            print(f'  fitted readout, degree {deg}    {v["sigma_deg"]:7.3f} +- {v["sigma_deg_se"]:.3f}'
                  f'   ({v["n_features"]} features, corr train {v["train_corr"]:.4f} / val {v["val_corr"]:.4f})')
        print(f'  exact h triple product      {r["exact_Tp"]["sigma_deg"]:7.3f}')
        print(f'  exact h optimal             {r["exact_optimal"]["sigma_deg"]:7.3f}')


if __name__ == '__main__':
    main()
