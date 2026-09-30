"""Mass-width control: validation AUC of the four mass-capable arms on v2, v3 and the mass-rescaled
v2r / v3r samples (same code, seeds and recipe as R20; unit weight, validation 89,500 events).

Error bars: event bootstrap standard deviation (200 resamples) from validation_predictions.npz.
usage: python compare_zmass.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ('truth_nu_transformer', 'truth_transformer', 'reco_transformer', 'reco_baseline')
SRC = {'v2': Path('/home/rbaba/simple-smearing-20260910/training'),
       'v3': Path('/home/rbaba/zspin-v3-20260921/training'),
       'v2r': Path('/home/rbaba/meeting-followup-20260930/zmass/training/v2r'),
       'v3r': Path('/home/rbaba/meeting-followup-20260930/zmass/training/v3r')}


def auc(s, y):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (y == 1).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main(out):
    rng = np.random.default_rng(1)
    res = {}
    for tag, root in SRC.items():
        for arm in ARMS:
            d = root / arm
            if not (d / 'result.json').exists():
                continue
            p = np.load(d / 'validation_predictions.npz')
            keys = p.files
            sk = next(k for k in ('scores', 'score', 'logits', 'probabilities', 'predictions') if k in keys)
            yk = next(k for k in ('labels', 'label', 'y') if k in keys)
            s, y = np.asarray(p[sk], float).ravel(), np.asarray(p[yk]).astype(int).ravel()
            a = auc(s, y)
            bs = [auc(s[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
            ref = json.loads((d / 'result.json').read_text()).get('validation_unweighted_auc')
            res.setdefault(arm, {})[tag] = dict(auc=a, boot_std=float(np.std(bs)), result_json_auc=ref, n=int(len(y)))
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"{'arm':22s}" + ''.join(f'{t:>18s}' for t in SRC))
    for arm, r in res.items():
        print(f'{arm:22s}' + ''.join(f"{r[t]['auc']:10.4f}±{r[t]['boot_std']:.4f}" if t in r else f'{"-":>18s}' for t in SRC))


if __name__ == '__main__':
    main(sys.argv[1])
