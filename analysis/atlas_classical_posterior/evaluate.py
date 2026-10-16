"""H/Z AUCs of the classical posterior vs the main-line learner, by mode pair, and their combination.

Same validation cohort (59,390), parent-overlap weights and weighted AUC as
gen3pi_teacher/compare_point_h.py.  Learner rows are the existing fixed-readout
scores of the gen3pi-teacher point-h arms (base = no IP/SV, full22 = +IP/SV).
Deltas: paired event bootstrap.  Combination: cross-fit on validation (fold =
event_number parity), logistic regression on [learner score, classical LR] and a
gradient-boosted classifier on the per-event moments; nothing is fitted on the
fold it scores.  Test is never loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
PAIRS = {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1),
         'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}
CH, CZ = np.array([1, 1, -1.0]), np.array([0, 0, 1.0])
MEASURES = ('met', 'ip3', 'sv', 'ip3_sv', 'ip3_sv_mass', 'ip3_shuffle')


def wauc(score, y, w):
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], y[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def lr_of(h):
    p = h[:, 0] * h[:, 1]
    return np.log(1 + p @ CH) - np.log(1 + p @ CZ)


def pair_masks(modes):
    sel = {k: ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))
           for k, (a, b) in PAIRS.items()}
    sel['overall'] = np.ones(len(modes), bool)
    return sel


def crossfit(X, y, w, fold, kind):
    out = np.zeros(len(y))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if kind == 'logistic':
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            m = LogisticRegression(C=1e3, max_iter=2000).fit((X[tr] - mu) / sd, y[tr], sample_weight=w[tr])
            out[te] = m.decision_function((X[te] - mu) / sd)
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, random_state=0)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posterior', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=500)
    a = ap.parse_args()
    P = np.load(a.posterior)
    ids, y, w = P['global_indices'], P['labels'].astype(int), P['weights'].astype(float)
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    assert np.array_equal(t['global_indices'][v], ids)
    h_gen = t['h'][v].astype(float)
    modes = P['modes']
    rmode = P['reco_h_mode'].astype(int)
    avail = P['reco_h_available_side'].all(1)

    scores = {'exact h (generator current)': lr_of(h_gen),
              'reco visible + truth nu': P['truthnu_lr']}
    for m in MEASURES:
        scores[f'classical:{m}'] = P[f'{m}_lr']
    learner = {}
    for arm in ('base_s43', 'full22_s42', 'full22_s43', 'ens_full22', 'idealip22_s42'):
        d = np.load(GEN / 'p11_readout' / f'readout_scores_{arm}.npz')
        assert np.array_equal(d['global_indices'], ids)
        scores[f'learner:{arm}'] = d['scores']
        learner[arm] = d['scores']
    hp = {}
    for arm in ('base_s43', 'full22_s42', 'full22_s43'):
        d = np.load(GEN / 'p11' / arm / 'validation_predictions.npz')
        assert np.array_equal(d['global_indices'], ids)
        hp[arm] = d['h_pred'].reshape(len(ids), 6).astype(float)

    fold = (P['event_numbers'] % 2).astype(int)
    feat = lambda m: np.concatenate([P[f'{m}_Eh'].reshape(len(ids), 6), P[f'{m}_Ehh'].reshape(len(ids), 9),
                                     P[f'{m}_lr'][:, None]], 1)
    onehot = np.stack([rmode[:, s] == k for s in (0, 1) for k in (0, 1, 3)], 1).astype(float)
    L = learner['ens_full22'][:, None]
    combos = {
        'combo-logit: learner + classical ip3_sv': np.c_[L, P['ip3_sv_lr']],
        'combo-logit: learner + classical ip3_sv_mass': np.c_[L, P['ip3_sv_mass_lr']],
        'combo-logit: learner + classical met (no IP/SV)': np.c_[L, P['met_lr']],
        'combo-logit: learner + classical ip3_shuffle': np.c_[L, P['ip3_shuffle_lr']],
        'combo-logit: learner + learner (self, control)': np.c_[L, learner['full22_s42']],
    }
    for k, X in combos.items():
        scores[k] = crossfit(X, y, w, fold, 'logistic')
    gbt = {
        'combo-gbt: learner h_pred only': np.c_[hp['full22_s42'], hp['full22_s43'], onehot],
        'combo-gbt: classical ip3_sv moments only': np.c_[feat('ip3_sv'), onehot],
        'combo-gbt: learner h_pred + classical ip3_sv': np.c_[hp['full22_s42'], hp['full22_s43'], feat('ip3_sv'), onehot],
        'combo-gbt: learner h_pred + classical ip3_shuffle': np.c_[hp['full22_s42'], hp['full22_s43'], feat('ip3_shuffle'), onehot],
        'combo-gbt: learner h_pred + classical met': np.c_[hp['full22_s42'], hp['full22_s43'], feat('met'), onehot],
    }
    for k, X in gbt.items():
        scores[k] = crossfit(X, y, w, fold, 'gbt')

    truth_sel, reco_sel = pair_masks(modes), pair_masks(np.where(avail[:, None], rmode, -1))
    res = {'events': int(len(y)), 'reco_available_both_fraction': float(avail.mean()),
           'auc_truth_pairs': {}, 'auc_reco_pairs': {}}
    for k, s in scores.items():
        res['auc_truth_pairs'][k] = {p: wauc(s[m], y[m], w[m]) for p, m in truth_sel.items()}
        res['auc_reco_pairs'][k] = {p: wauc(s[m], y[m], w[m]) for p, m in reco_sel.items() if m.sum() > 100}
    res['events_truth_pairs'] = {p: int(m.sum()) for p, m in truth_sel.items()}
    res['events_reco_pairs'] = {p: int(m.sum()) for p, m in reco_sel.items()}

    rng = np.random.default_rng(1)
    deltas = {
        'classical ip3_sv - learner ens_full22': ('classical:ip3_sv', 'learner:ens_full22'),
        'classical ip3_sv_mass - learner ens_full22': ('classical:ip3_sv_mass', 'learner:ens_full22'),
        'classical ip3_sv - classical met': ('classical:ip3_sv', 'classical:met'),
        'classical sv - classical met': ('classical:sv', 'classical:met'),
        'classical ip3 - classical met': ('classical:ip3', 'classical:met'),
        'classical ip3_shuffle - classical met': ('classical:ip3_shuffle', 'classical:met'),
        'learner full22 ens - learner base_s43': ('learner:ens_full22', 'learner:base_s43'),
        'logit learner+ip3_sv - learner': ('combo-logit: learner + classical ip3_sv', 'learner:ens_full22'),
        'logit learner+ip3_sv_mass - learner': ('combo-logit: learner + classical ip3_sv_mass', 'learner:ens_full22'),
        'logit learner+met - learner': ('combo-logit: learner + classical met (no IP/SV)', 'learner:ens_full22'),
        'logit learner+shuffle - learner': ('combo-logit: learner + classical ip3_shuffle', 'learner:ens_full22'),
        'gbt h_pred+ip3_sv - gbt h_pred': ('combo-gbt: learner h_pred + classical ip3_sv', 'combo-gbt: learner h_pred only'),
        'gbt h_pred+shuffle - gbt h_pred': ('combo-gbt: learner h_pred + classical ip3_shuffle', 'combo-gbt: learner h_pred only'),
        'gbt h_pred+met - gbt h_pred': ('combo-gbt: learner h_pred + classical met', 'combo-gbt: learner h_pred only'),
    }
    res['paired_delta'] = {}
    for name, (p, q) in deltas.items():
        out = {}
        for pair, m in truth_sel.items():
            idx = np.flatnonzero(m)
            d0 = wauc(scores[p][idx], y[idx], w[idx]) - wauc(scores[q][idx], y[idx], w[idx])
            bs = []
            nb = a.boot if pair == 'overall' else a.boot // 2
            for _ in range(nb):
                b = idx[rng.integers(0, len(idx), len(idx))]
                bs.append(wauc(scores[p][b], y[b], w[b]) - wauc(scores[q][b], y[b], w[b]))
            out[pair] = {'delta': d0, 'ci95': np.quantile(bs, [0.025, 0.975]).tolist()}
        res['paired_delta'][name] = out
    ess = {m: {'side_ess_median': float(np.median(np.r_[P[f'{m}_ess0'], P[f'{m}_ess1']])),
               'pair_ess_median': float(np.median(P[f'{m}_ess_pair'])),
               'pair_ess_q05': float(np.quantile(P[f'{m}_ess_pair'], 0.05)),
               'no_valid_pair_fraction': float(np.mean(P[f'{m}_nvalid'] == 0))} for m in MEASURES}
    res['ess'] = ess
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=1))
    np.savez_compressed(a.output.with_suffix('.scores.npz'), **{k.replace(' ', '_'): v for k, v in scores.items()},
                        labels=y, weights=w, modes=modes, reco_mode=rmode, avail=avail, fold=fold)
    for k in scores:
        print(f"{k:55s} {res['auc_truth_pairs'][k]['overall']:.4f}")


if __name__ == '__main__':
    main()
