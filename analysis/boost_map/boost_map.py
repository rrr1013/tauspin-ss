"""Where along the tau boost does the H/Z spin information live, and how much
of it does each reconstruction arm recover?

Inputs: the aligned validation cohort (59,390 events) in ../cp_mixing/data.
No network is retrained and the test split is never read.

Readout: one fixed function of the polarimeter pair for every bin, a weighted
logistic regression on the nine bilinears h-_i h+_j (the exact H/Z likelihood
ratio is a function of these), cross-fitted on two halves of the validation
cohort split by global index parity.  Every arm (exact h and each h_pred) gets
its own readout, fitted the same way.

Output: results.json and per-event scores (npz) next to this file.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'cp_mixing', 'data')
M_TAU = 1.77686

ARMS = {
    # name: (file, key)   -- gen3pi = generator-current 3pi teacher (current standard)
    'exact': ('gen3pi_full22_s42', 'h'),
    'base': ('gen3pi_base_s43', 'h_pred'),
    'ipsv': ('gen3pi_full22_s42', 'h_pred'),
    'ipsv_s43': ('gen3pi_full22_s43', 'h_pred'),
    'idealip': ('gen3pi_idealip22_s42', 'h_pred'),
    # CLEO-teacher arms: the only place with a shuffled-geometry control
    'cleo_exact': ('cleo_full22_s42', 'h'),
    'cleo_base': ('cleo_base_s43', 'h_pred'),
    'cleo_ipsv': ('cleo_full22_s42', 'h_pred'),
    'cleo_shuffle': ('cleo_full22_shuffle_s42', 'h_pred'),
}


def load():
    L = np.load(os.path.join(DATA, 'ladder_validation.npz'))
    gi = L['global_indices']
    out = {'labels': L['labels'].astype(int), 'w': L['weights'].astype(float),
           'modes': L['modes'], 'gi': gi, 'h': {}}
    for name, (f, key) in ARMS.items():
        d = np.load(os.path.join(DATA, f + '.npz'))
        assert (d['global_indices'] == gi).all() and (d['labels'] == out['labels']).all()
        assert np.allclose(d['overlap_weights'], out['w'])
        out['h'][name] = d[key].astype(float)
    tau = L['truth_visible_tau_lab4'] + L['truth_neutrino_lab4']       # (px, py, pz, E)
    p = np.linalg.norm(tau[..., :3], axis=-1)
    out['bg'] = p / M_TAU                                              # (N, 2)
    vis = L['reco_visible_tau_lab4']
    out['vis_pt'] = np.hypot(vis[..., 0], vis[..., 1])                 # (N, 2)
    tv = L['truth_visible_tau_lab4']
    out['x_vis'] = tv[..., 3] / tau[..., 3]                            # visible energy fraction
    # opening angle between truth visible and truth tau direction
    cosang = np.einsum('nsi,nsi->ns', tv[..., :3], tau[..., :3]) / (
        np.linalg.norm(tv[..., :3], axis=-1) * p)
    out['theta_vis_tau'] = np.arccos(np.clip(cosang, -1, 1))
    return out


def features(h):
    hm, hp = h[:, 0], h[:, 1]
    return np.einsum('ni,nj->nij', hm, hp).reshape(len(h), 9)


def fit_logistic(X, y, w, l2=1e-4, iters=50):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    mu, sd = X1[:, 1:].mean(0), X1[:, 1:].std(0) + 1e-12
    X1[:, 1:] = (X1[:, 1:] - mu) / sd
    b = np.zeros(X1.shape[1])
    for _ in range(iters):
        z = X1 @ b
        p = 1 / (1 + np.exp(-z))
        g = X1.T @ (w * (p - y)) + l2 * b
        Hs = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(b))
        step = np.linalg.solve(Hs, g)
        b -= step
        if np.abs(step).max() < 1e-9:
            break
    return b, mu, sd


def apply_logistic(model, X):
    b, mu, sd = model
    return b[0] + ((X - mu) / sd) @ b[1:]


def crossfit_scores(h, y, w, gi):
    X = features(h)
    s = np.empty(len(y))
    fold = gi % 2
    for k in (0, 1):
        tr, te = fold != k, fold == k
        s[te] = apply_logistic(fit_logistic(X[tr], y[tr], w[tr]), X[te])
    return s


def wauc(s, y, w):
    """Weighted Mann-Whitney AUC with ties counted half."""
    o = np.argsort(s, kind='mergesort')
    s, y, w = s[o], y[o], w[o]
    wp, wn = w * (y == 1), w * (y == 0)
    # group ties
    uniq, start = np.unique(s, return_index=True)
    cp = np.add.reduceat(wp, start)
    cn = np.add.reduceat(wn, start)
    cum_n = np.cumsum(cn) - cn
    return float(np.sum(cp * (cum_n + 0.5 * cn)) / (cp.sum() * cn.sum()))


def bin_stats(scores, y, w, idx, arms, pairs, nboot=200, seed=1):
    rng = np.random.default_rng(seed)
    res = {'n': int(idx.sum()), 'n_H': int((y[idx] == 1).sum()), 'auc': {}, 'auc_se': {}, 'diff': {}, 'diff_se': {}}
    ii = np.flatnonzero(idx)
    for a in arms:
        res['auc'][a] = wauc(scores[a][ii], y[ii], w[ii])
    boots = {a: [] for a in arms}
    for _ in range(nboot):
        k = rng.choice(ii, len(ii))
        for a in arms:
            boots[a].append(wauc(scores[a][k], y[k], w[k]))
    boots = {a: np.array(v) for a, v in boots.items()}
    for a in arms:
        res['auc_se'][a] = float(boots[a].std())
    for a, b in pairs:
        key = f'{a}-{b}'
        res['diff'][key] = res['auc'][a] - res['auc'][b]
        res['diff_se'][key] = float((boots[a] - boots[b]).std())
    return res


def wcorr(a, b, w):
    am, bm = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.average((a - am) * (b - bm), weights=w) /
                 np.sqrt(np.average((a - am) ** 2, weights=w) * np.average((b - bm) ** 2, weights=w)))


def h_quality(D, idx, arms):
    """Per-bin correlation of predicted vs exact h components, both sides pooled.
    exact for gen3pi arms is D['h']['exact']; for cleo arms D['h']['cleo_exact']."""
    out = {}
    for a in arms:
        ref = D['h']['cleo_exact' if a.startswith('cleo') else 'exact'][idx]
        hp = D['h'][a][idx]
        w = np.repeat(D['w'][idx], 2)
        R, P = ref.reshape(-1, 3), hp.reshape(-1, 3)
        out[a] = {'k': wcorr(P[:, 2], R[:, 2], w),
                  'T': float(np.mean([wcorr(P[:, j], R[:, j], w) for j in (0, 1)]))}
    return out


def main():
    D = load()
    y, w, gi = D['labels'], D['w'], D['gi']
    scores = {a: crossfit_scores(D['h'][a], y, w, gi) for a in ARMS}
    arms = list(ARMS)
    pairs = [('ipsv', 'base'), ('idealip', 'ipsv'), ('idealip', 'base'), ('exact', 'base'),
             ('cleo_ipsv', 'cleo_shuffle'), ('cleo_shuffle', 'cleo_base'), ('ipsv_s43', 'base')]

    bg_gm = np.sqrt(D['bg'][:, 0] * D['bg'][:, 1])
    bg_min, bg_max = D['bg'].min(1), D['bg'].max(1)
    vis_sum = D['vis_pt'].sum(1)
    one_prong = (D['modes'] != 3).all(1)
    pipi = (D['modes'] == 0).all(1)

    results = {'n_events': int(len(y)), 'global': bin_stats(scores, y, w, np.ones(len(y), bool), arms, pairs, nboot=200)}
    results['global']['hq'] = h_quality(D, np.ones(len(y), bool), arms)

    def quantile_bins(v, nb, mask=None):
        m = np.ones(len(v), bool) if mask is None else mask
        return np.quantile(v[m], np.linspace(0, 1, nb + 1))

    results['binnings'] = {}
    specs = [('bg_gm', bg_gm, 8, None), ('bg_min', bg_min, 6, None), ('bg_max', bg_max, 6, None),
             ('vis_pt_sum', vis_sum, 8, None),
             ('bg_gm_1prong', bg_gm, 6, one_prong), ('bg_gm_pipi', bg_gm, 4, pipi),
             ('bg_gm_with3p', bg_gm, 6, ~one_prong)]
    for name, v, nb, mask in specs:
        edges = quantile_bins(v, nb, mask)
        bins = []
        m = np.ones(len(v), bool) if mask is None else mask
        for i in range(nb):
            hi_ok = v <= edges[i + 1] if i == nb - 1 else v < edges[i + 1]
            idx = m & (v >= edges[i]) & hi_ok
            st = bin_stats(scores, y, w, idx, arms, pairs, nboot=150, seed=10 + i)
            st['lo'], st['hi'] = float(edges[i]), float(edges[i + 1])
            st['median'] = float(np.median(v[idx]))
            st['hq'] = h_quality(D, idx, arms)
            st['frac_H'] = float(np.average(y[idx], weights=w[idx]))
            st['median_theta_vis_tau_mrad'] = float(1e3 * np.median(D['theta_vis_tau'][idx]))
            st['median_bg_gm'] = float(np.median(bg_gm[idx]))
            bins.append(st)
        results['binnings'][name] = bins
        print(name, [f"{b['median']:.0f}:{b['auc']['exact']:.3f}/{b['auc']['base']:.3f}/{b['auc']['ipsv']:.3f}/{b['auc']['idealip']:.3f}" for b in bins])

    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(HERE, 'scores.npz'), gi=gi, y=y, w=w, bg=D['bg'], vis_pt=D['vis_pt'],
                        modes=D['modes'], theta_vis_tau=D['theta_vis_tau'], x_vis=D['x_vis'],
                        **{f's_{a}': scores[a] for a in arms})
    g = results['global']
    print('global auc', {a: round(v, 4) for a, v in g['auc'].items()})
    print('global diff', {k: f"{v:+.4f}±{g['diff_se'][k]:.4f}" for k, v in g['diff'].items()})


if __name__ == '__main__':
    main()
