"""Steps 2-4: model dependence of exact and learned polarimeters.

Worlds: 'gen' (the cohort as generated, weight w0 = parent-overlap weight) and
'cleo' (weight w0 * w_CLEO from reweight.py).  Estimators never change between
worlds; only event weights do.  test split is never read.
"""
import json

import numpy as np

import reweight as rw

ARMS = ('base_s43', 'full22_s42', 'full22_s43', 'ens_full22', 'idealip22_s42')
PAIRS = {'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}
BOOT = 300
RNG = np.random.default_rng(20261004)


def wauc(score, y, w):
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], y[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def lr(h):
    p = h[:, 0] * h[:, 1]
    return np.log(1 + p @ rw.C_H) - np.log(1 + p @ rw.C_Z)


def load_all(lineshape='generator'):
    d = rw.load(lineshape=lineshape)
    nets = {}
    for teacher, prefix in (('gen', 'gen3pi'), ('cleo', 'cleo')):
        for arm in ARMS:
            f = rw.DATA / teacher / f'readout_scores_{arm}.npz'
            z = np.load(f)
            assert np.array_equal(z['global_indices'], d['ids'])
            nets[(teacher, arm)] = {'score': z['scores'].astype(float)}
            hp = rw.COHORT / f'{prefix}_{arm}.npz'
            if hp.exists():
                zz = np.load(hp)
                assert np.array_equal(zz['global_indices'], d['ids'])
                assert np.array_equal(zz['labels'].astype(int), d['y'])
                nets[(teacher, arm)]['h_pred'] = zz['h_pred'].astype(float)
    return d, nets


def selections(modes):
    sel = {k: ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))
           for k, (a, b) in PAIRS.items()}
    sel['any 3pi'] = (modes == 3).any(1)
    sel['overall'] = np.ones(len(modes), bool)
    return sel


def auc_table(d, scores):
    """AUC for each score in both worlds and paired-bootstrap CIs of selected differences."""
    y, w0, w = d['y'], d['w0'], d['w']
    worlds = {'gen': w0, 'cleo': w0 * w}
    sel = selections(d['modes'])
    out = {}
    for name, s in scores.items():
        out[name] = {p: {wn: wauc(s[m], y[m], ww[m]) for wn, ww in worlds.items()} for p, m in sel.items()}
    return out, sel, worlds


def boot_diffs(d, scores, contrasts, sel, worlds, pairs=('any 3pi', '3pi x 3pi', 'overall')):
    """contrasts: list of (label, (score_a, world_a), (score_b, world_b)) -> b - a."""
    y = d['y']
    res = {}
    for p in pairs:
        idx = np.flatnonzero(sel[p])
        boots = {c[0]: [] for c in contrasts}
        for _ in range(BOOT):
            b = RNG.choice(idx, len(idx))
            cache = {}
            for lab, a, bb in contrasts:
                for key in (a, bb):
                    if key not in cache:
                        cache[key] = wauc(scores[key[0]][b], y[b], worlds[key[1]][b])
                boots[lab].append(cache[bb] - cache[a])
        for lab, a, bb in contrasts:
            m = sel[p]
            point = wauc(scores[bb[0]][m], y[m], worlds[bb[1]][m]) - wauc(scores[a[0]][m], y[m], worlds[a[1]][m])
            res.setdefault(lab, {})[p] = {'delta': point, 'ci68': np.quantile(boots[lab], [0.16, 0.84]).tolist(),
                                          'ci95': np.quantile(boots[lab], [0.025, 0.975]).tolist()}
    return res


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def analyzing_power(d, edges=np.linspace(0.55, 1.777, 15)):
    """Effective analyzing power of h_gen in the CLEO world, <h_gen . h_CLEO>, vs m3pi."""
    is3 = d['is3']
    dot = np.sum(d['h_gen'] * d['h_cleo'], -1)
    ww = np.repeat((d['w0'] * d['w'])[:, None], 2, 1)
    m3, dot, ww = d['m3'][is3], dot[is3], ww[is3]
    b = np.digitize(m3, edges) - 1
    cc = 0.5 * (edges[1:] + edges[:-1])
    alpha = [wmean(dot[b == i], ww[b == i]) if (b == i).sum() > 30 else None for i in range(len(cc))]
    frac = [float(ww[b == i].sum() / ww.sum()) for i in range(len(cc))]
    return {'centres': cc.tolist(), 'alpha': alpha, 'weight_fraction': frac,
            'alpha_all': wmean(dot, ww)}


def transverse_and_cp(d, nets):
    """H events with two... any 3pi side: transverse correlation strength and CP-odd triple product.

    A_perp = E_H[h-_n h+_n + h-_r h+_r] / 2 for the estimator; calibrated in the gen world against
    the true transverse correlation (C_nn = C_rr = 1), i.e. reported as A_perp(world)/A_perp(gen).
    T = E_H[(h- x h+) . k] must stay 0 (CP-even H, CP-symmetric current change).
    """
    y, w0, w, is3 = d['y'], d['w0'], d['w'], d['is3']
    m = (y == 1) & is3.any(1)
    est = {'exact h_gen': d['h_gen'], 'exact h_CLEO': d['h_cleo']}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v and arm in ('base_s43', 'full22_s42'):
            est[f'reco {arm} (teacher {teacher})'] = v['h_pred']
    out = {}
    for name, h in est.items():
        hm = h[m]
        a = 0.5 * (hm[:, 0, 0] * hm[:, 1, 0] + hm[:, 0, 1] * hm[:, 1, 1])
        t = np.cross(hm[:, 0], hm[:, 1])[:, 2]
        r = {}
        for wn, ww in (('gen', w0[m]), ('cleo', (w0 * w)[m])):
            r[wn] = {'A_perp': wmean(a, ww), 'T_k': wmean(t, ww),
                     'T_k_se': float(np.sqrt(np.sum(ww**2 * (t - wmean(t, ww))**2)) / ww.sum())}
            # bootstrap-free SE of A_perp
            r[wn]['A_perp_se'] = float(np.sqrt(np.sum(ww**2 * (a - r[wn]['A_perp'])**2)) / ww.sum())
        r['ratio_cleo_over_gen'] = r['cleo']['A_perp'] / r['gen']['A_perp']
        out[name] = r
    return out


def polarisation(d, nets):
    """P_tau from 3pi sides of Z, H as the P = 0 control, calibrated in the gen world.

    x = sum over 3pi sides of h_k (estimator).  P_hat = (E_Z[x] - E_H[x]) / s, where
    s = dE_Z[x]/dP from reweighting the gen-world Z by f(P) with the generator h.
    The bias of interest is P_hat(CLEO world) - P_hat(gen world) (the method's own
    gen-world bias cancels).  sigma_1 = per-event spread for converting to N*.
    """
    y, w0, w, is3 = d['y'], d['w0'], d['w'], d['is3']
    any3 = is3.any(1)
    hz = d['h_gen']
    f = 1 + rw.P_TAU_Z * (hz[:, 0, 2] + hz[:, 1, 2]) + (hz[:, 0] * hz[:, 1]) @ rw.C_Z
    score = (hz[:, 0, 2] + hz[:, 1, 2]) / f              # d log f / dP for Z
    est = {'exact h_gen': d['h_gen'], 'exact h_CLEO': d['h_cleo']}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v and arm in ('base_s43', 'full22_s42'):
            est[f'reco {arm} (teacher {teacher})'] = v['h_pred']
    out = {}
    mz, mh = any3 & (y == 0), any3 & (y == 1)
    for name, h in est.items():
        x = np.where(is3, h[..., 2], 0.0).sum(1)
        s_gen = wmean((x[mz] - wmean(x[mz], w0[mz])) * score[mz], w0[mz])
        r = {'slope_gen': s_gen}
        for wn, ww in (('gen', w0), ('cleo', w0 * w)):
            ez, eh = wmean(x[mz], ww[mz]), wmean(x[mh], ww[mh])
            r[wn] = {'E_Z': ez, 'E_H': eh, 'P_hat': (ez - eh) / s_gen}
        r['bias_P'] = r['cleo']['P_hat'] - r['gen']['P_hat']
        r['bias_sin2'] = r['bias_P'] / 7.870
        r['sigma1_P'] = float(np.sqrt(np.var(x[mz]) + np.var(x[mh])) / abs(s_gen))
        r['N_star_Z_events'] = float((r['sigma1_P'] / r['bias_P']) ** 2) if r['bias_P'] != 0 else None
        out[name] = r
    return out


def template_fraction(d, score, mask, nbins=20):
    """Asimov fit of the H fraction in a 50/50 mixture: templates from the gen world,
    pseudo-data from the CLEO world.  Returns bias in f and sigma_f per event."""
    y, w0, w = d['y'], d['w0'], d['w']
    s = score[mask]
    edges = np.quantile(s, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    def hist(lab, ww):
        m = y[mask] == lab
        h = np.histogram(s[m], edges, weights=ww[mask][m])[0]
        return h / h.sum()
    HG, ZG = hist(1, w0), hist(0, w0)
    HC, ZC = hist(1, w0 * w), hist(0, w0 * w)
    D = 0.5 * HC + 0.5 * ZC
    fs = np.linspace(0.3, 0.7, 4001)
    nll = [-np.sum(D * np.log(f * HG + (1 - f) * ZG)) for f in fs]
    fhat = float(fs[int(np.argmin(nll))])
    model = 0.5 * HG + 0.5 * ZG
    info = float(np.sum((HG - ZG) ** 2 / model))      # per event
    return {'f_hat': fhat, 'bias': fhat - 0.5, 'sigma_f_per_sqrtN': float(1 / np.sqrt(info)),
            'N_star': float((1 / np.sqrt(info) / (fhat - 0.5)) ** 2) if fhat != 0.5 else None}


def main(lineshape='generator'):
    d, nets = load_all(lineshape)
    scores = {'exact LR h_gen': lr(d['h_gen']), 'exact LR h_CLEO': lr(d['h_cleo'])}
    for (teacher, arm), v in nets.items():
        scores[f'{teacher}:{arm}'] = v['score']
    table, sel, worlds = auc_table(d, scores)
    contrasts = [('exact: world effect on h_gen', ('exact LR h_gen', 'gen'), ('exact LR h_gen', 'cleo')),
                 ('exact: right polarimeter in CLEO world', ('exact LR h_gen', 'cleo'), ('exact LR h_CLEO', 'cleo'))]
    for arm in ARMS:
        contrasts += [(f'{arm}: world effect, gen teacher', (f'gen:{arm}', 'gen'), (f'gen:{arm}', 'cleo')),
                      (f'{arm}: world effect, cleo teacher', (f'cleo:{arm}', 'gen'), (f'cleo:{arm}', 'cleo')),
                      (f'{arm}: teacher effect (cleo - gen) in gen world', (f'gen:{arm}', 'gen'), (f'cleo:{arm}', 'gen')),
                      (f'{arm}: teacher effect (cleo - gen) in CLEO world', (f'gen:{arm}', 'cleo'), (f'cleo:{arm}', 'cleo'))]
    diffs = boot_diffs(d, scores, contrasts, sel, worlds)
    out = {'auc': table, 'differences': diffs,
           'analyzing_power': analyzing_power(d),
           'transverse_and_cp_H': transverse_and_cp(d, nets),
           'polarisation': polarisation(d, nets),
           'template_fraction': {}}
    for name in ('exact LR h_gen', 'gen:base_s43', 'gen:full22_s42', 'gen:ens_full22', 'cleo:full22_s42'):
        out['template_fraction'][name] = {p: template_fraction(d, scores[name], sel[p]) for p in ('any 3pi', 'overall')}
    (rw.HERE / 'results' / ('analysis.json' if lineshape == 'generator' else f'analysis_{lineshape}_lineshape.json')).write_text(json.dumps(out, indent=2))
    if lineshape == 'generator':
        np.savez_compressed(rw.HERE / 'results' / 'arrays.npz', w=d['w'], ids=d['ids'])
    short = {k: {p: {wn: round(x, 4) for wn, x in v[p].items()} for p in ('any 3pi', '3pi x 3pi', 'overall')}
             for k, v in table.items()}
    print(json.dumps(short, indent=0))
    print(json.dumps({k: {p: [round(v[p]['delta'], 4)] + [round(c, 4) for c in v[p]['ci95']] for p in v}
                      for k, v in diffs.items()}, indent=0))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
