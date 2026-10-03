"""Step 1: are the weights right?

(a) the cohort's 3pi Dalitz marginals vs flat phase space x omega_gen and x omega_CLEO
    (independent check that omega_gen is the sample's rate),
(b) toy normalisation n(m3pi) and the in-sample E[u | m3pi] (selection bias),
(c) h_CLEO . h_gen vs m3pi and the weight distribution,
(d) closure: under the CLEO-world weights, h_CLEO restores the 3pi x 3pi correlation
    matrix that h_gen gives in the generator world.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import reweight as rw
from currents import dalitz

OUT = rw.HERE / 'results'
OUT.mkdir(exist_ok=True)
COL = dict(sample='black', gen='#0072B2', cleo='#D55E00', other='#009E73')


def chi2(h1, e1, h2, e2):
    m = (e1 > 0) | (e2 > 0)
    return float(np.sum((h1[m] - h2[m]) ** 2 / (e1[m] ** 2 + e2[m] ** 2))), int(m.sum())


def main():
    d = rw.load()
    t = np.load(rw.DATA / 'toy.npz')
    te = np.load(rw.DATA / 'toy_eval.npz')
    e = np.load(rw.DATA / 'eval.npz')
    res = {}
    # sample invariants (all 3pi sides)
    q2s, a_s, b_s = [], [], []
    for side in (0, 1):
        q2, a, b = dalitz(e[f'ordered_side{side}'])
        q2s.append(q2); a_s.append(a); b_s.append(b)
    a_s, b_s = np.concatenate(a_s), np.concatenate(b_s)
    shi_s, slo_s = np.maximum(a_s, b_s), np.minimum(a_s, b_s)
    tq2, ta, tb = dalitz(t['pions'])
    shi_t, slo_t = np.maximum(ta, tb), np.minimum(ta, tb)
    parent = t['parent']
    wt = {}
    for k, col in (('gen', 'out_gen'), ('cleo', 'out_cleo')):
        om = te[col][:, 4]
        norm = np.bincount(parent, om)
        wt[k] = om / norm[parent]          # each sample side carries total weight 1
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    res['dalitz_chi2'] = {}
    for j, (xs, xt, lab) in enumerate(((shi_s, shi_t, r'larger $s_{\pi^\mp\pi^\pm}$ [GeV$^2$]'),
                                       (slo_s, slo_t, r'smaller $s_{\pi^\mp\pi^\pm}$ [GeV$^2$]'))):
        edges = np.linspace(0.0, 2.6, 53) if j == 0 else np.linspace(0.0, 1.6, 41)
        hs, _ = np.histogram(xs, edges)
        es = np.sqrt(hs)
        c = 0.5 * (edges[1:] + edges[:-1])
        a = ax[0, j]
        a.errorbar(c, hs, es, fmt='o', ms=3, color=COL['sample'], label=f'cohort 3$\\pi$ sides ({len(xs)})')
        for k, ls in (('gen', '-'), ('cleo', '--')):
            ht, _ = np.histogram(xt, edges, weights=wt[k])
            et = np.sqrt(np.histogram(xt, edges, weights=wt[k] ** 2)[0])
            x2, nd = chi2(hs, es, ht, et)
            res['dalitz_chi2'][f'{"high" if j == 0 else "low"}_{k}'] = {'chi2': x2, 'bins': nd}
            name = 'flat PS $\\times\\,\\omega_{\\rm gen}$' if k == 'gen' else 'flat PS $\\times\\,\\omega_{\\rm CLEO}$'
            a.stairs(ht, edges, color=COL[k], ls=ls, lw=1.8, label=f'{name}  ($\\chi^2$/bins = {x2:.0f}/{nd})')
        a.set_xlabel(lab); a.set_ylabel('3$\\pi$ sides / bin')
        a.legend(fontsize=8, frameon=False)
        a.set_title('(a) Dalitz marginal: higher pair mass' if j == 0 else '(b) Dalitz marginal: lower pair mass', fontsize=10, loc='left')
    # (c) normalisation and sample E[u|m]
    centres, n, cnt = rw.toy_normalisation()
    a = ax[1, 0]
    ok = np.isfinite(n)
    m3 = d['m3'][d['is3']]
    u = d['u'][d['is3']]
    edges = np.linspace(0.55, 1.777, 15)
    b = np.digitize(m3, edges) - 1
    cc = 0.5 * (edges[1:] + edges[:-1])
    mu = np.array([u[b == i].mean() if (b == i).sum() > 30 else np.nan for i in range(len(cc))])
    se = np.array([u[b == i].std() / np.sqrt((b == i).sum()) if (b == i).sum() > 30 else np.nan for i in range(len(cc))])
    a.errorbar(cc, mu, se, fmt='s', ms=5, color=COL['sample'], label=r'cohort: $E[u\,|\,m_{3\pi}]$ (should be 1)')
    a.plot(centres[ok], n[ok] / np.nanmedian(n[ok]), '-', color=COL['other'], marker='.', lw=1.5,
           label=r'toy $n(m_{3\pi})$ / median (lineshape factor removed)')
    a.axhline(1, color='0.5', lw=0.8)
    a.set_xlabel(r'$m_{3\pi}$ [GeV]'); a.set_ylabel('ratio')
    a.legend(fontsize=8, frameon=False)
    a.set_title(r'(c) per-$m_{3\pi}$ normalisation of $u=\omega_{\rm CLEO}/\omega_{\rm gen}\cdot n$', fontsize=10, loc='left')
    res['E_u_given_m'] = {'centres': cc.tolist(), 'mean': mu.tolist(), 'se': se.tolist()}
    # (d) closure of the correlation matrix, 3pi x 3pi
    a = ax[1, 1]
    both = d['is3'].all(1)
    y, w = d['y'], d['w']
    rows = []
    labels = []
    for lab, name in ((1, 'H'), (0, 'Z')):
        m = both & (y == lab)
        for hk, wk, tag in (('h_gen', None, 'gen world, $h_{\\rm gen}$'), ('h_gen', 'w', 'CLEO world, $h_{\\rm gen}$'),
                            ('h_cleo', 'w', 'CLEO world, $h_{\\rm CLEO}$')):
            h = d[hk][m]
            ww = np.ones(m.sum()) if wk is None else w[m]
            diag = [float(np.sum(ww * h[:, 0, i] * h[:, 1, i]) / ww.sum() * 9) for i in range(3)]
            rows.append(diag); labels.append(f'{name}: {tag}')
    res['corr_3pi3pi_9mean_hh'] = dict(zip(labels, rows))
    xs = np.arange(3)
    styles = [('o', '-'), ('s', '--'), ('^', ':')]
    for i, (r, l) in enumerate(zip(rows, labels)):
        mk, ls = styles[i % 3]
        col = COL['gen'] if i % 3 == 0 else (COL['cleo'] if i % 3 == 2 else '0.45')
        a.plot(xs + (i // 3) * 0.08 - 0.04, r, marker=mk, ls=ls, color=col, ms=6, lw=1,
               mfc='none' if i >= 3 else col, label=l)
    a.set_xticks(xs, [r'$9\langle h^-_nh^+_n\rangle$', r'$9\langle h^-_rh^+_r\rangle$', r'$9\langle h^-_kh^+_k\rangle$'])
    a.axhline(0, color='0.6', lw=0.6)
    a.set_ylabel(r'$3\pi\times3\pi$ (filled H, open Z)')
    a.legend(fontsize=7, frameon=False, ncol=1)
    a.set_title(r'(d) closure: $h_{\rm CLEO}$ in the CLEO world vs $h_{\rm gen}$ in the generator world', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_weight_validation.png', dpi=140)
    # scalars
    is3 = d['is3']
    dot = np.sum(d['h_gen'] * d['h_cleo'], -1)[is3]
    any3 = is3.any(1)
    res['weights'] = {lab: {'events': int((any3 & (y == v)).sum()),
                            'mean_w': float(w[any3 & (y == v)].mean()),
                            'ess_fraction': float(w[any3 & (y == v)].sum() ** 2 / (w[any3 & (y == v)] ** 2).sum() / (any3 & (y == v)).sum()),
                            'w_quantiles_0_1_50_99_100': np.quantile(w[any3 & (y == v)], [0, .01, .5, .99, 1]).tolist()}
                      for lab, v in (('H', 1), ('Z', 0))}
    res['mean_dot_hgen_hcleo'] = {'generator_world': float(dot.mean()),
                                  'cleo_world': float(np.sum(np.repeat(w[:, None], 2, 1)[is3] * dot) / np.repeat(w[:, None], 2, 1)[is3].sum())}
    (OUT / 'validation.json').write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != 'E_u_given_m'}, indent=1))


if __name__ == '__main__':
    main()
