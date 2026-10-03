"""Can the 3pi current be told apart in data from the reconstructed Dalitz plot?

Per 3pi side, the unpolarised density ratio CLEO/generator is u (reweight.py).
The expected information per decay for 'CLEO world vs generator world' is the
KL divergence E_CLEO[log u].  With only observable x, it is E_CLEO[log uhat(x)],
uhat(x) = E_gen[u | x], estimated in bins of x.  Observables: (m3pi, s_hi, s_lo)
from truth pions or from the three reconstructed core tracks (pion mass).  The
number of decays for an expected 5 sigma separation is 12.5 / KL.
Spin is ignored (single-side marginal; spin terms average out over orientation).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import reweight as rw
from currents import dalitz

MPI = 0.13957018


def reco_invariants(r, d):
    p = r['reco_core_pion_lab4'].astype(float)          # (N,2,3,4) px,py,pz,E
    q = r['reco_core_charges']
    ok = d['is3'] & (r['reco_core_count'] == 3)
    out = np.full(d['is3'].shape + (3,), np.nan)
    for side, ch in ((0, -1), (1, 1)):
        sel = np.flatnonzero(ok[:, side])
        pp, qq = p[sel, side], q[sel, side]
        pat = (np.sum(qq == ch, -1) == 2) & (np.sum(qq == -ch, -1) == 1)
        sel, pp, qq = sel[pat], pp[pat], qq[pat]
        order = np.argsort(qq != ch, axis=-1, kind='stable')
        po = np.take_along_axis(pp, order[..., None], axis=1)
        e = np.sqrt(np.sum(po[..., :3] ** 2, -1) + MPI ** 2)    # pion-mass hypothesis
        po = np.concatenate([po[..., :3], e[..., None]], -1)
        q2, a, b = dalitz(po)
        out[sel, side] = np.stack([np.sqrt(q2), np.maximum(a, b), np.minimum(a, b)], -1)
    return out


def kl_binned(x, u, edges):
    """E_C[log uhat(bin)] with weights u (CLEO) vs 1 (generator); x (n,k)."""
    idx = np.zeros(len(x), int)
    mult = 1
    for j in range(x.shape[1] - 1, -1, -1):
        b = np.clip(np.digitize(x[:, j], edges[j]) - 1, 0, len(edges[j]) - 2)
        idx += b * mult
        mult *= len(edges[j]) - 1
    n = np.bincount(idx, minlength=mult).astype(float)
    su = np.bincount(idx, u, minlength=mult)
    uh = np.where(n > 0, su / np.maximum(n, 1), 1.0)
    return float(np.sum(u * np.log(uh[idx])) / np.sum(u))


def main():
    d = rw.load()
    r = np.load(rw.DATA / 'reco_pions_validation.npz')
    assert np.array_equal(r['global_indices'], d['ids'])
    e = np.load(rw.DATA / 'eval.npz')
    truth = np.full(d['is3'].shape + (3,), np.nan)
    truth[..., 0] = d['m3']
    truth[..., 1] = np.fmax(d['s13'], d['s23'])
    truth[..., 2] = np.fmin(d['s13'], d['s23'])
    reco = reco_invariants(r, d)
    both = d['is3'] & np.isfinite(reco).all(-1)
    xt, xr, u = truth[both], reco[both], d['u'][both]
    u = u / u.mean()
    res = {'sides': int(both.sum()), 'exact_KL_truth_u': float(np.sum(u * np.log(u)) / u.sum())}
    edges = {'m3pi': np.linspace(0.5, 1.8, 14), 'shi': np.linspace(0.0, 2.0, 21), 'slo': np.linspace(0.0, 1.2, 13)}
    for name, x in (('truth', xt), ('reco', xr)):
        res[f'KL_{name}_3d'] = kl_binned(x, u, [edges['m3pi'], edges['shi'], edges['slo']])
        res[f'KL_{name}_dalitz2d'] = kl_binned(x[:, 1:], u, [edges['shi'], edges['slo']])
        res[f'KL_{name}_shi'] = kl_binned(x[:, 1:2], u, [np.linspace(0, 2.0, 41)])
    # finite-sample bias check: same binning, u from a random permutation (no information)
    perm = np.random.default_rng(3).permutation(len(u))
    res['KL_null_permuted_3d'] = kl_binned(xr, u[perm], [edges['m3pi'], edges['shi'], edges['slo']])
    for k in list(res):
        if k.startswith('KL_') and 'null' not in k:
            res[k.replace('KL_', 'N5sigma_')] = 12.5 / max(res[k] - res['KL_null_permuted_3d'] * ('3d' in k), 1e-12)
    res['resolution_reco_minus_truth'] = {
        n: np.quantile(xr[:, j] - xt[:, j], [0.16, 0.5, 0.84]).tolist() for j, n in enumerate(('m3pi', 's_hi', 's_lo'))}
    (rw.HERE / 'results' / 'dalitz_discrimination.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    # figure: reco s_hi in both worlds, truth for reference
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (x, lab) in zip(ax, ((xt, 'truth pions'), (xr, 'reconstructed core tracks'))):
        ed = np.linspace(0.0, 1.8, 46)
        hg = np.histogram(x[:, 1], ed)[0]
        hc = np.histogram(x[:, 1], ed, weights=u)[0]
        c = 0.5 * (ed[1:] + ed[:-1])
        a.stairs(hg, ed, color='#0072B2', lw=1.8, label='generator world')
        a.stairs(hc, ed, color='#D55E00', lw=1.8, ls='--', label='CLEO world')
        a.set_xlabel(r'larger $s_{\pi^\mp\pi^\pm}$ [GeV$^2$] (' + lab + ')')
        a.set_ylabel('3$\\pi$ sides / bin')
        a.legend(frameon=False, fontsize=8)
        a.set_title(('(a) ' if a is ax[0] else '(b) ') + lab + f', {len(x)} sides', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(rw.HERE / 'results' / 'fig4_reco_dalitz.png', dpi=140)


if __name__ == '__main__':
    main()
