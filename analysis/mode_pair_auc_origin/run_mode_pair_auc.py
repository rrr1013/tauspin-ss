"""Why does exact-h H/Z separation depend on the decay-mode pair?

Inputs: the canonical exact-h validation cohort cached as truth_surface.npz
(ICEPP /home/rbaba/azimuth-nullspace-20260918/artifacts/truth_surface.npz,
59,390 events, truth pions/pi0/nu and the canonical h_ref).  Nothing is trained.

Steps
  1. closure: generic spinor polarimeter reproduces h_ref for pi and rho.
  2. 3pi: h from the generator's own (TauDecay UFO) current vs h_ref (CLEO current).
  3. per mode pair: fixed-C likelihood-ratio AUC, C matrices, single-side <h_k>.
  4. theory: AUC for unit, isotropic polarimeters with the ideal C.
  5. re-decay toy: each event's true tau momenta, decays replaced by randomly
     rotated rest-frame templates of a chosen mode, ideal spin correlation by
     accept-reject, with and without a visible pT/eta selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from polarimeter import boost, canonical_h, frames

CH = np.diag([1., 1., -1.])
CZ = np.diag([0., 0., 1.])
PAIRS = [(0, 0, 'pi x pi'), (1, 1, 'rho x rho'), (3, 3, '3pi x 3pi'),
         (0, 1, 'pi x rho'), (0, 3, 'pi x 3pi'), (1, 3, 'rho x 3pi')]
MODES = {0: 'pi', 1: 'rho', 3: '3pi'}


def auc(s, y):
    y = np.asarray(y).astype(bool)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = y.sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def boot_auc(s, y, groups=None, B=300, seed=1):
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(s))
    ug, gi = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(gi, kind='stable'), np.cumsum(np.bincount(gi))[:-1])
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[p] for p in pick])
        vals.append(auc(s[idx], y[idx]))
    return float(np.std(vals))


def llr(h):
    o = np.einsum('ni,nj->nij', h[:, 0], h[:, 1])
    return (np.log(np.maximum(1 + np.einsum('ij,nij->n', CH, o), 1e-12))
            - np.log(np.maximum(1 + np.einsum('ij,nij->n', CZ, o), 1e-12)))


def pair_mask(modes, a, b):
    return ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))


def cmat(h, k):
    return (9 * np.einsum('ni,nj->ij', h[k, 0], h[k, 1]) / k.sum()).tolist()


def rand_rot(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1)[:, None]
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--toy-repeats', type=int, default=6)
    ap.add_argument('--pt-cut', type=float, default=20.0)
    ap.add_argument('--eta-cut', type=float, default=2.5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    D = np.load(args.surface)
    h_ref = D['h_ref'].astype(float)
    y = D['labels']
    m = D['modes']
    q = D['pion_charges']
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    out = {'rows': int(len(y)), 'H': int(y.sum()), 'Z': int((y == 0).sum())}

    # 1+2. closure for pi/rho, generator-consistent h for 3pi
    h_gen = h_ref.copy()
    out['closure'] = {}
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, m, q, s, md)
            d = np.sum(h * h_ref[sel, s], -1)
            out['closure'][f'{MODES[md]}_side{s}'] = {
                'n': int(len(sel)), 'dot_min': float(d.min()), 'dot_mean': float(d.mean()),
                'dot_median': float(np.median(d)), 'frac_negative': float(np.mean(d < 0)),
                'norm_max_dev': float(np.abs(np.linalg.norm(h, axis=-1) - 1).max())}
            h_gen[sel, s] = h
    for md in (0, 1):
        for s in (0, 1):
            if out['closure'][f'{MODES[md]}_side{s}']['dot_min'] < 1 - 1e-6:
                raise RuntimeError('generic polarimeter does not close on ' + MODES[md])

    # 3. per mode pair on the real cohort
    out['observed'] = {}
    for tag, h in (('CLEO_hRef', h_ref), ('generator_current', h_gen)):
        s = llr(h)
        rows = {'overall': {'n': int(len(y)), 'auc': auc(s, y), 'se': boot_auc(s, y, B=200)}}
        for a, b, nm in PAIRS:
            k = pair_mask(m, a, b)
            rows[nm] = {'n': int(k.sum()), 'auc': auc(s[k], y[k]), 'se': boot_auc(s[k], y[k]),
                        'C_H': cmat(h, k & (y == 1)), 'C_Z': cmat(h, k & (y == 0))}
        out['observed'][tag] = rows
    vis = D['pions'].sum(2) + D['pi0']
    x = vis[..., 3] / (vis + D['nu4'])[..., 3]
    out['single_side'] = {}
    for md in (0, 1, 3):
        for lab, cn in ((1, 'H'), (0, 'Z')):
            hk = np.concatenate([h_gen[(m[:, s] == md) & (y == lab), s, 2] for s in (0, 1)])
            xs = np.concatenate([x[(m[:, s] == md) & (y == lab), s] for s in (0, 1)])
            out['single_side'][f'{cn}_{MODES[md]}'] = {
                'n': int(len(hk)), 'three_mean_hk': float(3 * hk.mean()),
                'mean_hk2': float(np.mean(hk**2)), 'corr_hk_x': float(np.corrcoef(hk, xs)[0, 1])}

    # 4. theory: unit isotropic polarimeters, ideal C, no acceptance
    rng = np.random.default_rng(0)

    def iso(n):
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=-1)[:, None]

    def sample(C, n):
        got, parts = 0, []
        while got < n:
            a, b = iso(4 * n), iso(4 * n)
            keep = rng.random(4 * n) < (1 + np.einsum('ni,ij,nj->n', a, C, b)) / 2
            parts.append(np.stack([a[keep], b[keep]], 1))
            got += keep.sum()
        return np.concatenate(parts)[:n]

    n = 400000
    yy = np.r_[np.ones(n), np.zeros(n)]
    out['theory'] = {'ideal_auc': auc(llr(np.concatenate([sample(CH, n), sample(CZ, n)])), yy)}
    for alpha in (0.889,):
        out['theory'][f'alpha{alpha}_one_side'] = auc(llr(np.concatenate([sample(CH * alpha, n), sample(CZ * alpha, n)])), yy)
        out['theory'][f'alpha{alpha}_both_sides'] = auc(llr(np.concatenate([sample(CH * alpha**2, n), sample(CZ * alpha**2, n)])), yy)

    # 5. re-decay toy
    rng = np.random.default_rng(7)
    pools = {}
    for md in (0, 1, 3):
        for s in (0, 1):
            sel = np.flatnonzero(m[:, s] == md)
            hc = np.einsum('nji,nj->ni', fr['basis'][sel], h_gen[sel, s])
            vis_rest = fr['rest']['pions'][sel, s].sum(1) + fr['rest']['pi0'][sel, s]
            pools[(md, s)] = (vis_rest, hc)

    def pt_eta(v):
        pt = np.hypot(v[..., 0], v[..., 1])
        return pt, np.arcsinh(v[..., 2] / pt)

    out['toy'] = {'selection': f'visible pT > {args.pt_cut} GeV and |eta| < {args.eta_cut} on both sides',
                  'repeats_per_parent': args.toy_repeats}
    toy_arrays = {}
    for a, b, nm in PAIRS:
        idx = np.repeat(np.arange(len(y)), args.toy_repeats)
        hs, vl = [], []
        for s, md in ((0, a), (1, b)):
            vis_rest, hc = pools[(md, s)]
            t = rng.integers(0, len(hc), len(idx))
            R = rand_rot(rng, len(idx))
            hs.append(np.einsum('nij,nj->ni', fr['basis'][idx], np.einsum('nij,nj->ni', R, hc[t])))
            vr = np.concatenate([np.einsum('nij,nj->ni', R, vis_rest[t, :3]), vis_rest[t, 3:4]], -1)
            vl.append(boost(boost(vr, -fr['bt'][idx, s]), -fr['beta'][idx]))
        hh = np.stack(hs, 1)
        lab = y[idx]
        C = np.where(lab[:, None, None] == 1, CH, CZ)
        acc = rng.random(len(idx)) < (1 + np.einsum('ni,nij,nj->n', hh[:, 0], C, hh[:, 1])) / 2
        (p0, e0), (p1, e1) = pt_eta(vl[0]), pt_eta(vl[1])
        cut = (p0 > args.pt_cut) & (p1 > args.pt_cut) & (abs(e0) < args.eta_cut) & (abs(e1) < args.eta_cut)
        s = llr(hh)
        row = {}
        for tag, k in (('no_selection', acc), ('with_selection', acc & cut)):
            row[tag] = {'n': int(k.sum()), 'auc': auc(s[k], lab[k]),
                        'se': boot_auc(s[k], lab[k], groups=idx[k], B=100),
                        'C_H': cmat(hh, k & (lab == 1)), 'C_Z': cmat(hh, k & (lab == 0)),
                        'three_mean_hk_H_tauminus': float(3 * hh[k & (lab == 1), 0, 2].mean())}
        row['selection_efficiency'] = float(cut[acc].mean())
        out['toy'][nm] = row
        if a == b:
            key = MODES[a]
            toy_arrays[f'{key}_hk'] = hh[..., 2].astype(np.float32)
            toy_arrays[f'{key}_label'] = lab.astype(np.int8)
            toy_arrays[f'{key}_acc'] = acc
            toy_arrays[f'{key}_cut'] = cut
        print(nm, json.dumps({k: round(v['auc'], 4) for k, v in row.items() if isinstance(v, dict)}), flush=True)

    np.savez_compressed(args.output / 'arrays.npz', h_gen=h_gen.astype(np.float32),
                        x=x.astype(np.float32), **toy_arrays)
    (args.output / 'results.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('closure', 'theory')}, indent=1))


if __name__ == '__main__':
    main()
