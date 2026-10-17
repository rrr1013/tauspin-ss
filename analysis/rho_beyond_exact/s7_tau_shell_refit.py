"""Does the tau mass shell explain the denoised reference?  Classical pi0 refit with the truth neutrino.

Per reco-rho side (reco_h_mode 1, available) with the truth neutrino nu and the reco charged pion pi, the pi0 momentum
is moved by the smallest resolution-weighted amount that puts the tau on its mass shell:
    minimise  d^T A d   subject to  (pi + nu + pi0')^2 = m_tau^2,   pi0' = (p0 + d, sqrt(|p0 + d|^2 + m_pi0^2))
A is diagonal in the frame of the reco pi0: along p0 1/sigma_p^2, transverse 1/(sigma_theta |p0|)^2.  sigma_theta and
sigma_p/p are 68%-core widths per reco pi0 energy quintile, fitted on a TRAIN subset (truth pi0 re-read from ROOT with
gen3pi_teacher/build_targets.read_truth).  Variant 'dir': energy held (only the direction moves, via a large along-axis
weight).  Solved by vectorised Gauss-Newton on the linearised constraint (25 iterations).

The exact formula (s1) is then evaluated with the refit pi0.  Reported: rho-side quality (s3 sides), weighted H/Z AUC by
truth pair (formula readout), and the formula / denoised / learner quality in bins of the pre-refit shell violation
|m(pi + pi0 + nu) - m_tau|.  Test never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'gen3pi_teacher'))
import s1_reference_ladder as s1  # noqa: E402
from s2_readouts import formula, pair_masks, wauc  # noqa: E402

M_TAU, M_PI0 = 1.77686, 0.1349768
GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def train_resolutions(n_events, seed):
    from build_targets import read_truth
    t = np.load(GEN / 'targets' / 'train.npz')
    v = t['h_valid'].astype(bool)
    with np.load(s1.LADDER / 'train.npz') as d:
        L = {k: d[k] for k in ('global_indices', 'reco_h_neutral_lab4', 'reco_h_mode', 'reco_h_available_side', 'modes')}
    pos = {int(g): i for i, g in enumerate(L['global_indices'])}
    cand = np.flatnonzero(v & ((t['modes'] == 1).any(1)))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(cand, min(n_events, len(cand)), replace=False))
    tr = read_truth(t['labels'][rows], t['file_indices'][rows], t['entry_indices'][rows], t['event_numbers'][rows],
                    t['modes'][rows])
    lr = np.array([pos[int(g)] for g in t['global_indices'][rows]])
    rn = L['reco_h_neutral_lab4'][lr]
    side = (L['reco_h_mode'][lr] == 1) & L['reco_h_available_side'][lr] & (t['modes'][rows] == 1)
    p0, pt = rn[..., :3][side], tr['pi0'][..., :3][side]
    e0 = rn[..., 3][side]
    cos = np.sum(p0 * pt, -1) / np.maximum(np.linalg.norm(p0, axis=-1) * np.linalg.norm(pt, axis=-1), 1e-300)
    dth = np.arccos(np.clip(cos, -1, 1))
    rp = np.linalg.norm(p0, axis=-1) / np.maximum(np.linalg.norm(pt, axis=-1), 1e-12)
    edges = np.quantile(e0, np.linspace(0, 1, 6))
    edges[0], edges[-1] = 0.0, np.inf
    b = np.clip(np.searchsorted(edges, e0) - 1, 0, 4)
    sig_th = np.array([np.quantile(dth[b == i], 0.68) / 1.515 for i in range(5)])      # Rayleigh 68% radius
    sig_p = np.array([np.diff(np.quantile(rp[b == i], [0.16, 0.84]))[0] / 2 for i in range(5)])
    return {'edges': edges, 'sig_theta': sig_th, 'sig_p_rel': sig_p, 'n_sides': int(side.sum())}


def refit(pi, nu, p0, sig_th, sig_p, iters=25):
    """pi, nu (n,4) lab (px,py,pz,E); p0 (n,3) reco pi0; sig_th (n,) rad; sig_p (n,) GeV.  -> pi0' (n,4), |g| final."""
    Pc = pi + nu
    u = p0 / np.linalg.norm(p0, axis=-1, keepdims=True)
    z = np.zeros_like(u)
    z[:, 2] = 1.0
    e1 = np.cross(u, z)
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-12)
    e2 = np.cross(u, e1)
    R = np.stack([u, e1, e2], 1)                                   # rows: local axes
    pm = np.linalg.norm(p0, axis=-1)
    ainv_local = np.stack([sig_p ** 2, (sig_th * pm) ** 2, (sig_th * pm) ** 2], 1)
    Ainv = np.einsum('nji,nj,njk->nik', R, ainv_local, R)          # A^-1 in lab
    d = np.zeros_like(p0)

    def g_and_J(d):
        p = p0 + d
        E = np.sqrt(np.sum(p * p, -1) + M_PI0 ** 2)
        tot_E, tot_p = Pc[:, 3] + E, Pc[:, :3] + p
        g = tot_E ** 2 - np.sum(tot_p ** 2, -1) - M_TAU ** 2
        J = 2 * tot_E[:, None] * p / E[:, None] - 2 * tot_p
        return g, J
    for _ in range(iters):
        g, J = g_and_J(d)
        AJ = np.einsum('nij,nj->ni', Ainv, J)
        lam = (np.sum(J * d, -1) - g) / np.maximum(np.sum(J * AJ, -1), 1e-300)
        d = AJ * lam[:, None]
    g, _ = g_and_J(d)
    p = p0 + d
    return np.concatenate([p, np.sqrt(np.sum(p * p, -1) + M_PI0 ** 2)[:, None]], -1), np.abs(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-events', type=int, default=40000)
    a = ap.parse_args()
    res = {}
    rs = train_resolutions(a.train_events, 20261018)
    res['resolutions'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rs.items()}
    print('resolutions', json.dumps(res['resolutions']), flush=True)

    V = np.load(s1.LADDER / 'validation.npz')
    vt = np.load(GEN / 'targets' / 'validation.npz')
    vv = vt['h_valid'].astype(bool)
    assert np.array_equal(vt['global_indices'][vv], V['global_indices'])
    hx, w, y, modes = vt['h'][vv].astype(float), vt['weights'][vv].astype(float), vt['labels'][vv].astype(int), vt['modes'][vv]
    pol = s1.install(s1.ns.load_polarimeter()[0])
    D = {k: V[k] for k in V.files}
    sel = D['reco_h_available_side'] & (D['reco_h_mode'] == 1)
    pi = D['reco_h_pions_lab4'][..., 0, :][sel]
    nu = D['truth_neutrino_lab4'][sel].copy()
    nu[:, 3] = np.linalg.norm(nu[:, :3], axis=-1)
    p0 = D['reco_h_neutral_lab4'][..., :3][sel]
    e0 = D['reco_h_neutral_lab4'][..., 3][sel]
    b = np.clip(np.searchsorted(rs['edges'], e0) - 1, 0, 4)
    sth, sp = rs['sig_theta'][b], rs['sig_p_rel'][b] * np.linalg.norm(p0, axis=-1)
    tot = pi + nu + np.concatenate([p0, np.sqrt(np.sum(p0 ** 2, -1) + M_PI0 ** 2)[:, None]], -1)
    shell = np.full(sel.shape, np.nan)
    shell[sel] = np.sqrt(np.maximum(tot[:, 3] ** 2 - np.sum(tot[:, :3] ** 2, -1), 0)) - M_TAU
    variants = {'reco': D['reco_h_neutral_lab4']}
    for name, spv in (('shell', sp), ('shell_dir', 1e-4 * np.ones_like(sp))):
        new, gres = refit(pi, nu, p0, sth, spv)
        n = D['reco_h_neutral_lab4'].copy()
        n[sel] = new
        variants[name] = n
        res[f'{name}_converged'] = float(np.mean(gres < 1e-4))
        print(name, 'converged (|g|<1e-4 GeV^2)', res[f'{name}_converged'], flush=True)
    H = {k: s1.evaluate(pol, D, D['reco_h_pions_lab4'], n) for k, n in variants.items()}
    L = np.load(a.run_dir / 'out' / 's1' / 'ladder_validation.npz')
    assert np.allclose(np.nan_to_num(H['reco']), np.nan_to_num(L['h_reco']))
    Dn = np.load(a.run_dir / 'out' / 's4' / 'denoised.npz')
    hp = [np.load(GEN / 'p11' / arm / 'validation_predictions.npz')['h_pred'].astype(float) for arm in ('full22_s42', 'full22_s43')]
    rows = {'formula reco': H['reco'], 'formula shell refit': H['shell'], 'formula shell refit (dir only)': H['shell_dir'],
            'formula truth pi0 dir': L['h_pi0dir'], 'denoised': Dn['h_validation'], 'learner full22': 0.5 * (hp[0] + hp[1])}
    q = L['rho_side'] & np.isfinite(hx).all(-1)
    for h in rows.values():
        q &= np.isfinite(h).all(-1)
    W = np.broadcast_to(w[:, None], q.shape)
    res['quality'] = {k: {c: wcorr(h[..., i][q], hx[..., i][q], W[q]) for i, c in enumerate('nrk')} for k, h in rows.items()}
    res['quality_sides'] = int(q.sum())
    masks = pair_masks(modes)
    res['auc'] = {k: {p: wauc(formula(h)[m], y[m], w[m]) for p, m in masks.items()} for k, h in rows.items() if k != 'learner full22'}
    sv = np.abs(shell[q])
    edges = np.quantile(sv, np.linspace(0, 1, 7))
    edges[-1] += 1e-9
    bb = np.clip(np.searchsorted(edges, sv, side='right') - 1, 0, 5)
    res['shell_bins'] = {'edges_GeV': edges.tolist(), 'rows': {k: [wcorr(h[..., 2][q][bb == i], hx[..., 2][q][bb == i], W[q][bb == i])
                                                                   for i in range(6)] for k, h in rows.items()}}
    a.output.write_text(json.dumps(res, indent=1))
    print('quality (sides', res['quality_sides'], ')')
    for k, d in res['quality'].items():
        print(f'  {k:32s}', ' '.join(f'{c}:{d[c]:.3f}' for c in 'nrk'))
    keys = ('pi x rho', 'rho x rho', 'rho x 3pi', 'overall')
    print('AUC formula readout', keys)
    for k, d in res['auc'].items():
        print(f'  {k:32s}', ' '.join(f'{d[p]:.4f}' for p in keys))
    print('h_k quality vs |shell violation|, edges', np.round(edges, 3).tolist())
    for k, v in res['shell_bins']['rows'].items():
        print(f'  {k:32s}', ' '.join(f'{x:.2f}' for x in v))


if __name__ == '__main__':
    main()
