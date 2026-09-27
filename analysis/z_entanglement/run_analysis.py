"""v3 Z -> tau tau: tomography with exact h vs the prediction from tau directions alone.

usage: python run_analysis.py OUTDIR data1.npz [data2.npz ...]

The Z spin density <R> is measured in Collins-Soper axes from the tau+
direction moments only (per bin of pair qT, so every event gets the <R> of its
own production class).  From it ztheory predicts each event's (B-, B+, C) in
any (n, r, k) basis.  The exact h (generator current for 3pi) measures the same
state with C = 9<h- h+^T>, B = -3<h> (canonical sign).  No selection => both
are unbiased; they use independent information (tau direction vs tau decay).
"""
import json
import sys
from pathlib import Path

import numpy as np

import ztheory as zt
import zent

QT_EDGES = np.array([0, 150, 200, 250, 300, 400, 1e9])
CT_EDGES = np.linspace(0, 1, 11)                 # |cos theta_CS|
NBOOT = 200
rng = np.random.default_rng(20260927)


def state_quantities(Bm, Bp, C):
    q = zt.quantum_summary(Bm, Bp, C)
    q['d_nr'] = float(C[0, 0] - C[1, 1])
    return q


def bootstrap(hm, hp, idx_sets):
    """bootstrap concurrence etc. for event index sets."""
    out = []
    for idx in idx_sets:
        vals = []
        for _ in range(NBOOT):
            s = rng.choice(idx, idx.size)
            vals.append(state_quantities(*zent.measured_state(hm[s], hp[s])))
        out.append({k: float(np.std([v[k] for v in vals])) for k in vals[0]})
    return out


def main(outdir, paths):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ev = zent.load(paths)
    n = len(ev['mode'])
    fr, h_can, h_cart = zent.exact_h(ev)
    hn = np.linalg.norm(h_cart, axis=-1)
    good = np.all(np.isfinite(hn), axis=1) & np.all(np.abs(hn - 1) < 1e-4, axis=1)
    res = dict(n_events=int(n), n_good_h=int(good.sum()),
               h_norm_max_dev=float(np.nanmax(np.abs(hn - 1))),
               weights=dict(unique=np.unique(np.round(ev['w'], 8)).tolist()[:5],
                            n_negative=int((ev['w'] < 0).sum())))
    pv = zent.pair_frame_vectors(ev, fr)
    CS = zent.collins_soper(ev, fr, pv)
    k = pv['k']
    k_cs = zent.to_frame(k, CS)
    qt = np.hypot(pv['pair'][:, 0], pv['pair'][:, 1])
    mtt = np.sqrt(np.maximum(pv['pair'][:, 3]**2 - np.sum(pv['pair'][:, :3]**2, -1), 0))
    res['qt_quantiles'] = np.quantile(qt, [.01, .16, .5, .84, .99]).round(1).tolist()
    res['mtt_quantiles'] = np.quantile(mtt, [.01, .16, .5, .84, .99]).round(3).tolist()
    res['mode_pair_counts'] = {f'{a}{b}': int(np.sum((ev['mode'][:, 0] == a) & (ev['mode'][:, 1] == b)))
                               for a in range(3) for b in range(3)}

    zdec = zt.ZDecay()
    A, basis = zt.direction_moment_map(zdec, 120)

    # train/test halves: <R> is fitted on one half and tested on the other
    half = rng.random(n) < 0.5
    qbin = np.digitize(qt, QT_EDGES) - 1
    R_bins = {}
    for b in range(len(QT_EDGES) - 1):
        m = good & half & (qbin == b)
        R_bins[b] = zent.fit_R_from_directions(k_cs[m], A, basis)
    res['R_cs_by_qt'] = {f'{QT_EDGES[b]:.0f}-{QT_EDGES[b + 1]:.0f}':
                         dict(re=np.real(R).round(4).tolist(), im=np.imag(R).round(4).tolist(),
                              n_fit=int(np.sum(good & half & (qbin == b))))
                         for b, R in R_bins.items()}
    R_ev = np.stack([R_bins[b] for b in qbin])

    # bases (rows n, r, k in pair-frame Cartesian)
    zcs = CS[:, 2]
    bases = {
        'canonical_lab_beam': fr['basis'],
        # the -z beam boosted into the pair frame (as in the 2026-09-21 basis check)
        'boosted_beam': zt.helicity_basis(k, np.where((ev['beam'][:, 0, 2] > 0)[:, None], pv['b2'], pv['b1'])),
        'cs_z': zt.helicity_basis(k, zcs),
    }
    # adaptive: rotate the CS-referenced basis about k so the predicted
    # transverse block is diagonal with C_nn >= C_rr (uses tau directions only)
    O_cs_in_cs = np.einsum('nai,nji->naj', bases['cs_z'], CS)        # basis rows in CS coords
    _, _, _, Cp = zdec.state(zt.rotate(R_ev, O_cs_in_cs))
    alpha = 0.5 * np.arctan2(Cp[:, 0, 1] + Cp[:, 1, 0], Cp[:, 0, 0] - Cp[:, 1, 1])
    ca, sa = np.cos(alpha), np.sin(alpha)
    nb, rb, kb = bases['cs_z'][:, 0], bases['cs_z'][:, 1], bases['cs_z'][:, 2]
    bases['adaptive'] = np.stack((ca[:, None] * nb + sa[:, None] * rb,
                                  -sa[:, None] * nb + ca[:, None] * rb, kb), -2)

    ct = np.abs(k_cs[:, 2])
    cbin = np.minimum(np.digitize(ct, CT_EDGES) - 1, len(CT_EDGES) - 2)
    test = good & ~half
    res['bases'] = {}
    arrays = dict(ct=ct, qt=qt, mtt=mtt, test=test, good=good, mode=ev['mode'], k_cs=k_cs,
                  phi_cs=np.arctan2(k_cs[:, 1], k_cs[:, 0]))
    for name, O in bases.items():
        hm = np.einsum('nab,nb->na', O, h_cart[:, 0])
        hp = np.einsum('nab,nb->na', O, h_cart[:, 1])
        O_in_cs = np.einsum('nai,nji->naj', O, CS)
        _, Bm_p, Bp_p, C_p = zdec.state(zt.rotate(R_ev, O_in_cs))
        arrays[f'{name}_hm'] = hm
        arrays[f'{name}_hp'] = hp
        arrays[f'{name}_Cpred'] = C_p
        arrays[f'{name}_Bpred'] = np.stack((Bm_p, Bp_p), 1)
        rows = []
        idx_sets = [np.flatnonzero(test)] + [np.flatnonzero(test & (cbin == b)) for b in range(len(CT_EDGES) - 1)]
        errs = bootstrap(hm, hp, idx_sets)
        for j, idx in enumerate(idx_sets):
            Bm, Bp, C = zent.measured_state(hm[idx], hp[idx])
            eBm, eBp, eC = zent.measured_state_err(hm[idx], hp[idx])
            Cpm, Bmpm, Bppm = C_p[idx].mean(0), Bm_p[idx].mean(0), Bp_p[idx].mean(0)
            pull = (C - Cpm) / eC
            rows.append(dict(
                bin='all' if j == 0 else [CT_EDGES[j - 1], CT_EDGES[j]], n=int(idx.size),
                C_meas=C.round(4).tolist(), C_err=eC.round(4).tolist(), C_pred=Cpm.round(4).tolist(),
                Bm_meas=Bm.round(4).tolist(), Bm_err=eBm.round(4).tolist(), Bm_pred=Bmpm.round(4).tolist(),
                Bp_meas=Bp.round(4).tolist(), Bp_err=eBp.round(4).tolist(), Bp_pred=Bppm.round(4).tolist(),
                chi2_C=float(np.sum(pull**2)), max_pull_C=float(np.max(np.abs(pull))),
                q_meas=state_quantities(Bm, Bp, C), q_meas_err=errs[j],
                q_pred=state_quantities(Bmpm, Bppm, Cpm)))
        res['bases'][name] = rows
    # mode-pair split of the measured transverse block in the adaptive basis
    O = bases['adaptive']
    hm, hp = arrays['adaptive_hm'], arrays['adaptive_hp']
    res['mode_pairs_adaptive'] = {}
    for a in range(3):
        for b in range(3):
            idx = np.flatnonzero(test & (ev['mode'][:, 0] == a) & (ev['mode'][:, 1] == b))
            if idx.size < 500:
                continue
            _, _, C = zent.measured_state(hm[idx], hp[idx])
            _, _, eC = zent.measured_state_err(hm[idx], hp[idx])
            Cpm = arrays['adaptive_Cpred'][idx].mean(0)
            res['mode_pairs_adaptive'][f'{a}{b}'] = dict(
                n=int(idx.size), diag_meas=np.diag(C).round(4).tolist(), diag_err=np.diag(eC).round(4).tolist(),
                diag_pred=np.diag(Cpm).round(4).tolist())
    np.savez_compressed(outdir / 'arrays.npz', **arrays)
    (outdir / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('n_events', 'n_good_h', 'h_norm_max_dev', 'weights',
                                           'qt_quantiles', 'mtt_quantiles', 'mode_pair_counts')}, indent=1))
    for name, rows in res['bases'].items():
        r = rows[0]
        print(name, 'all: C_meas diag', np.diag(r['C_meas']), 'pred', np.diag(r['C_pred']),
              'conc meas %.4f +- %.4f pred %.4f' % (r['q_meas']['concurrence'], r['q_meas_err']['concurrence'],
                                                   r['q_pred']['concurrence']), 'chi2 %.1f' % r['chi2_C'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
