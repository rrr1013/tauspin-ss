"""Where does the instrumental correlation of the two predicted polarimeters come from?

Measured on Z events, whose generated transverse spin correlation is zero, so no
reweighting is needed and anything seen here is spin-independent.

Three questions:
  1. Is the transverse part of the instrumental correlation an alignment, i.e.
     do the two predicted transverse vectors share an azimuth?
  2. Is the 3x3 instrumental block rank-1 (one fixed direction) or isotropic in
     the transverse plane (a common *event-dependent* direction)?
  3. If it is event-dependent, is it one of the reconstructed directions the
     network can see -- the missing transverse momentum, the visible tau
     directions, their sum or difference -- projected into the truth (n, r)
     plane of that event?

Only the *transverse* block is instrumental on Z: the generated Z state has
C_nn = C_rr = 0 but C_kk = +1, so the kk element of the raw Z correlation is
real spin correlation and is excluded from every instrumental statement below.
"""
import json
import sys
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import DATA, load_arm, load_surface

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
from polarimeter import frames  # noqa: E402

ARMS = ['base_s43', 'full22_s42', 'idealip22_s42']


def transverse(v, basis):
    """(v.n, v.r) for a lab 3-vector and the per-event (n, r, k) basis."""
    return np.stack([np.sum(v * basis[:, 0], -1), np.sum(v * basis[:, 1], -1)], -1)


def unit(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def main():
    S = load_surface()
    y = S['labels']
    D = np.load(DATA / 'truth_surface_cleo.npz')
    L = np.load(DATA / 'ladder_validation.npz')
    assert np.array_equal(L['global_indices'], S['global_indices'])
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    basis = fr['basis'][~y]                      # Z events only

    met = L['reco_met_xy'][~y]
    met3 = np.concatenate([met, np.zeros((len(met), 1))], -1)
    vis = L['reco_visible_tau_lab4'][~y][..., :3]
    cands = {
        'MET': transverse(met3, basis),
        'visible tau-': transverse(vis[:, 0], basis),
        'visible tau+': transverse(vis[:, 1], basis),
        'visible difference': transverse(vis[:, 0] - vis[:, 1], basis),
        'visible sum': transverse(vis[:, 0] + vis[:, 1], basis),
    }
    out = {'cohort': 'Z validation events', 'n': int((~y).sum())}

    rows = {}
    for arm in ARMS:
        A = load_arm(arm, S)
        hp = A['h_pred'][~y]
        g = E.features(hp)
        Aout = g.mean(0)[6:].reshape(3, 3)
        sv = np.linalg.svd(Aout, compute_uv=False)
        t_m, t_p = hp[:, 0, :2], hp[:, 1, :2]
        dphi = np.mod(np.arctan2(t_m[:, 1], t_m[:, 0])
                      - np.arctan2(t_p[:, 1], t_p[:, 0]), 2 * np.pi)
        # alignment strength independent of magnitude
        cos_dphi = np.sum(unit(t_m) * unit(t_p), -1)
        # is the transverse block isotropic?
        Tblock = Aout[:2, :2]
        row = {'A_outer': Aout.tolist(), 'singular_values': sv.tolist(),
               'transverse_block': Tblock.tolist(),
               'transverse_trace': float(np.trace(Tblock)),
               'transverse_antisym': float(Tblock[0, 1] - Tblock[1, 0]),
               'transverse_traceless_norm': float(np.linalg.norm(
                   Tblock - np.trace(Tblock) / 2 * np.eye(2))),
               'mean_cos_dphi': float(cos_dphi.mean()),
               'mean_cos_dphi_sd': float(cos_dphi.std() / np.sqrt(len(cos_dphi))),
               'candidates': {}}
        # how much of the alignment is explained by each reco direction
        for nm, c in cands.items():
            u = unit(c)
            pm = np.sum(unit(t_m) * u, -1)
            pp = np.sum(unit(t_p) * u, -1)
            # residual alignment after removing the common projection on u
            # In two dimensions, projecting a unit vector out of another unit
            # vector and renormalising is degenerate, so there is no meaningful
            # "residual alignment".  The question a candidate axis can answer is
            # whether each prediction individually lines up with it: if the
            # common axis were u, both <u.h_-> and <u.h_+> would be large.
            row['candidates'][nm] = {
                'proj_tau_minus': float(pm.mean()),
                'proj_tau_minus_sd': float(pm.std() / np.sqrt(len(pm))),
                'proj_tau_plus': float(pp.mean()),
                'proj_tau_plus_sd': float(pp.std() / np.sqrt(len(pp))),
            }
        rows[arm] = row
    out['arms'] = rows

    # exact h control on the same events
    hx = S['h_gen'][~y]
    t_m, t_p = hx[:, 0, :2], hx[:, 1, :2]
    cos_dphi = np.sum(unit(t_m) * unit(t_p), -1)
    out['exact_h_control'] = {'mean_cos_dphi': float(cos_dphi.mean()),
                              'mean_cos_dphi_sd': float(cos_dphi.std()
                                                        / np.sqrt(len(cos_dphi)))}

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'origin.json').write_text(json.dumps(out, indent=1))

    print(f"Z validation events: {out['n']};  generated transverse spin correlation is zero")
    print(f"exact h control: <cos dphi> = {out['exact_h_control']['mean_cos_dphi']:+.4f}"
          f" +- {out['exact_h_control']['mean_cos_dphi_sd']:.4f}\n")
    for arm, r in rows.items():
        T = np.array(r['transverse_block'])
        print(f"{arm}")
        print(f"  raw <h_pred,- h_pred,+> on Z (n,r,k); only the 2x2 transverse block"
              " is instrumental:\n"
              + '\n'.join('    ' + ' '.join(f'{v:+.4f}' for v in row)
                          for row in np.array(r['A_outer'])))
        print("    (the kk element is the real Z spin correlation, C_kk = +1)")
        print(f"  transverse 2x2: trace {r['transverse_trace']:+.4f}, "
              f"traceless part {r['transverse_traceless_norm']:.4f}, "
              f"antisymmetric {r['transverse_antisym']:+.4f}"
              "   -> isotropic means a common *event-dependent* axis")
        print(f"  <cos dphi(h_pred-, h_pred+)> = {r['mean_cos_dphi']:+.4f}"
              f" +- {r['mean_cos_dphi_sd']:.4f}")
        print("    does either prediction line up with a reconstructed direction?")
        for nm, c in r['candidates'].items():
            print(f"    {nm:20s} <u.h_-> {c['proj_tau_minus']:+.4f}"
                  f"+-{c['proj_tau_minus_sd']:.4f}"
                  f"   <u.h_+> {c['proj_tau_plus']:+.4f}+-{c['proj_tau_plus_sd']:.4f}")
        print()


if __name__ == '__main__':
    main()
