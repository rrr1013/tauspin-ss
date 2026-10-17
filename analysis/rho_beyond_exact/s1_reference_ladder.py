"""Exact polarimeter of the reco visible system with the truth neutrinos, and its truth-replacement ladder.

For every event the generator-current HybridPolarimeter (as gen3pi teacher and the 10/17 'truthnu' row)
is evaluated on the reco visible system plus the truth neutrinos.  On validation, on sides that are
truth rho AND reco rho (reco_h_mode 1, available), the reco constituents are replaced one at a time:

  reco      reco pi+- and reco pi0 (PFO sum) -- the 10/17 reference
  pi0dir    reco pi0 rotated onto the truth pi0 direction (|p|, E kept)
  pi0E      reco pi0 four-vector scaled by E_truth / E_reco (direction, m/E kept)
  pi0       truth pi0 four-vector
  pich      truth charged pion, reco pi0
  both      truth pi+- and truth pi0 (closure to the exact h on those sides)

Other sides keep the reco visible system.  The same replacement is applied on both sides of an event,
so rho x rho events see it on both.  TRAIN gets the 'reco' row only (for the readout and the
denoised reference).  Test is never loaded.

Output: <out>/ladder_{train,validation}.npz with h (N,2,3) per variant (NaN where invalid) and ids.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
SURFACE = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/targets/truth_surface_gen3pi.npz')
NS_DIR = Path('/home/rbaba/met-nullspace-20260919/scripts')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(NS_DIR))
sys.path.insert(0, str(HERE.parent / 'gen3pi_teacher'))
import nullspace as ns  # noqa: E402
from generator_current import install  # noqa: E402

VARIANTS = ('reco', 'pi0dir', 'pi0E', 'pi0', 'pich', 'both')


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def evaluate(pol, D, pions, neutral, block=40000):
    n = len(pions)
    h = np.full((n, 2, 3), np.nan)
    for s in range(0, n, block):
        e = slice(s, s + block)
        out = pol(pions[e], D['reco_h_charges'][e], D['reco_h_counts'][e], neutral[e],
                  D['reco_h_neutral_counts'][e], D['reco_h_mode'][e].astype(np.int64),
                  D['truth_neutrino_lab4'][e][..., :3], availability=D['reco_h_available_side'][e])
        hh = out['h']
        hh[~out['valid_side']] = np.nan
        h[e] = hh
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pol = install(ns.load_polarimeter()[0])

    with np.load(LADDER / 'train.npz') as d:
        Tr = {k: d[k] for k in ('global_indices', 'reco_h_pions_lab4', 'reco_h_charges', 'reco_h_counts',
                                'reco_h_neutral_lab4', 'reco_h_neutral_counts', 'reco_h_mode',
                                'truth_neutrino_lab4', 'reco_h_available_side')}
    h_tr = evaluate(pol, Tr, Tr['reco_h_pions_lab4'], Tr['reco_h_neutral_lab4'])
    np.savez(a.output_dir / 'ladder_train.npz', global_indices=Tr['global_indices'], h_reco=h_tr)
    print(f'train done {time.time() - t0:.0f}s, valid sides {np.isfinite(h_tr).all(-1).mean():.4f}', flush=True)
    del Tr

    with np.load(LADDER / 'validation.npz') as d:
        V = {k: d[k] for k in d.files}
    with np.load(SURFACE) as d:
        S = {k: d[k] for k in d.files}
    assert np.array_equal(S['global_indices'], V['global_indices'])
    assert np.array_equal(S['modes'], V['modes'])

    rho = V['reco_h_available_side'] & (V['modes'] == 1) & (V['reco_h_mode'] == 1)
    rp, rn = V['reco_h_pions_lab4'], V['reco_h_neutral_lab4']
    tp, tn = S['pions'], S['pi0']
    # truth charged pion of a rho side is pions[...,0,:]; check the reco/truth charge agreement
    assert np.all(S['pion_counts'][rho] == 1) and np.all(V['reco_h_counts'][rho] == 1)
    qmatch = np.mean(np.sign(S['pion_charges'][..., 0][rho]) == np.sign(V['reco_h_charges'][..., 0][rho]))
    print(f'rho sides {rho.sum()}  charge agreement {qmatch:.5f}', flush=True)

    def with_neutral(new):
        out = rn.copy()
        out[rho] = new[rho]
        return out

    def with_pion(new):
        out = rp.copy()
        out[rho, 0] = new[rho]
        return out

    p_r, E_r = rn[..., :3], rn[..., 3]
    pmag = np.linalg.norm(p_r, axis=-1, keepdims=True)
    n_dir = np.concatenate([unit(tn[..., :3]) * pmag, E_r[..., None]], -1)
    n_E = rn * (tn[..., 3] / np.maximum(E_r, 1e-12))[..., None]
    inputs = {
        'reco': (rp, rn),
        'pi0dir': (rp, with_neutral(n_dir)),
        'pi0E': (rp, with_neutral(n_E)),
        'pi0': (rp, with_neutral(tn)),
        'pich': (with_pion(tp[..., 0, :]), rn),
        'both': (with_pion(tp[..., 0, :]), with_neutral(tn)),
    }
    res = {'global_indices': V['global_indices'], 'rho_side': rho}
    for name in VARIANTS:
        res[f'h_{name}'] = evaluate(pol, V, *inputs[name])
        print(f'{name} done {time.time() - t0:.0f}s', flush=True)
    # closure: 'both' must reproduce the surface h_ref (gen3pi exact h) where both sides are replaced
    # (the pair basis depends on both taus, so single-side replacement does not close)
    both = rho & rho.all(1, keepdims=True)
    hb, hx = res['h_both'][both], S['h_ref'][both].astype(float)
    ok = np.isfinite(hb).all(-1)
    print(f"closure both vs h_ref on rho sides: valid {ok.mean():.4f}, max |dh| {np.abs(hb[ok] - hx[ok]).max():.2e}",
          flush=True)
    np.savez(a.output_dir / 'ladder_validation.npz', **res)


if __name__ == '__main__':
    main()
