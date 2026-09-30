"""Shared pieces: event loading, exact h, frames, predicted and measured ditau states.

Conventions
-----------
* Four-vectors (px, py, pz, E) as in polarimeter.frames.
* All angles live in the tau-pair rest frame reached by the pure boost of
  polarimeter.frames (= Z rest frame for pp -> Z j).
* k = tau+ direction there.  A basis is (n, r, k) with n = ref x k and
  r = ref_perp for a reference axis ref (helicity_basis).
* h_cart: the canonical h rotated back to pair-frame Cartesian axes, so it can
  be projected on any (n, r, k).  Canonical = -physical, hence physical
  B = -3 <h>, and C = 9 <h- h+^T> (sign-free).
"""
import numpy as np

import polarimeter as pol
import ztheory as zt


def load(paths):
    parts = [np.load(p) for p in paths]
    keys = ('w', 'beam_id', 'beam', 'jet_id', 'jet', 'mode', 'pions', 'charges', 'pi0', 'nu')
    return {k: np.concatenate([d[k] for d in parts]) for k in keys}


def exact_h(ev):
    """canonical exact h (generator current for 3pi) in pair-frame Cartesian axes."""
    fr = pol.frames(ev['pions'], ev['pi0'], ev['nu'])
    n = len(ev['mode'])
    h_can = np.full((n, 2, 3), np.nan)
    for side in (0, 1):
        for mode in (0, 1, 2):
            sel, h = pol.canonical_h(fr, ev['mode'], ev['charges'], side, mode)
            h_can[sel, side] = h
    h_cart = np.einsum('nia,nsi->nsa', fr['basis'], h_can)   # basis rows (n,r,k) -> Cartesian
    return fr, h_can, h_cart


def pair_frame_vectors(ev, fr):
    """tau+ direction k, the two beam directions and the jet in the pair frame."""
    beta = fr['beta']
    b = pol.boost(ev['beam'], beta[:, None, :])
    jet = pol.boost(ev['jet'], beta)
    tau = ev['pions'].sum(2) + ev['pi0'] + ev['nu']
    tc = pol.boost(tau, beta[:, None, :])
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    return dict(k=unit(tc[:, 1, :3]), b1=unit(b[:, 0, :3]), b2=unit(b[:, 1, :3]),
                jet=unit(jet[:, :3]), pair=tau.sum(1))


def collins_soper(ev, fr, pv):
    """CS axes (rows x, y, z) in the pair frame; z along the pair rapidity direction."""
    p1, p2 = pv['b1'], pv['b2']
    # orient: 'plus' beam is the one moving along +z in the lab
    plus_is_1 = (ev['beam'][:, 0, 2] > 0)[:, None]
    bp = np.where(plus_is_1, p1, p2)
    bm = np.where(plus_is_1, p2, p1)
    z = bp - bm
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    sgn = np.sign(pv['pair'][:, 2])[:, None]            # pair longitudinal momentum sign
    sgn[sgn == 0] = 1
    z = z * sgn
    y = np.cross(bp, -bm)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    y = y * sgn
    x = np.cross(y, z)
    return np.stack((x, y, z), -2)


def to_frame(v, F):
    """components of pair-frame vectors v (..., 3) along rows of F (..., 3, 3)."""
    return np.einsum('...ij,...j->...i', F, v)


def measured_state(hm, hp, w=None):
    """physical B-, B+, C from canonical h projected on a per-event basis."""
    w = np.ones(len(hm)) if w is None else w
    W = w.sum()
    Bm = -3 * np.einsum('n,na->a', w, hm) / W
    Bp = -3 * np.einsum('n,na->a', w, hp) / W
    C = 9 * np.einsum('n,na,nb->ab', w, hm, hp) / W
    return Bm, Bp, C


def measured_state_err(hm, hp):
    n = len(hm)
    eBm = 3 * hm.std(0) / np.sqrt(n)
    eBp = 3 * hp.std(0) / np.sqrt(n)
    eC = 9 * np.einsum('na,nb->nab', hm, hp).std(0) / np.sqrt(n)
    return eBm, eBp, eC


def predicted_states(zdec, R_frame, O_in_frame):
    """per-event (N, B-, B+, C) for a Z density R (given in frame F) and bases O (rows in F)."""
    return zdec.state(zt.rotate(R_frame, O_in_frame))


def fit_R_from_directions(k_in_frame, A, basis, w=None):
    w = np.ones(len(k_in_frame)) if w is None else w
    W = w.sum()
    mk = np.einsum('n,ni->i', w, k_in_frame) / W
    mkk = np.einsum('n,ni,nj->ij', w, k_in_frame, k_in_frame) / W
    return zt.fit_R(mk, mkk, A, basis)
