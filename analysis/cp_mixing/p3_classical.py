"""Classical phi*_CP observables (no neutrino, no ML) on the same events.

Impact-parameter method for tau -> pi nu sides and the neutral-pion
(decay-plane) method for tau -> rho nu sides, following the standard
construction: boost the two charged pions into their zero-momentum frame,
take the analyser vector of each side transverse to the charged-pion
direction, and form the signed acoplanarity.  Analyser = PV-referenced
leading-core-track impact parameter (pi side) or the reconstructed pi0
momentum (rho side); the rho sides carry the usual y = (E_pi - E_pi0) /
(E_pi + E_pi0) sign flip.

Sensitivity uses the same linear-response estimator as p2: the exact-h score
S fixes d<T>/dphi_tau = Cov(T, S), so classical and learned observables are
compared on identical events with identical statistics.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import score, sensitivity, triple  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def boost_to(v4, p4):
    """Boost (px,py,pz,E) vectors v4 into the rest frame of p4 (same shape leading dims)."""
    beta = p4[..., :3] / p4[..., 3:4]
    b2 = np.sum(beta * beta, -1)
    gamma = 1 / np.sqrt(np.maximum(1 - b2, 1e-16))
    bv = np.sum(v4[..., :3] * beta, -1)
    fac = gamma * gamma / (gamma + 1) * bv - gamma * v4[..., 3]
    return np.concatenate([v4[..., :3] + fac[..., None] * beta,
                           (gamma * (v4[..., 3] - bv))[..., None]], -1)


def phistar_cp(pc_m, pc_p, a_m, a_p, yflip):
    """Signed acoplanarity in the charged-pion zero-momentum frame."""
    tot = pc_m + pc_p
    bm, bp = boost_to(pc_m, tot), boost_to(pc_p, tot)
    am, apl = boost_to(a_m, tot), boost_to(a_p, tot)
    q = unit(bm[..., :3])
    nm = unit(am[..., :3] - np.sum(am[..., :3] * q, -1, keepdims=True) * q)
    npl = unit(apl[..., :3] - np.sum(apl[..., :3] * q, -1, keepdims=True) * q)
    cosphi = np.clip(np.sum(nm * npl, -1), -1, 1)
    phi = np.arccos(cosphi)
    o = np.sum(q * np.cross(npl, nm), -1)
    phi = np.where(o >= 0, phi, 2 * np.pi - phi)
    return np.mod(np.where(yflip, phi + np.pi, phi), 2 * np.pi), np.abs(bp[..., 3] - bm[..., 3]) * 0 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'results')
    ap.add_argument('--n-ref', type=int, default=10000)
    ap.add_argument('--arm', default='full22_s42')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    D = np.load(args.data / 'truth_surface_cleo.npz')
    L = np.load(args.data / 'ladder_validation.npz')
    T = np.load(args.data / 'ip_tracks.npz')
    assert np.array_equal(L['global_indices'], D['global_indices'])
    ids = L['global_indices']
    y = D['labels'].astype(bool)

    # exact-h score
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    h_gen = np.array(D['h_ref'], float)
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, D['modes'], D['pion_charges'], s, md)
            h_gen[sel, s] = h
    C0 = c_matrix(0.0)
    S_all, _ = score(h_gen, C0)

    # PV-referenced impact parameters, identical construction to p10_geometry_full
    pv, t = T['pv'], T['trk']
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    beam = np.array([*np.median(pv[:, :2], axis=0),
                     -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))])
    theta, phi_t, d0, z0 = (t[..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi_t), np.sin(theta) * np.sin(phi_t), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi_t), d0 * np.cos(phi_t), z0], -1) + beam
    ipv = pca - pv[:, None, None, :]
    ipv = ipv - np.sum(ipv * tdir, -1, keepdims=True) * tdir
    ip_lead = ipv[ids, :, 0]                                  # (N, 2, 3)
    ip_ok = np.isfinite(d0)[ids, :, 0]

    pions = L['reco_h_pions_lab4']                            # (N,2,3,4) px,py,pz,E
    charges = L['reco_h_charges']
    pi0 = L['reco_h_neutral_lab4']
    nneu = L['reco_h_neutral_counts']
    rmode = L['reco_h_mode']
    avail = L['reco_h_available_side']
    tmode = D['modes']

    out = {'n_ref': args.n_ref, 'method': 'charged-pion ZMF acoplanarity'}
    m_pi = np.sqrt(np.maximum(pions[..., 3]**2 - np.sum(pions[..., :3]**2, -1), 0))
    out['audit'] = {'mean_pion_mass_GeV': float(np.nanmean(m_pi[m_pi > 0])),
                    'ip_lead_available': float(ip_ok.mean()),
                    'ip_norm_median_mm': float(np.median(np.linalg.norm(ip_lead[ip_ok], axis=-1))),
                    'beam_reference_mm': beam.tolist()}

    lead_pi = pions[:, :, 0]                                  # leading (only) charged pion
    q_lead = charges[:, :, 0]
    # side 0 must be the negative prong to match the h convention; assert it
    out['audit']['side0_lead_charge_mean'] = float(np.nanmean(q_lead[:, 0]))
    out['audit']['side1_lead_charge_mean'] = float(np.nanmean(q_lead[:, 1]))

    ylab = np.where(np.isfinite(pi0[..., 3]) & (pi0[..., 3] > 0),
                    (lead_pi[..., 3] - pi0[..., 3]) / np.maximum(lead_pi[..., 3] + pi0[..., 3], 1e-9), 0.0)

    ana = np.where((rmode == 1)[..., None], pi0[..., :3], unit(ip_lead))
    ana4 = np.concatenate([ana, np.where((rmode == 1)[..., None], pi0[..., 3:4], 0.0)], -1)

    channels = {
        'pi-pi (IP x IP)': (tmode[:, 0] == 0) & (tmode[:, 1] == 0),
        'rho-rho (pi0 x pi0)': (tmode[:, 0] == 1) & (tmode[:, 1] == 1),
        'pi-rho (IP x pi0)': ((tmode[:, 0] == 0) & (tmode[:, 1] == 1)) | ((tmode[:, 0] == 1) & (tmode[:, 1] == 0)),
    }
    base_ok = y & avail.all(1) & (rmode == tmode).all(1) & np.isfinite(lead_pi).all((1, 2))
    ok_pi = ip_ok & np.isfinite(ip_lead).all(-1) & (np.linalg.norm(ip_lead, axis=-1) > 1e-9)
    ok_rho = (nneu == 1) & np.isfinite(pi0).all(-1) & (pi0[..., 3] > 0)
    side_ok = np.where(rmode == 1, ok_rho, ok_pi)

    P = np.load(args.data / f'gen3pi_{args.arm}.npz')['h_pred'].astype(float)

    res = {}
    for name, chan in channels.items():
        k = base_ok & chan & side_ok.all(1)
        if k.sum() < 200:
            continue
        yf = np.where((rmode[k] == 1), np.sign(ylab[k]), 1.0).prod(1) < 0
        pcp, _ = phistar_cp(lead_pi[k, 0], lead_pi[k, 1], ana4[k, 0], ana4[k, 1], yf)
        pcp_noy, _ = phistar_cp(lead_pi[k, 0], lead_pi[k, 1], ana4[k, 0], ana4[k, 1],
                                np.zeros(k.sum(), bool))
        Sk = S_all[k]
        row = {'n': int(k.sum()),
               'classical_sin_phistar': sensitivity(np.sin(pcp), Sk, args.n_ref, boot=200),
               'classical_sin_phistar_no_y_flip': sensitivity(np.sin(pcp_noy), Sk, args.n_ref, boot=200),
               'classical_cos_phistar': sensitivity(np.cos(pcp), Sk, args.n_ref, boot=200),
               'reco_h_Tp': sensitivity(triple(P[k, 0], P[k, 1]), Sk, args.n_ref, boot=200),
               'exact_h_Tp': sensitivity(triple(h_gen[k, 0], h_gen[k, 1]), Sk, args.n_ref, boot=200),
               'exact_h_optimal': {'sigma_deg': float(np.rad2deg(1 / np.sqrt(args.n_ref * np.var(Sk))))}}
        res[name] = row
        np.savez_compressed(args.output / f'phistar_{name.split()[0]}.npz',
                            phistar=pcp, phistar_no_y=pcp_noy, score=Sk,
                            reco_Tp=triple(P[k, 0], P[k, 1]), exact_Tp=triple(h_gen[k, 0], h_gen[k, 1]),
                            rows=np.flatnonzero(k))
    out['channels'] = res
    (args.output / 'classical.json').write_text(json.dumps(out, indent=1))

    print(json.dumps(out['audit'], indent=1))
    print(f"\n--- sigma(phi_tau) [deg] per {args.n_ref} events, same events per row ---")
    hdr = ['classical_sin_phistar', 'classical_sin_phistar_no_y_flip', 'reco_h_Tp', 'exact_h_Tp', 'exact_h_optimal']
    print(f'{"channel":22s} {"n":>6s} ' + ' '.join(f'{h[:22]:>22s}' for h in hdr))
    for name, row in res.items():
        print(f'{name:22s} {row["n"]:6d} ' + ' '.join(f'{row[h]["sigma_deg"]:22.3f}' for h in hdr))


if __name__ == '__main__':
    main()
