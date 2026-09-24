"""Classical phi*_CP, done as well as the visible decay products allow.

Adds to p3:
  * truth-input version of the same observables (truth pi+-, truth pi0, truth
    tau direction as the ideal impact parameter) -> separates the intrinsic
    dilution of the method from detector resolution;
  * |y1 y2| event weighting, the standard amplitude enhancement;
  * binned Asimov Delta chi^2 over the full phi*_CP shape rather than <sin>;
  * an explicit reweighted finite-difference cross-check of the linear
    response for one classical observable.
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
from cp_tools import score, sensitivity, triple, weights  # noqa: E402
from p3_classical import boost_to, phistar_cp, unit  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402


def asimov_sigma(x, S, n_ref, bins=24, rng=(0, 2 * np.pi), probe_deg=1.0):
    """sigma(phi_tau) from the binned shape, Gaussian Asimov, linearised at 0."""
    d = np.deg2rad(probe_deg)
    h0, _ = np.histogram(x, bins=bins, range=rng)
    h1, _ = np.histogram(x, bins=bins, range=rng, weights=1.0 + d * S)
    s = n_ref / len(x)
    h0f, h1f = h0 * s, h1 * s * (h0.sum() / max(h1.sum(), 1e-9))
    k = h0f > 0
    chi2 = float(np.sum((h1f[k] - h0f[k])**2 / h0f[k]))
    return float(np.rad2deg(d / np.sqrt(chi2))) if chi2 > 0 else float('inf')


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
    ids = L['global_indices']
    y = D['labels'].astype(bool)
    tmode = D['modes']

    fr = frames(D['pions'], D['pi0'], D['nu4'])
    h_gen = np.array(D['h_ref'], float)
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, tmode, D['pion_charges'], s, md)
            h_gen[sel, s] = h
    C0 = c_matrix(0.0)
    S_all, _ = score(h_gen, C0)

    # --- reco analysers ---
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
    ip_lead = ipv[ids, :, 0]
    ip_ok = np.isfinite(d0)[ids, :, 0] & np.isfinite(ip_lead).all(-1) \
        & (np.linalg.norm(ip_lead, axis=-1) > 1e-9)

    pions_r = L['reco_h_pions_lab4'][:, :, 0]
    pi0_r = L['reco_h_neutral_lab4']
    rmode = L['reco_h_mode']
    avail = L['reco_h_available_side']
    nneu = L['reco_h_neutral_counts']
    ana_r = np.concatenate([np.where((rmode == 1)[..., None], pi0_r[..., :3], unit(ip_lead)),
                            np.where((rmode == 1)[..., None], pi0_r[..., 3:4], 0.0)], -1)
    y_r = np.where(np.isfinite(pi0_r[..., 3]) & (pi0_r[..., 3] > 0),
                   (pions_r[..., 3] - pi0_r[..., 3]) / np.maximum(pions_r[..., 3] + pi0_r[..., 3], 1e-9), 0.0)

    # --- truth analysers (ideal IP = tau direction perpendicular to the pion) ---
    pions_t = D['pions'][:, :, 0]
    pi0_t = D['pi0']
    tau_dir = unit((D['pions'].sum(2) + D['pi0'] + D['nu4'])[..., :3])
    pdir = unit(pions_t[..., :3])
    ip_t = unit(tau_dir - np.sum(tau_dir * pdir, -1, keepdims=True) * pdir)
    ana_t = np.concatenate([np.where((tmode == 1)[..., None], pi0_t[..., :3], ip_t),
                            np.where((tmode == 1)[..., None], pi0_t[..., 3:4], 0.0)], -1)
    y_t = np.where(tmode == 1,
                   (pions_t[..., 3] - pi0_t[..., 3]) / np.maximum(pions_t[..., 3] + pi0_t[..., 3], 1e-9), 0.0)

    P = np.load(args.data / f'gen3pi_{args.arm}.npz')['h_pred'].astype(float)

    channels = {
        'pi-pi': (tmode[:, 0] == 0) & (tmode[:, 1] == 0),
        'rho-rho': (tmode[:, 0] == 1) & (tmode[:, 1] == 1),
        'pi-rho': ((tmode[:, 0] == 0) & (tmode[:, 1] == 1)) | ((tmode[:, 0] == 1) & (tmode[:, 1] == 0)),
    }
    side_ok_r = np.where(rmode == 1, (nneu == 1) & np.isfinite(pi0_r).all(-1) & (pi0_r[..., 3] > 0), ip_ok)
    base_ok = y & avail.all(1) & (rmode == tmode).all(1) & np.isfinite(pions_r).all((1, 2)) & side_ok_r.all(1)

    out = {'n_ref': args.n_ref, 'arm': args.arm, 'channels': {}, 'fd_check': {}}
    store = {}
    for name, chan in channels.items():
        k = base_ok & chan
        if k.sum() < 200:
            continue
        Sk = S_all[k]
        yf_r = np.where(rmode[k] == 1, np.sign(y_r[k]), 1.0).prod(1) < 0
        yf_t = np.where(tmode[k] == 1, np.sign(y_t[k]), 1.0).prod(1) < 0
        pcp_r, _ = phistar_cp(pions_r[k, 0], pions_r[k, 1], ana_r[k, 0], ana_r[k, 1], yf_r)
        pcp_t, _ = phistar_cp(pions_t[k, 0], pions_t[k, 1], ana_t[k, 0], ana_t[k, 1], yf_t)
        wy_r = np.abs(np.where(rmode[k] == 1, y_r[k], 1.0)).prod(1)
        wy_t = np.abs(np.where(tmode[k] == 1, y_t[k], 1.0)).prod(1)
        row = {'n': int(k.sum())}
        row['classical_reco_sin'] = sensitivity(np.sin(pcp_r), Sk, args.n_ref, boot=200)
        row['classical_reco_sin_yweighted'] = sensitivity(wy_r * np.sin(pcp_r), Sk, args.n_ref, boot=200)
        row['classical_reco_asimov_sigma_deg'] = asimov_sigma(pcp_r, Sk, args.n_ref)
        row['classical_truth_sin'] = sensitivity(np.sin(pcp_t), Sk, args.n_ref, boot=200)
        row['classical_truth_sin_yweighted'] = sensitivity(wy_t * np.sin(pcp_t), Sk, args.n_ref, boot=200)
        row['classical_truth_asimov_sigma_deg'] = asimov_sigma(pcp_t, Sk, args.n_ref)
        row['learned_reco_h_Tp'] = sensitivity(triple(P[k, 0], P[k, 1]), Sk, args.n_ref, boot=200)
        row['exact_h_Tp'] = sensitivity(triple(h_gen[k, 0], h_gen[k, 1]), Sk, args.n_ref, boot=200)
        row['exact_h_optimal_sigma_deg'] = float(np.rad2deg(1 / np.sqrt(args.n_ref * np.var(Sk))))
        out['channels'][name] = row
        store[name] = dict(phistar_reco=pcp_r, phistar_truth=pcp_t, score=Sk,
                           reco_Tp=triple(P[k, 0], P[k, 1]), exact_Tp=triple(h_gen[k, 0], h_gen[k, 1]),
                           rows=np.flatnonzero(k), wy_reco=wy_r)

        if name == 'rho-rho':
            for deg in (2.0, 5.0):
                d = np.deg2rad(deg)
                wp, wm = weights(h_gen[k], d, C0), weights(h_gen[k], -d, C0)
                fd = (np.average(np.sin(pcp_r), weights=wp) - np.average(np.sin(pcp_r), weights=wm)) / (2 * d)
                out['fd_check'][f'rho-rho classical sin, +-{deg:g}deg'] = \
                    {'finite_difference': float(fd), 'cov_response': row['classical_reco_sin']['response']}

    np.savez_compressed(args.output / 'classical_arrays.npz',
                        **{f'{c}__{k}': v for c, d in store.items() for k, v in d.items()})
    (args.output / 'classical_full.json').write_text(json.dumps(out, indent=1))

    keys = ['classical_reco_sin', 'classical_reco_sin_yweighted', 'classical_truth_sin',
            'classical_truth_sin_yweighted', 'learned_reco_h_Tp', 'exact_h_Tp']
    print(f'\n--- sigma(phi_tau) [deg] per {args.n_ref} events ---')
    print(f'{"channel":9s} {"n":>6s} ' + ' '.join(f'{k.replace("classical_","cl_").replace("_sin","")[:16]:>17s}' for k in keys)
          + f' {"cl_reco_asimov":>15s} {"cl_truth_asimov":>16s} {"optimal":>8s}')
    for name, r in out['channels'].items():
        print(f'{name:9s} {r["n"]:6d} ' + ' '.join(f'{r[k]["sigma_deg"]:17.3f}' for k in keys)
              + f' {r["classical_reco_asimov_sigma_deg"]:15.3f} {r["classical_truth_asimov_sigma_deg"]:16.3f}'
              f' {r["exact_h_optimal_sigma_deg"]:8.3f}')
    print('\nfinite-difference cross-check:', json.dumps(out['fd_check'], indent=1))


if __name__ == '__main__':
    main()
