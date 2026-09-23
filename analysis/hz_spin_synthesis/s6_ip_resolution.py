"""Measured IP error structure, TRAIN split only.

For every core track the true impact-parameter vector lies along the component
of the truth tau direction perpendicular to the track (the decay point is on the
tau line), so the measured PV-referenced IP component along
e_perp = t x unit(perp_t(tau_truth)) is pure measurement error (track + PV).
Its variance as a function of the orientation of e_perp between the transverse
(d0-like) and longitudinal (z0-like) axes of the track frame gives
Var = s_T^2 cos^2 + s_L^2 sin^2, fitted per (pT, |eta|) bin by least squares.

Also records the |IP| scale relative to the expected L sin(alpha) with
L ~ Exp(beta gamma c tau), for the magnitude part of an IP likelihood.
Output: json with s_T, s_L tables and summary numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1/train.npz')
TRACKS = Path('/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz')
OUT = Path('/home/rbaba/hz-spin-synthesis-20260923/artifacts/s6_ip_resolution.json')
PT_EDGES = np.array([0, 5, 10, 20, 40, 80, 1e9])
ETA_EDGES = np.array([0, 0.8, 1.5, 2.6])
CTAU_UM = 87.03
M_TAU = 1.77686


def main():
    tr = np.load(LADDER)
    ids = np.asarray(tr['global_indices'])
    tau = tr['truth_visible_tau_lab4'] + tr['truth_neutrino_lab4']
    tdir_truth = tau[..., :3] / np.linalg.norm(tau[..., :3], axis=-1, keepdims=True)
    p_tau = np.linalg.norm(tau[..., :3], axis=-1)
    tk = np.load(TRACKS)
    pv, trk, n_core = tk['pv'], tk['trk'], tk['n_core']
    ok_all = np.isfinite(trk[:, :, 0, 0])
    bxy = np.median(pv[:, :2], 0)
    bz = -float(np.median((trk[:, :, 0, 5] - pv[:, None, 2])[ok_all]))
    rows = []
    for j in range(3):
        t = trk[ids][:, :, j]
        pt, theta, phi, d0, z0 = (t[..., i] for i in (0, 1, 2, 4, 5))
        avail = np.isfinite(d0) & (n_core[ids] > j)
        tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
        pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + np.array([bxy[0], bxy[1], bz])
        b = pca - pv[ids][:, None, :]
        b -= np.sum(b * tdir, -1, keepdims=True) * tdir
        # track frame: eT transverse (perp to track and to z), eL = t x eT
        eT = np.cross(np.broadcast_to([0.0, 0.0, 1.0], tdir.shape), tdir)
        eT /= np.linalg.norm(eT, axis=-1, keepdims=True)
        eL = np.cross(tdir, eT)
        perp = tdir_truth - np.sum(tdir_truth * tdir, -1, keepdims=True) * tdir
        sin_a = np.linalg.norm(perp, axis=-1)
        e_par = perp / np.maximum(sin_a[..., None], 1e-12)
        e_perp = np.cross(tdir, e_par)
        err = np.sum(b * e_perp, -1) * 1e3          # um
        along = np.sum(b * e_par, -1) * 1e3         # um, = L sin(a) + error
        c = np.sum(e_perp * eT, -1)
        s = np.sum(e_perp * eL, -1)
        lam = p_tau / M_TAU * CTAU_UM               # mean decay length [um]
        good = avail & np.isfinite(err) & (np.abs(err) < 2000)
        rows.append(dict(j=j, pt=pt[good], eta=np.abs(-np.log(np.tan(theta[good] / 2))), err=err[good],
                         c2=c[good] ** 2, s2=s[good] ** 2, along=along[good], exp_along=(lam * sin_a)[good]))
    out = {'bins_pt': PT_EDGES.tolist(), 'bins_eta': ETA_EDGES.tolist(), 'sigma_T_um': [], 'sigma_L_um': [],
           'counts': []}
    pt = np.concatenate([r['pt'] for r in rows])
    eta = np.concatenate([r['eta'] for r in rows])
    err = np.concatenate([r['err'] for r in rows])
    c2 = np.concatenate([r['c2'] for r in rows])
    s2 = np.concatenate([r['s2'] for r in rows])
    for i in range(len(PT_EDGES) - 1):
        rT, rL, rN = [], [], []
        for k in range(len(ETA_EDGES) - 1):
            sel = (pt >= PT_EDGES[i]) & (pt < PT_EDGES[i + 1]) & (eta >= ETA_EDGES[k]) & (eta < ETA_EDGES[k + 1])
            if sel.sum() < 200:
                rT.append(None); rL.append(None); rN.append(int(sel.sum())); continue
            e = err[sel]
            # robust: clip at 5 x MAD-sigma before the fit
            scale = 1.4826 * np.median(np.abs(e)) + 1e-9
            keep = np.abs(e) < 5 * scale
            X = np.stack([c2[sel][keep], s2[sel][keep]], 1)
            coef, *_ = np.linalg.lstsq(X, e[keep] ** 2, rcond=None)
            rT.append(float(np.sqrt(max(coef[0], 1.0)))); rL.append(float(np.sqrt(max(coef[1], 1.0)))); rN.append(int(sel.sum()))
        out['sigma_T_um'].append(rT); out['sigma_L_um'].append(rL); out['counts'].append(rN)
    lead = rows[0]
    ratio = lead['along'] / np.maximum(lead['exp_along'], 1e-9)
    out['lead_along_over_expected_quantiles'] = np.quantile(ratio, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    out['lead_expected_along_um_quantiles'] = np.quantile(lead['exp_along'], [0.1, 0.5, 0.9]).tolist()
    out['lead_abs_err_um_quantiles'] = np.quantile(np.abs(lead['err']), [0.5, 0.68, 0.9, 0.99]).tolist()
    out['frac_err_tail_gt_5sigma'] = float(np.mean(np.abs(err) > 5 * 1.4826 * np.median(np.abs(err))))
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
