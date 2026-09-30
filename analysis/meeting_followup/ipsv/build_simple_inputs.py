"""Simple-smearing sample (Chen, R20 v2 chain): h targets and IP/SV geometry blocks.

For the h-valid train/validation rows of /home/rbaba/simple-smearing-20260910/data:
  * h_gen: canonical exact h with the 3pi sides from the generator (TauDecay UFO) current,
    same convention as the ATLAS targets of 2026-09-24 (gen3pi_teacher).  Sides are put in
    (tau-, tau+) order for the polarimeter, then mapped back to the pack sides; the pi and
    rho sides are a closure test against the stored R20 h_truth.
  * geometry blocks (22 channels per tau, appended to the tau token), identical layouts for
    the ATLAS sample (build_atlas_geometry.py):
      none           all zero
      legacy         IP as the ntuple stores it: d0/50um, z0/50mm (beam-spot z0), avail per
                     track (x3) + SV lab block
      lab            PV-referenced 3D IP in lab x,y,z /50um, avail per track (x3)
                     + SV lab block (PV->SV x,y,z /1mm, avail)
      local          hz_beyond_ceiling p10 v2 layout in the reco visible (n, r, k) frame:
                     per track (u.n, u.r, log(|IP|/50um), avail, t.n/10mrad, t.r/10mrad)
                     + SV (s.n/10mrad, s.r/10mrad, log(L/1mm), avail)
      local_shuffle  local permuted among same-split sides with the same reco mode
    The simple sample has one IP (leading charged track) per tau; tracks 2, 3 are unavailable.
    SV is given only to reco 3-prong taus, as in ATLAS (Chen's file also smears a vertex for
    1-prong taus, which a real vertex fit cannot provide).
  * resolution arrays per side for the resolution comparison figure.

usage: python build_simple_inputs.py OUT.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mode_pair_auc_origin'))
from polarimeter import canonical_h, frames  # noqa: E402

DATA = Path('/home/rbaba/simple-smearing-20260910/data')
ROOT = Path('/home/chen/data8/ZH_tautau/reco_training/training')
FILES = {1: ('h_pub_lhe.root', 'treeS'), 0: ('z_pub_lhe.root', 'treeB')}
SPLITS = {'train': 0, 'validation': 1}
G = 22


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def read_root(label, entries):
    fn, tree = FILES[label]
    keys = [f'{p}{t}_{c}' for p in ('ip',) for t in (1, 2) for c in 'xyz'] + \
           [f'sv_{c}{t}' for t in (1, 2) for c in 'xyz'] + [f'{v}_{t}' for v in ('d0', 'z0') for t in (1, 2)] + \
           [f'sv_{c}{t}_t' for t in (1, 2) for c in 'xyz'] + [f'pv_{c}_t' for c in 'xyz'] + \
           [f'tau{t}_pi1_{v}' for t in (1, 2) for v in ('pt', 'eta', 'phi')]
    with uproot.open(ROOT / fn) as f:
        a = f[tree].arrays(keys, library='np')
    return {k: v[entries] for k, v in a.items()}


def gen_h(truth_p4, nu, charges, digits):
    """canonical h per pack side, 3pi with the generator current; tau- / tau+ from the charges."""
    n = len(digits)
    qsum = np.where(digits == 3, charges.sum(-1), charges[..., 0])
    minus_first = qsum[:, 0] < 0
    order = np.where(minus_first[:, None], [0, 1], [1, 0])            # pack side of (tau-, tau+)
    take = lambda x: np.take_along_axis(x, order.reshape((n, 2) + (1,) * (x.ndim - 2)), axis=1)
    p4, nu_o, ch_o, dg = take(truth_p4), take(nu), take(charges), take(digits)
    pions = np.zeros((n, 2, 3, 4))
    pi0 = np.zeros((n, 2, 4))
    three = dg == 3
    pions[three] = p4[three]
    pions[~three, 0] = p4[~three][:, 0]
    rho = dg == 2
    pi0[rho] = p4[rho][:, 1]
    mode = np.select([dg == 1, dg == 2, dg == 3], [0, 1, 2], -1)
    fr = frames(pions, pi0, nu_o)
    h = np.full((n, 2, 3), np.nan)
    for side in (0, 1):
        for m in (0, 1, 2):
            sel, hv = canonical_h(fr, mode, np.sign(ch_o).astype(int), side, m)
            h[sel, side] = hv
    back = np.empty_like(h)
    rows = np.arange(n)
    back[rows, order[:, 0]] = h[:, 0]
    back[rows, order[:, 1]] = h[:, 1]
    return back


def blocks(ip, d0, z0, sv, sv_ok, tdir, basis, reco_digit, rng):
    """ip [n,2,3] (mm, PV ref), d0/z0 [n,2], sv [n,2,3] (mm, PV ref), tdir [n,2,3], basis [n,2,3,3]."""
    n = len(ip)
    out = {k: np.zeros((n, 2, G), np.float32) for k in ('none', 'legacy', 'lab', 'local')}
    ok = np.isfinite(ip).all(-1) & (np.linalg.norm(ip, axis=-1) > 0)
    svb = np.concatenate([np.clip(sv / 1.0, -50, 50), np.ones((n, 2, 1))], -1) * sv_ok[..., None]
    # track 0 only (simple sample has one IP per tau)
    out['legacy'][..., 0:3] = np.stack([np.clip(d0 / 0.05, -20, 20), np.clip(z0 / 50.0, -20, 20), np.ones((n, 2))], -1) * ok[..., None]
    out['lab'][..., 0:4] = np.concatenate([np.clip(ip / 0.05, -20, 20), np.ones((n, 2, 1))], -1) * ok[..., None]
    for k in ('legacy', 'lab'):
        out[k][..., 12:16] = svb
    u = unit(ip)
    nb, rb = basis[:, :, 0], basis[:, :, 1]
    cols = [np.sum(u * nb, -1), np.sum(u * rb, -1),
            np.clip(np.log(np.maximum(np.linalg.norm(ip, axis=-1), 1e-9) / 0.05), -4, 4), np.ones((n, 2)),
            np.clip(np.sum(tdir * nb, -1) / 0.01, -20, 20), np.clip(np.sum(tdir * rb, -1) / 0.01, -20, 20)]
    out['local'][..., 0:6] = np.stack(cols, -1) * ok[..., None]
    s = unit(sv)
    L = np.linalg.norm(sv, axis=-1)
    svl = np.stack([np.clip(np.sum(s * nb, -1) / 0.01, -20, 20), np.clip(np.sum(s * rb, -1) / 0.01, -20, 20),
                    np.clip(np.log(np.maximum(L, 1e-6) / 1.0), -6, 6), np.ones((n, 2))], -1)
    out['local'][..., 18:22] = svl * sv_ok[..., None]
    sh = out['local'].reshape(-1, G).copy()
    key = reco_digit.reshape(-1)
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        sh[idx] = sh[rng.permutation(idx)]
    out['local_shuffle'] = sh.reshape(n, 2, G)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in out.items()}


def main(out_path):
    d = np.load(DATA / 'diagnostics.npz')
    rng = np.random.default_rng(20260930)
    out, report = {}, {}
    for split, sid in SPLITS.items():
        rows = np.flatnonzero((d['split_ids'] == sid) & d['h_truth_valid'])
        labels = d['labels'][rows].astype(int)
        n = len(rows)
        tp4, nu, tch = d['truth_p4'][rows], d['truth_neutrino_lab4'][rows], d['truth_charges'][rows]
        tdig, rdig = d['truth_modes_digit'][rows], d['reco_modes_digit'][rows]
        basis, rp4 = d['reco_basis'][rows], d['reco_p4'][rows]
        # ROOT geometry, per label file, pack side s = ROOT tau{s+1}
        ip = np.zeros((n, 2, 3)); sv = np.zeros((n, 2, 3)); d0 = np.zeros((n, 2)); z0 = np.zeros((n, 2))
        svt = np.zeros((n, 2, 3)); lead = np.zeros((n, 2, 3))
        for lab in (1, 0):
            m = labels == lab
            a = read_root(lab, d['entry_indices'][rows][m])
            for s, t in ((0, 1), (1, 2)):
                ip[m, s] = np.stack([a[f'ip{t}_{c}'] for c in 'xyz'], -1)
                sv[m, s] = np.stack([a[f'sv_{c}{t}'] for c in 'xyz'], -1)
                d0[m, s], z0[m, s] = a[f'd0_{t}'], a[f'z0_{t}']
                svt[m, s] = np.stack([a[f'sv_{c}{t}_t'] - a[f'pv_{c}_t'] for c in 'xyz'], -1)
                lead[m, s] = np.stack([a[f'tau{t}_pi1_{v}'] for v in ('pt', 'eta', 'phi')], -1)
        # identity check: pack reco leading track == ROOT tau{s+1}_pi1
        rl = rp4[:, :, 0]
        pt = np.hypot(rl[..., 0], rl[..., 1])
        dpt = np.abs(pt / lead[..., 0] - 1)
        dphi = np.abs(np.angle(np.exp(1j * (np.arctan2(rl[..., 1], rl[..., 0]) - lead[..., 2]))))
        if dpt.max() > 1e-4 or dphi.max() > 1e-4:
            raise RuntimeError(f'{split}: pack/ROOT side identity failed (dpt {dpt.max()}, dphi {dphi.max()})')
        h = gen_h(tp4, nu, tch, tdig)
        stored = d['h_truth'][rows].astype(np.float64)
        closure = {}
        for dg, name in ((1, 'pi'), (2, 'rho'), (3, '3pi')):
            m = tdig == dg
            closure[name] = float(np.nanmax(np.abs(h[m] - stored[m])))
        if closure['pi'] > 1e-3 or closure['rho'] > 1e-3:
            raise RuntimeError(f'{split}: pi/rho closure failed {closure}')
        tdir = unit(rl[..., :3])
        sv_ok = (rdig == 3) & (np.linalg.norm(sv, axis=-1) > 0)
        geo = blocks(ip, d0, z0, sv, sv_ok, tdir, basis, rdig, rng)
        modes = np.select([tdig == 1, tdig == 2, tdig == 3], [0, 1, 3], -1)
        tau_true = unit(tp4.sum(2)[..., :3] + nu[..., :3])
        out[f'{split}_global_indices'] = d['global_indices'][rows]
        out[f'{split}_labels'] = labels
        out[f'{split}_modes'] = modes
        out[f'{split}_reco_digit'] = rdig
        out[f'{split}_h'] = h.astype(np.float32)
        out[f'{split}_h_stored'] = stored.astype(np.float32)
        out[f'{split}_truth_tau_direction'] = tau_true.astype(np.float32)
        for k, v in geo.items():
            out[f'{split}_{k}'] = v
        if split == 'validation':        # resolution arrays
            out['res_ip'] = ip; out['res_sv'] = sv; out['res_sv_ok'] = sv_ok; out['res_track_dir'] = tdir
            out['res_tau_true'] = tau_true; out['res_sv_true'] = svt; out['res_reco_digit'] = rdig
            out['res_truth_digit'] = tdig; out['res_basis'] = basis
        report[split] = dict(rows=int(n), H=int((labels == 1).sum()), closure_max_abs=closure,
                             h_norm_dev=float(np.nanmax(np.abs(np.linalg.norm(h, axis=-1) - 1))),
                             sv_available_fraction=float(sv_ok.mean()),
                             ip_available_fraction=float((geo['lab'][..., 3] > 0).mean()))
    np.savez_compressed(out_path, **out)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
