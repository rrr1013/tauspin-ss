"""CP-angle-type distributions for H (v2), Z v2 (two-step) and Z v3 (full ME), LHE truth.

Inputs: filtered-LHE extracts (z_entanglement/extract_lhe.py) of the three samples, 100 subjobs each.

Exact h (generator current for 3pi, z_entanglement/polarimeter.py).  In an (n, r, k) basis with
k = tau+ direction in the pair frame the transverse part of the ditau density is
    |T-||T+|/2 [ (C_nn + C_rr) cos(phi- - phi+) + (C_nn - C_rr) cos(phi- + phi+) ],
phi = atan2(h_r, h_n).  The difference angle dphi (the ideal CP angle) measures C_nn + C_rr
(H: +2, Z: 0); the sum angle sphi measures C_nn - C_rr, where the full-ME Z has its new
transverse correlation.  sphi needs a reference axis: the beam boosted to the pair frame
("boosted beam", physical) or the lab beam (canonical basis of the past runs).
For unit h, <|T|> = pi/4 so the modulation amplitude is (pi^2/16) (C_nn +- C_rr)/2.

Experimental estimators on the same events (truth four-vectors, no smearing):
  rho-rho: neutral-pion decay-plane method, charged-pion-pair ZMF, y+ y- sign flip;
  pi-pi:   impact-parameter method with the ideal IP direction (true tau direction).

usage: python cp_angle_distributions.py EXTRACT_DIR OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / 'z_entanglement'
sys.path.insert(0, str(HERE))
import polarimeter as pol  # noqa: E402
import zent  # noqa: E402
import ztheory as zt  # noqa: E402

SAMPLES = {'h_v2': 'H (v2)', 'z_v2': 'Z v2 (two-step)', 'z_v3': 'Z v3 (full ME)'}
NB = 24


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def phistar(p_minus, p_plus, a_minus, a_plus):
    """acoplanarity angle in the ZMF of (p_minus, p_plus); a = 4-vectors defining each plane."""
    beta = (p_minus + p_plus)[:, :3] / (p_minus + p_plus)[:, 3:4]
    pm, pp = pol.boost(p_minus, beta), pol.boost(p_plus, beta)
    am, ap = pol.boost(a_minus, beta), pol.boost(a_plus, beta)
    um, up = unit(pm[:, :3]), unit(pp[:, :3])
    lm = unit(am[:, :3] - np.sum(am[:, :3] * um, -1, keepdims=True) * um)
    lp = unit(ap[:, :3] - np.sum(ap[:, :3] * up, -1, keepdims=True) * up)
    phi = np.arccos(np.clip(np.sum(lm * lp, -1), -1, 1))
    o = np.sum(um * np.cross(lp, lm), -1)
    return np.where(o < 0, 2 * np.pi - phi, phi)


def analyse(ev):
    fr, h_can, h_cart = zent.exact_h(ev)
    good = np.all(np.isfinite(h_cart), axis=(1, 2))
    pv = zent.pair_frame_vectors(ev, fr)
    k = pv['k']
    boosted_ref = np.where((ev['beam'][:, 0, 2] > 0)[:, None], pv['b2'], pv['b1'])   # -z beam in pair frame
    bases = {'boosted_beam': zt.helicity_basis(k, boosted_ref), 'canonical_lab_beam': fr['basis']}
    out = {'good': good, 'mode': ev['mode']}
    for name, O in bases.items():
        hm = np.einsum('nab,nb->na', O, h_cart[:, 0])
        hp = np.einsum('nab,nb->na', O, h_cart[:, 1])
        fm, fp = np.arctan2(hm[:, 1], hm[:, 0]), np.arctan2(hp[:, 1], hp[:, 0])
        out[name] = dict(dphi=np.mod(fm - fp, 2 * np.pi), sphi=np.mod(fm + fp, 2 * np.pi),
                         C=zent.measured_state(hm[good], hp[good])[2])
    # experimental phi*_CP (side 0 = tau-, side 1 = tau+)
    pim, pip = ev['pions'][:, 0, 0], ev['pions'][:, 1, 0]
    rr = good & (ev['mode'][:, 0] == 1) & (ev['mode'][:, 1] == 1)
    ps = phistar(pim, pip, ev['pi0'][:, 0], ev['pi0'][:, 1])
    y = (pim[:, 3] - ev['pi0'][:, 0, 3]) / (pim[:, 3] + ev['pi0'][:, 0, 3]), \
        (pip[:, 3] - ev['pi0'][:, 1, 3]) / (pip[:, 3] + ev['pi0'][:, 1, 3])
    ps_rr = np.where(y[0] * y[1] < 0, np.mod(ps + np.pi, 2 * np.pi), ps)
    tau = ev['pions'].sum(2) + ev['pi0'] + ev['nu']
    ip = []
    for s in (0, 1):
        t, u = unit(tau[:, s, :3]), unit(ev['pions'][:, s, 0, :3])
        n = unit(t - np.sum(t * u, -1, keepdims=True) * u)
        ip.append(np.concatenate([n, np.zeros((len(n), 1))], -1))
    pp = good & (ev['mode'][:, 0] == 0) & (ev['mode'][:, 1] == 0)
    ps_pp = phistar(pim, pip, ip[0], ip[1])
    out['phistar_rhorho'] = ps_rr[rr]
    out['phistar_pipi_ip'] = ps_pp[pp]
    return out


def modulation(x, harmonic=1):
    """fit 1 + a cos(x): a = 2 <cos x>, with its statistical error."""
    c = np.cos(harmonic * x)
    return float(2 * c.mean()), float(2 * c.std() / np.sqrt(len(c)))


def main(extract_dir, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res, data = {}, {}
    for tag in SAMPLES:
        paths = sorted(Path(extract_dir).glob(f'chunk_{tag}_0*.npz'))
        ev = zent.load(paths)
        a = analyse(ev)
        data[tag] = a
        g = a['good']
        r = {'n_events': int(len(g)), 'n_good_h': int(g.sum())}
        for name in ('boosted_beam', 'canonical_lab_beam'):
            C = a[name]['C']
            r[name] = dict(C_diag=np.diag(C).round(4).tolist(),
                           C_offdiag_nr=[round(float(C[0, 1]), 4), round(float(C[1, 0]), 4)],
                           dphi_cos_amp=modulation(a[name]['dphi'][g]),
                           sphi_cos_amp=modulation(a[name]['sphi'][g]),
                           dphi_sin_amp=modulation(a[name]['dphi'][g] - np.pi / 2),
                           pred_dphi_amp=float(np.pi ** 2 / 16 * (C[0, 0] + C[1, 1]) / 2),
                           pred_sphi_amp=float(np.pi ** 2 / 16 * (C[0, 0] - C[1, 1]) / 2))
        r['phistar_rhorho'] = dict(n=int(a['phistar_rhorho'].size), cos_amp=modulation(a['phistar_rhorho']))
        r['phistar_pipi_ip'] = dict(n=int(a['phistar_pipi_ip'].size), cos_amp=modulation(a['phistar_pipi_ip']))
        res[tag] = r
    (outdir / 'cp_angle_results.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(outdir / 'cp_angle_arrays.npz', **{
        f'{t}_{n}_{v}': data[t][n][v][data[t]['good']] for t in SAMPLES
        for n in ('boosted_beam', 'canonical_lab_beam') for v in ('dphi', 'sphi')},
        **{f'{t}_{v}': data[t][v] for t in SAMPLES for v in ('phistar_rhorho', 'phistar_pipi_ip')})
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
