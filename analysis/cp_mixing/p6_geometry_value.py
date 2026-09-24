"""What each piece of detector geometry is worth: CP mixing angle vs H/Z.

The 2026-09-23 beyond-ceiling arms (base, SV only, IP only, IP+SV v1, IP+SV v2,
shuffled geometry, ideal-IP oracle) are evaluated twice from the *same*
predicted h with plug-in readouts:

  H/Z   fixed-C likelihood ratio  log(1 + h C_H h) - log(1 + h C_Z h)  -> AUC
  CP    triple product            (h- x h+) . k_hat                    -> sigma(phi_tau)

Restricted to events with no 3-prong side, where the CLEO and generator-current
polarimeters are identical, so the CLEO-teacher arms are usable without the
2026-09-24 teacher correction.  Both are converted to an effective-statistics
ratio against the geometry-free baseline: (sigma_base/sigma)^2 for CP and
(d'/d'_base)^2 with d' = sqrt(2) Phi^-1(AUC) for H/Z.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import score, sensitivity, triple  # noqa: E402

CH, CZ = np.diag([1., 1., -1.]), np.diag([0., 0., 1.])
ARMS = [('base_s43', 'no geometry'), ('sv_s42', '+ SV only'), ('ip22_s42', '+ 3 track IPs only'),
        ('full_s42', '+ IP and SV (v1, 16 ch)'), ('full22_s42', '+ IP and SV (v2, 22 ch)'),
        ('full22_shuffle_s42', 'geometry shuffled (control)'), ('idealip22_s42', 'ideal-IP oracle')]


def auc(s, y):
    y = np.asarray(y).astype(bool)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = y.sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def boot_auc(s, y, B=200, seed=1):
    rng = np.random.default_rng(seed)
    v = [auc(s[k], y[k]) for k in (rng.integers(0, len(s), len(s)) for _ in range(B))]
    return float(np.std(v))


def llr(h):
    o = np.einsum('ni,nj->nij', h[:, 0], h[:, 1])
    return (np.log(np.maximum(1 + np.einsum('ij,nij->n', CH, o), 1e-12))
            - np.log(np.maximum(1 + np.einsum('ij,nij->n', CZ, o), 1e-12)))


def ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'results')
    ap.add_argument('--n-ref', type=int, default=10000)
    args = ap.parse_args()

    D = np.load(args.data / 'truth_surface_cleo.npz')
    G = np.load(args.data / 'gen3pi_full22_s42.npz')
    assert np.array_equal(G['global_indices'], D['global_indices'])
    h_exact = G['h'].astype(float)                     # generator-current target
    modes, y = D['modes'], D['labels'].astype(bool)
    no3p = (modes != 3).all(1)
    # on no-3prong events the CLEO surface and generator target must agree
    dev = np.abs(D['h_ref'][no3p] - h_exact[no3p]).max()

    C0 = c_matrix(0.0)
    S_all, _ = score(h_exact, C0)
    kH = no3p & y
    out = {'n_ref': args.n_ref, 'selection': 'no 3-prong side',
           'n_events': int(no3p.sum()), 'n_H': int(kH.sum()),
           'cleo_vs_generator_max_dev_on_selection': float(dev),
           'exact_h': {'cp_sigma_deg': sensitivity(triple(h_exact[kH, 0], h_exact[kH, 1]),
                                                   S_all[kH], args.n_ref, boot=200)['sigma_deg'],
                       'cp_optimal_sigma_deg': float(np.rad2deg(1 / np.sqrt(args.n_ref * np.var(S_all[kH])))),
                       'hz_auc': auc(llr(h_exact[no3p]), y[no3p])},
           'arms': {}}
    for arm, desc in ARMS:
        f = args.data / f'cleo_{arm}.npz'
        if not f.exists():
            continue
        P = np.load(f)
        assert np.array_equal(P['global_indices'], D['global_indices'])
        hp = P['h_pred'].astype(float)
        cp = sensitivity(triple(hp[kH, 0], hp[kH, 1]), S_all[kH], args.n_ref, boot=200)
        a = auc(llr(hp[no3p]), y[no3p])
        out['arms'][arm] = {'description': desc, 'cp_sigma_deg': cp['sigma_deg'],
                            'cp_sigma_deg_se': cp['sigma_deg_se'], 'hz_auc': a,
                            'hz_auc_se': boot_auc(llr(hp[no3p]), y[no3p]),
                            'mean_pred_norm': float(np.linalg.norm(hp, axis=-1).mean())}
    b = out['arms']['base_s43']
    dprime_b = math.sqrt(2) * ppf(b['hz_auc'])
    for arm, r in out['arms'].items():
        r['cp_effective_lumi_vs_base'] = (b['cp_sigma_deg'] / r['cp_sigma_deg'])**2
        r['hz_effective_lumi_vs_base'] = (math.sqrt(2) * ppf(r['hz_auc']) / dprime_b)**2
    (args.output / 'geometry_value.json').write_text(json.dumps(out, indent=1))

    print(f"no-3prong events {out['n_events']} (H {out['n_H']}), CLEO/generator h max dev {dev:.2e}")
    print(f"{'arm':30s} {'sigma(phi) deg':>16s} {'H/Z AUC':>10s} {'CP effL':>9s} {'H/Z effL':>9s}")
    for arm, r in out['arms'].items():
        print(f"{r['description']:30s} {r['cp_sigma_deg']:9.3f}+-{r['cp_sigma_deg_se']:.3f}"
              f" {r['hz_auc']:10.4f} {r['cp_effective_lumi_vs_base']:9.3f} {r['hz_effective_lumi_vs_base']:9.3f}")
    e = out['exact_h']
    print(f"{'exact h (ceiling)':30s} {e['cp_sigma_deg']:9.3f}        {e['hz_auc']:10.4f}"
          f"   optimal sigma {e['cp_optimal_sigma_deg']:.3f}")


if __name__ == '__main__':
    main()
