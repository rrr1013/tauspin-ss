"""End-to-end closure: inject phi_tau, fit it back, check bias and spread.

The quoted sigma comes from a linear-response formula.  This checks it the
direct way.  For a grid of injected phi_tau the H rows are resampled with the
exact-h spin weight w(phi_true), 10,000 events per pseudo-experiment; each
pseudo-experiment is then fitted by inverting the calibration curve
<T>(phi) measured once on the full sample by the same reweighting.

A correct sigma means mean(phi_hat) = phi_true and sd(phi_hat) = sigma.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'results')
    ap.add_argument('--n-toy-events', type=int, default=10000)
    ap.add_argument('--toys', type=int, default=400)
    ap.add_argument('--inject', type=float, nargs='+', default=[0., 5., 10., 20., 30.])
    args = ap.parse_args()

    D = np.load(args.data / 'truth_surface_cleo.npz')
    G = np.load(args.data / 'gen3pi_full22_s42.npz')
    assert np.array_equal(G['global_indices'], D['global_indices'])
    y = D['labels'].astype(bool)
    h = G['h'].astype(float)[y]
    hp = G['h_pred'].astype(float)[y]
    C0 = c_matrix(0.0)
    S, _ = score(h, C0)

    obs = {'exact_h_Tp': triple(h[:, 0], h[:, 1]),
           'reco_h_Tp': triple(hp[:, 0], hp[:, 1]),
           'exact_h_score': S}
    grid = np.deg2rad(np.arange(-40, 40.5, 1.0))
    W = np.stack([weights(h, g, C0) for g in grid])           # (G, N)

    out = {'n_parent_H': int(y.sum()), 'n_toy_events': args.n_toy_events,
           'toys': args.toys, 'observables': {}}
    rng = np.random.default_rng(20260925)
    for name, T in obs.items():
        cal = (W @ T) / W.sum(1)                              # <T>(phi) on the full sample
        pred = sensitivity(T, S, args.n_toy_events)['sigma_deg']
        rows = {'predicted_sigma_deg': pred, 'injections': {}}
        for deg in args.inject:
            w = weights(h, np.deg2rad(deg), C0)
            p = np.maximum(w, 0)
            p = p / p.sum()
            hat = []
            for _ in range(args.toys):
                k = rng.choice(len(T), size=args.n_toy_events, replace=True, p=p)
                hat.append(np.interp(T[k].mean(), cal[::-1] if cal[-1] < cal[0] else cal,
                                     grid[::-1] if cal[-1] < cal[0] else grid))
            hat = np.rad2deg(np.asarray(hat))
            rows['injections'][f'{deg:g}'] = {
                'mean_phi_hat_deg': float(hat.mean()),
                'bias_deg': float(hat.mean() - deg),
                'sd_phi_hat_deg': float(hat.std(ddof=1)),
                'sd_over_predicted': float(hat.std(ddof=1) / pred),
                'ess_frac': float(w.sum()**2 / (len(w) * (w**2).sum()))}
        out['observables'][name] = rows

    (args.output / 'closure_fit.json').write_text(json.dumps(out, indent=1))
    for name, r in out['observables'].items():
        print(f'\n{name}: predicted sigma {r["predicted_sigma_deg"]:.3f} deg '
              f'for {args.n_toy_events} events')
        print(f'  {"injected":>9s} {"fitted mean":>12s} {"bias":>8s} {"sd":>8s} {"sd/pred":>8s} {"ESS":>6s}')
        for k, v in r['injections'].items():
            print(f'  {k:>9s} {v["mean_phi_hat_deg"]:12.3f} {v["bias_deg"]:8.3f} '
                  f'{v["sd_phi_hat_deg"]:8.3f} {v["sd_over_predicted"]:8.3f} {v["ess_frac"]:6.3f}')


if __name__ == '__main__':
    main()
