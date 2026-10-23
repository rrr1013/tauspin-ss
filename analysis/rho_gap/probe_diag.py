"""Quick diagnostic script to evaluate probe runs (stride 10, 5939 events)."""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN, lr_of, pair_masks, wauc
from hquality import wcorr

MEAS = ('met', 'ip3_sv', 'ip3_sv_mass')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', nargs='+', required=True)
    args = parser.parse_args()

    arms = dict(s.split('=', 1) for s in args.arm)
    P = {k: np.load(Path(v) / 'posterior.npz') for k, v in arms.items()}
    ref = P[next(iter(P))]
    ids, y, w = ref['global_indices'], ref['labels'].astype(int), ref['weights'].astype(float)
    modes, rm, av = ref['modes'], ref['reco_h_mode'].astype(int), ref['reco_h_available_side']
    W = np.broadcast_to(w[:, None], av.shape)

    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    look = {int(g): i for i, g in enumerate(t['global_indices'][v])}
    pos = np.array([look[int(g)] for g in ids])
    hx = t['h'][v][pos].astype(float)

    dd = np.load(GEN / 'p11_readout' / 'readout_scores_ens_full22.npz')
    s_ens = dd['scores'][pos]

    pairs = pair_masks(modes)
    print("=" * 95)
    print(f"{'Arm':28s} | {'Overall':7s} | {'rho x rho':9s} {'pi x rho':9s} {'rho x 3pi':9s} | {'rho hn':6s} {'rho hr':6s} {'rho hk':6s}")
    print("=" * 95)

    # Reference: exact h & learner
    exact_lr = lr_of(hx)
    print(f"{'exact h':28s} | {wauc(exact_lr, y, w):.4f}  | {wauc(exact_lr[pairs['rho x rho']], y[pairs['rho x rho']], w[pairs['rho x rho']]):.4f}    {wauc(exact_lr[pairs['pi x rho']], y[pairs['pi x rho']], w[pairs['pi x rho']]):.4f}    {wauc(exact_lr[pairs['rho x 3pi']], y[pairs['rho x 3pi']], w[pairs['rho x 3pi']]):.4f}    | 1.000  1.000  1.000")
    print(f"{'learner (ens full22)':28s} | {wauc(s_ens, y, w):.4f}  | {wauc(s_ens[pairs['rho x rho']], y[pairs['rho x rho']], w[pairs['rho x rho']]):.4f}    {wauc(s_ens[pairs['pi x rho']], y[pairs['pi x rho']], w[pairs['pi x rho']]):.4f}    {wauc(s_ens[pairs['rho x 3pi']], y[pairs['rho x 3pi']], w[pairs['rho x 3pi']]):.4f}    | 0.644  0.653  0.750")
    print("-" * 95)

    msk_rho = av & (rm == 1) & (modes == 1)

    for k, d in P.items():
        lr = d['ip3_sv_mass_lr']
        auc_tot = wauc(lr, y, w)
        auc_rr = wauc(lr[pairs['rho x rho']], y[pairs['rho x rho']], w[pairs['rho x rho']])
        auc_pr = wauc(lr[pairs['pi x rho']], y[pairs['pi x rho']], w[pairs['pi x rho']])
        auc_r3 = wauc(lr[pairs['rho x 3pi']], y[pairs['rho x 3pi']], w[pairs['rho x 3pi']])
        h_side = d['ip3_sv_mass_Eh']
        corrs = [wcorr(h_side[..., c][msk_rho], hx[..., c][msk_rho], W[msk_rho]) for c in range(3)]
        print(f"{k:28s} | {auc_tot:.4f}  | {auc_rr:.4f}    {auc_pr:.4f}    {auc_r3:.4f}    | {corrs[0]:.3f}  {corrs[1]:.3f}  {corrs[2]:.3f}")

    print("=" * 95)


if __name__ == '__main__':
    main()
