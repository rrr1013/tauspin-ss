"""Denoised reference: E[h_exact | reco visible rho, truth nu], learned on TRAIN.

Per reco-rho side (reco_h_mode 1, both sides available, exact h valid) the features are built in the
pair-CM -> tau rest frame of (reco visible + truth nu), as frames() of mode_pair_auc_origin/polarimeter, and
expressed in the (n, r, k) pair basis of that frame:
  reco pi+- (E, 3), reco pi0 (E, 3), truth nu (3)  in the tau rest frame
  formula h of s1 ('reco', 3)
  reco m_vis, reco tau mass |vis + nu|, lab E_vis, lab y = (E_pi - E_pi0)/(E_pi + E_pi0), reco_mode_raw, n neutral PFOs
Target: the exact (gen3pi) h of that side, in the truth pair basis (as the learner's target).
The basis mismatch is part of what is learned.  CP: tau+ sides are mirrored as in canonical_h (sign of the
3-vectors) so one network serves both sides; the side index is also an input.

Model: MLP 21 -> 256 x 4 (GELU) -> 3, MSE, AdamW 1e-3 cosine, batch 1024, early stopping on a 10% TRAIN holdout
(patience 6, max 80 epochs).  TRAIN predictions are 2-fold cross-fitted (event_number parity); VALIDATION
gets the mean of the two fold models.  Test never loaded.

Output: <out>/denoised.npz  h_train (N_tr,2,3), h_validation (N_va,2,3) with rho sides replaced by the
prediction and every other side equal to the s1 formula h; plus the side masks and fold histories.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
from polarimeter import frames  # noqa: E402


def features(D, h_formula):
    n = len(D['reco_h_mode'])
    act = np.arange(3)[None, None, :] < D['reco_h_counts'][..., None]
    pions = np.where(act[..., None], D['reco_h_pions_lab4'], 0.0)
    nu4 = D['truth_neutrino_lab4'].copy()
    nu4[..., 3] = np.linalg.norm(nu4[..., :3], axis=-1)
    both = D['reco_h_available_side'].all(1)
    rho = both[:, None] & (D['reco_h_mode'] == 1)
    F = np.full((n, 2, 21), np.nan)
    ix = np.flatnonzero(both)
    for s0 in range(0, len(ix), 50000):
        e = ix[s0:s0 + 50000]
        fr = frames(pions[e], D['reco_h_neutral_lab4'][e], nu4[e])
        B = fr['basis']                                                  # (n,3,3) rows n,r,k
        for side in (0, 1):
            sgn = 1.0 if side == 0 else -1.0
            proj = lambda v: sgn * np.einsum('nij,nj->ni', B, v[..., :3])
            pi = fr['rest']['pions'][:, side, 0]
            z0 = fr['rest']['pi0'][:, side]
            nu = fr['rest']['nu'][:, side]
            vis = D['reco_visible_tau_lab4'][e, side]
            mvis = np.sqrt(np.maximum(vis[:, 3] ** 2 - np.sum(vis[:, :3] ** 2, -1), 0))
            tau = pions[e, side].sum(1) + D['reco_h_neutral_lab4'][e, side] + nu4[e, side]
            mtau = np.sqrt(np.maximum(tau[:, 3] ** 2 - np.sum(tau[:, :3] ** 2, -1), 0))
            ep, ez = D['reco_h_pions_lab4'][e, side, 0, 3], D['reco_h_neutral_lab4'][e, side, 3]
            F[e, side] = np.concatenate([
                pi[:, 3:4], proj(pi), z0[:, 3:4], proj(z0), proj(nu), h_formula[e, side],
                np.stack([mvis, mtau, np.log(vis[:, 3]), (ep - ez) / np.maximum(ep + ez, 1e-9),
                          D['reco_mode_raw'][e, side].astype(float),
                          D['reco_h_neutral_counts'][e, side].astype(float), np.full(len(e), side, float)], 1)], 1)
    return F, rho


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        layers, k = [], d
        for _ in range(4):
            layers += [nn.Linear(k, 256), nn.GELU()]
            k = 256
        self.f = nn.Sequential(*layers, nn.Linear(k, 3))

    def forward(self, x):
        return self.f(x)


def fit(X, Y, seed, device, max_epochs=80, patience=6):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    nh = len(X) // 10
    ho, tr = perm[:nh], perm[nh:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    net = Net(X.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    best, best_state, bad, hist = np.inf, None, 0, []
    tri = torch.tensor(tr, device=device)
    for ep in range(max_epochs):
        net.train()
        order = tri[torch.randperm(len(tri), device=device)]
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            hl = float(((net(Xt[ho]) - Yt[ho]) ** 2).sum(-1).mean())
        hist.append(hl)
        if hl < best - 1e-5:
            best, bad = hl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(Z):
        with torch.no_grad():
            return net(torch.tensor((Z - mu) / sd, dtype=torch.float32, device=device)).cpu().numpy()
    return predict, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = {}
    hist = {}
    for split in ('train', 'validation'):
        with np.load(LADDER / f'{split}.npz') as d:
            D = {k: d[k] for k in ('global_indices', 'event_numbers', 'reco_h_pions_lab4', 'reco_h_counts',
                                   'reco_h_neutral_lab4', 'reco_h_neutral_counts', 'reco_h_mode', 'reco_mode_raw',
                                   'truth_neutrino_lab4', 'reco_h_available_side', 'reco_visible_tau_lab4')}
        t = np.load(GEN / 'targets' / f'{split}.npz')
        v = t['h_valid'].astype(bool)
        pos = {int(g): i for i, g in enumerate(D['global_indices'])}
        r = np.array([pos[int(g)] for g in t['global_indices'][v]])
        D = {k: x[r] for k, x in D.items()}
        L = np.load(a.ladder_dir / f'ladder_{split}.npz')
        lp = {int(g): i for i, g in enumerate(L['global_indices'])}
        hf = L['h_reco'][[lp[int(g)] for g in D['global_indices']]]
        F, rho = features(D, hf)
        out[split] = dict(F=F, rho=rho & np.isfinite(F).all(-1), hx=t['h'][v].astype(float), hf=hf,
                          ids=D['global_indices'], ev=D['event_numbers'])
        print(split, 'rho sides', int(out[split]['rho'].sum()), flush=True)

    T, Va = out['train'], out['validation']
    e_tr, s_tr = np.nonzero(T['rho'] & np.isfinite(T['hx']).all(-1))
    X, Y = T['F'][e_tr, s_tr], T['hx'][e_tr, s_tr]
    fold = (T['ev'][e_tr] % 2).astype(int)
    pred_tr = np.zeros_like(Y)
    e_va, s_va = np.nonzero(Va['rho'])
    pred_va = np.zeros((len(e_va), 3))
    for f in (0, 1):
        predict, hist[f'fold{f}'] = fit(X[fold != f], Y[fold != f], 20261018 + f, device)
        pred_tr[fold == f] = predict(X[fold == f])
        pred_va += 0.5 * predict(Va['F'][e_va, s_va])
        print(f'fold {f}: epochs {len(hist[f"fold{f}"])}, best holdout MSE {min(hist[f"fold{f}"]):.4f}', flush=True)
    h_tr, h_va = T['hf'].copy(), Va['hf'].copy()
    h_tr[e_tr, s_tr] = pred_tr
    h_va[e_va, s_va] = pred_va
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.output_dir / 'denoised.npz', train_global_indices=T['ids'], validation_global_indices=Va['ids'],
             h_train=h_tr, h_validation=h_va, rho_train=T['rho'], rho_validation=Va['rho'])
    (a.output_dir / 'denoised_history.json').write_text(json.dumps(hist))
    m = Va['rho'] & np.isfinite(Va['hx']).all(-1)
    for name, h in (('formula', Va['hf']), ('denoised', h_va)):
        c = [np.corrcoef(h[..., i][m], Va['hx'][..., i][m])[0, 1] for i in range(3)]
        print(name, 'unweighted corr n r k', np.round(c, 3), flush=True)


if __name__ == '__main__':
    main()
