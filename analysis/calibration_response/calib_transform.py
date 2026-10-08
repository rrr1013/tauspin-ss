"""Physically consistent calibration shifts applied to one packed validation event.

The packed features are those of tauspin-ss/NN/build_dataset.py (relative-v3), train-standardised
with stats.json.  A shift is applied to the raw (unstandardised) quantities and every derived
feature that depends on them is rebuilt:

  pi0   all PFOs (neutral calorimeter objects): pT, E -> (1+eps) pT, (1+eps) E
  trk   all tracks:                              pT -> (1+eps) pT
  sag   all tracks, sagitta-type, eps in 1/TeV:  pT -> pT / (1 + q eps pT[TeV])
  met   MET vector only:                         MET -> (1+eps) MET (visible objects unchanged)

The visible tau 3-momentum (massless in the ntuple) moves by the change of its constituents
(core tracks and pi0-flagged PFOs), scaled by |p_tau| / |sum of constituents| so that a
constituent change propagates in proportion to the tau energy calibration.  With
met_mode='consistent' the MET vector absorbs minus the visible-tau change (the true neutrinos
are unchanged); 'fixed' keeps MET as stored.  Relative features are shifted additively from the
stored values, so eps = 0 reproduces the stored features exactly.  Token order is not changed
(the network has no positional encoding).
"""
import math

import numpy as np
import torch

EVENT = ('log1p_met_et', 'sin_met_phi', 'cos_met_phi', 'log1p_met_sumet', 'abs_tau_pair_dEta',
         'sin_tau_pair_dPhi', 'cos_tau_pair_dPhi', 'log_tau_minus_over_plus_pt',
         'sin_met_tau_minus_dPhi', 'cos_met_tau_minus_dPhi', 'sin_met_tau_plus_dPhi',
         'cos_met_tau_plus_dPhi', 'met_over_tau_pair_pt')
TRK_COLS = dict(log1p_pt=0, eta=1, sin_phi=2, cos_phi=3, charge=4, isCore=7, dEta=15, sin_dphi=16,
                cos_dphi=17, log1p_frac=18)
PFO_COLS = dict(log1p_pt=0, eta=1, sin_phi=2, cos_phi=3, log1p_e=4, isPi0=5, dEta=6, sin_dphi=7,
                cos_dphi=8, log1p_frac=9)
TAU_COLS = dict(log1p_pt=0, eta=1, sin_phi=2, cos_phi=3)


class Stats:
    def __init__(self, payload):
        self.m, self.s, self.f = {}, {}, {}
        for block in ('event', 'tau', 'track', 'pfo'):
            b = payload[block]
            self.m[block] = np.asarray(b['mean'], np.float64)
            self.s[block] = np.asarray(b['std'], np.float64)
            self.f[block] = np.asarray(b['standardize'], bool)

    def raw(self, block, x):
        x = np.asarray(x, np.float64)
        return np.where(self.f[block], x * self.s[block] + self.m[block], x)

    def std(self, block, x):
        return np.where(self.f[block], (x - self.m[block]) / self.s[block], x)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def p3(pt, eta, phi):
    pt, eta, phi = (np.asarray(v, np.float64) for v in (pt, eta, phi))
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1).reshape(-1, 3)


def kin(v):
    pt = np.hypot(v[..., 0], v[..., 1])
    return pt, np.arcsinh(v[..., 2] / pt), np.arctan2(v[..., 1], v[..., 0])


def _rec(cols, x, names):
    arrays = [x[:, cols[n]] for n in names]
    return arrays


def unpack(item, stats):
    ev = stats.raw('event', item['event_features'].numpy())
    tau = stats.raw('tau', item['tau_features'][:, :10].numpy())
    trk = stats.raw('track', item['track_features'].numpy()) if len(item['track_features']) else np.zeros((0, 19))
    pfo = stats.raw('pfo', item['pfo_features'].numpy()) if len(item['pfo_features']) else np.zeros((0, 10))
    c = TRK_COLS
    trk_rec = np.rec.fromarrays(
        [np.expm1(trk[:, c['log1p_pt']]), trk[:, c['eta']], np.arctan2(trk[:, c['sin_phi']], trk[:, c['cos_phi']]),
         trk[:, c['charge']], trk[:, c['isCore']], trk[:, c['dEta']],
         np.arctan2(trk[:, c['sin_dphi']], trk[:, c['cos_dphi']]), np.expm1(trk[:, c['log1p_frac']])],
        names='pt,eta,phi,charge,isCore,dEta,dPhi,ptFraction')
    c = PFO_COLS
    pfo_rec = np.rec.fromarrays(
        [np.expm1(pfo[:, c['log1p_pt']]), pfo[:, c['eta']], np.arctan2(pfo[:, c['sin_phi']], pfo[:, c['cos_phi']]),
         np.expm1(pfo[:, c['log1p_e']]), pfo[:, c['isPi0']], pfo[:, c['dEta']],
         np.arctan2(pfo[:, c['sin_dphi']], pfo[:, c['cos_dphi']]), np.expm1(pfo[:, c['log1p_frac']])],
        names='pt,eta,phi,e,isPi0,dEta,dPhi,ptFraction')
    tau_rec = dict(pt=np.expm1(tau[:, 0]), eta=tau[:, 1], phi=np.arctan2(tau[:, 2], tau[:, 3]))
    met = dict(et=float(np.expm1(ev[0])), phi=float(math.atan2(ev[1], ev[2])), sumet=float(np.expm1(ev[3])))
    return dict(trk=trk_rec, pfo=pfo_rec, tau=tau_rec, met=met, event_raw=ev,
                trk_side=item['track_sides'].numpy(), pfo_side=item['pfo_sides'].numpy())


def event_features_raw(raw):
    tau, met = raw['tau'], raw['met']
    pt, eta, phi = tau['pt'], tau['eta'], tau['phi']
    d = phi[0] - phi[1]
    dm, dp = met['phi'] - phi[0], met['phi'] - phi[1]
    return np.array([
        math.log1p(met['et']), math.sin(met['phi']), math.cos(met['phi']), math.log1p(met['sumet']),
        abs(eta[0] - eta[1]), math.sin(d), math.cos(d), math.log(pt[0] / pt[1]),
        math.sin(dm), math.cos(dm), math.sin(dp), math.cos(dp), met['et'] / (pt[0] + pt[1])])


def scale_factors(raw, kind, eps):
    trk, pfo = raw['trk'], raw['pfo']
    s_trk = np.ones(len(trk))
    s_pfo = np.ones(len(pfo))
    if kind == 'pi0':
        s_pfo[:] = 1.0 + eps
    elif kind == 'trk':
        s_trk[:] = 1.0 + eps
    elif kind == 'sag':
        s_trk = 1.0 / (1.0 + trk['charge'] * eps * trk['pt'] * 1e-3)
    elif kind not in ('none', 'met'):
        raise ValueError(kind)
    if not (np.all(np.isfinite(s_trk)) and np.all(s_trk > 0)):
        raise ValueError(f'{kind} {eps}: track scale outside its domain')
    return s_trk, s_pfo


def perturb(raw, kind, eps, met_mode='consistent'):
    """Return a new raw dict with the shift applied (eps = 0 or kind 'none' is the identity)."""
    trk, pfo, tau = raw['trk'].copy(), raw['pfo'].copy(), {k: v.copy() for k, v in raw['tau'].items()}
    s_trk, s_pfo = scale_factors(raw, kind, eps)
    tau3 = p3(tau['pt'], tau['eta'], tau['phi'])
    new3 = tau3.copy()
    for s in (0, 1):
        mt = (raw['trk_side'] == s) & (trk['isCore'] > 0.5)
        mp = (raw['pfo_side'] == s) & (pfo['isPi0'] > 0.5)
        c_t = p3(trk['pt'][mt], trk['eta'][mt], trk['phi'][mt])
        c_p = p3(pfo['pt'][mp], pfo['eta'][mp], pfo['phi'][mp])
        total = c_t.sum(0) + c_p.sum(0)
        delta = ((s_trk[mt] - 1)[:, None] * c_t).sum(0) + ((s_pfo[mp] - 1)[:, None] * c_p).sum(0)
        norm = np.linalg.norm(total)
        ratio = np.linalg.norm(tau3[s]) / norm if norm > 0 else 1.0
        new3[s] = tau3[s] + ratio * delta
    moved = np.any(new3 != tau3, axis=1)
    pt_n, eta_n, phi_n = kin(new3)
    pt_n = np.where(moved, pt_n, tau['pt'])
    eta_n = np.where(moved, eta_n, tau['eta'])
    phi_n = np.where(moved, phi_n, tau['phi'])
    deta = eta_n - tau['eta']
    dphi = np.where(moved, wrap(phi_n - tau['phi']), 0.0)
    ptr = tau['pt'] / pt_n
    trk.pt = trk.pt * s_trk
    pfo.pt = pfo.pt * s_pfo
    pfo.e = pfo.e * s_pfo
    ts, ps = raw['trk_side'], raw['pfo_side']
    trk.dEta = trk.dEta - deta[ts]
    trk.dPhi = np.where(dphi[ts] != 0, wrap(trk.dPhi - dphi[ts]), trk.dPhi)
    trk.ptFraction = trk.ptFraction * s_trk * ptr[ts]
    pfo.dEta = pfo.dEta - deta[ps]
    pfo.dPhi = np.where(dphi[ps] != 0, wrap(pfo.dPhi - dphi[ps]), pfo.dPhi)
    pfo.ptFraction = pfo.ptFraction * s_pfo * ptr[ps]
    met = dict(raw['met'])
    if met_mode == 'consistent' and moved.any():
        mv = np.array([met['et'] * math.cos(met['phi']), met['et'] * math.sin(met['phi'])])
        mv = mv - (new3[:, :2] - tau3[:, :2]).sum(0)
        met['et'] = float(np.hypot(*mv))
        met['phi'] = float(math.atan2(mv[1], mv[0]))
        met['sumet'] = float(met['sumet'] + (pt_n - tau['pt']).sum())
    elif met_mode not in ('fixed', 'consistent'):
        raise ValueError(met_mode)
    if kind == 'met' and eps != 0:
        met['et'] = met['et'] * (1.0 + eps)
    out = dict(raw)
    out.update(trk=trk, pfo=pfo, met=met,
               tau=dict(pt=pt_n, eta=eta_n, phi=phi_n),
               tau_shift=dict(deta=deta, dphi=dphi, ptr=ptr))
    return out


def pack(item, raw_new, stats, raw_old):
    """Return a copy of `item` whose packed features encode raw_new.  Only columns that a
    calibration shift can change are rewritten; when eps = 0 the stored values are returned."""
    new = dict(item)
    # event block: rebuild raw, write only the columns whose raw value changed
    ev_old = raw_old['event_raw']
    ev_new = event_features_raw(raw_new)
    ev_old_rebuilt = event_features_raw(raw_old)
    ev_raw = ev_old + (ev_new - ev_old_rebuilt)
    changed = np.abs(ev_new - ev_old_rebuilt) > 0
    ev_std = item['event_features'].numpy().astype(np.float64).copy()
    ev_std[changed] = stats.std('event', ev_raw)[changed]
    new['event_features'] = torch.as_tensor(ev_std, dtype=item['event_features'].dtype)

    tf = item['tau_features'].numpy().astype(np.float64).copy()
    t = raw_new['tau']
    moved = (t['pt'] != raw_old['tau']['pt']) | (t['eta'] != raw_old['tau']['eta']) | (t['phi'] != raw_old['tau']['phi'])
    raw_tau = stats.raw('tau', tf[:, :10])
    raw_tau[:, 0] = np.log1p(t['pt'])
    raw_tau[:, 1] = t['eta']
    raw_tau[:, 2] = np.sin(t['phi'])
    raw_tau[:, 3] = np.cos(t['phi'])
    tf[:, :10] = np.where(moved[:, None], stats.std('tau', raw_tau), tf[:, :10])
    new['tau_features'] = torch.as_tensor(tf, dtype=item['tau_features'].dtype)

    for key, block, rec_new, rec_old, cols in (
            ('track_features', 'track', raw_new['trk'], raw_old['trk'], TRK_COLS),
            ('pfo_features', 'pfo', raw_new['pfo'], raw_old['pfo'], PFO_COLS)):
        x = item[key].numpy().astype(np.float64).copy()
        if not len(x):
            continue
        r = stats.raw(block, x)
        r[:, cols['log1p_pt']] = np.log1p(rec_new.pt)
        if block == 'pfo':
            r[:, cols['log1p_e']] = np.log1p(rec_new.e)
        r[:, cols['dEta']] = rec_new.dEta
        r[:, cols['sin_dphi']] = np.sin(rec_new.dPhi)
        r[:, cols['cos_dphi']] = np.cos(rec_new.dPhi)
        r[:, cols['log1p_frac']] = np.log1p(rec_new.ptFraction)
        moved = ((rec_new.pt != rec_old.pt) | (rec_new.dEta != rec_old.dEta) | (rec_new.dPhi != rec_old.dPhi)
                 | (rec_new.ptFraction != rec_old.ptFraction))
        x = np.where(moved[:, None], stats.std(block, r), x)
        if not np.all(np.isfinite(x)):
            raise ValueError(f'non-finite {key} after shift')
        new[key] = torch.as_tensor(x, dtype=item[key].dtype)
    return new
