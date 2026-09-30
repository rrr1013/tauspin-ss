"""Rescale the tau-pair invariant mass of every LHE event to a fixed value.

Mass-width control for the v3 (full-ME) Z sample: the H sample is a delta at
91.1872 GeV while v3 Z has a Breit-Wigner line shape, so any classifier that
can build m_tautau separates H/Z by mass.  This script moves every event onto
the H mass while keeping the spin information exactly:

  1. go to the tau-pair rest frame (pair 3-momentum P is kept in the lab);
  2. keep both tau directions there, set |p*| so that the pair mass is M0;
  3. every tau descendant keeps its four-momentum in its own tau rest frame
     (reached by the pure boost along the tau direction in the pair frame),
     i.e. the decay angular distributions and polarimeter vectors are unchanged;
  4. boost back to the lab with the new pair four-momentum (P, sqrt(P^2+M0^2)).

The tau-pair parent line (pdg 23/25, if present) is updated to the new pair
four-momentum.  Everything else (incoming partons, recoil jet) is left as is;
the tiny energy non-conservation this causes is irrelevant for the tau system.

usage: python lhe_mass_rescale.py --src SRC_DIR --dst DST_DIR --dsid 999802 --mass 91.1872 [--workers 24]
Writes DST_DIR/test_<dsid>_subN_n2000/mc.TXT._00001.tar.gz with the same
member name, plus DST_DIR/rescale_summary_<dsid>.json.
"""
import argparse
import glob
import io
import json
import math
import os
import tarfile
from multiprocessing import Pool

import numpy as np


def boost(p, b):
    """Four-vector(s) p=(px,py,pz,E) seen from a frame moving with velocity b."""
    b2 = float(b @ b)
    if b2 <= 0.0:
        return p.copy()
    g = 1.0 / math.sqrt(1.0 - b2)
    bp = p[..., :3] @ b
    out = np.empty_like(p)
    out[..., 3] = g * (p[..., 3] - bp)
    coef = (g - 1.0) * bp / b2 - g * p[..., 3]
    out[..., :3] = p[..., :3] + coef[..., None] * b
    return out


def rescale_event(p4, taus, desc, parent, m0):
    """Return rescaled copy of p4 [n,4]; taus = two row indices; desc[i] = descendant rows of tau i."""
    q = p4.copy()
    t = p4[list(taus)]
    P = t.sum(0)
    b_old = P[:3] / P[3]
    E_new = math.sqrt(float(P[:3] @ P[:3]) + m0 * m0)
    b_new = P[:3] / E_new
    mt = np.sqrt(np.maximum(t[:, 3] ** 2 - (t[:, :3] ** 2).sum(1), 0.0))
    t_cm = boost(t, b_old)
    for i, row in enumerate(taus):
        u = t_cm[i, :3] / np.linalg.norm(t_cm[i, :3])
        e_new = 0.5 * m0 if abs(mt[0] - mt[1]) < 1e-3 else (m0 * m0 + mt[i] ** 2 - mt[1 - i] ** 2) / (2 * m0)
        pst = math.sqrt(max(e_new * e_new - mt[i] ** 2, 0.0))
        tn_cm = np.concatenate([pst * u, [e_new]])
        bt_old = t_cm[i, :3] / t_cm[i, 3]
        bt_new = tn_cm[:3] / tn_cm[3]
        q[row] = boost(tn_cm[None], -b_new)[0]
        rows = desc[i]
        if rows:
            d = p4[rows]
            d = boost(boost(boost(d, b_old), bt_old), -bt_new)
            q[rows] = boost(d, -b_new)
    if parent is not None:
        q[parent] = np.concatenate([P[:3], [E_new]])
    return q


def fmt(x):
    return f'{x:+.10e}'


def process(args):
    src_tar, dst_tar, m0 = args
    with tarfile.open(src_tar, 'r:gz') as tf:
        member = tf.getmembers()[0]
        text = tf.extractfile(member).read().decode('utf-8', 'replace')
    lines = text.split('\n')
    out = []
    stats = {'events': 0, 'rescaled': 0, 'skipped': 0, 'm_old': [], 'm_new': []}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip().startswith('<event'):
            out.append(line)
            i += 1
            continue
        out.append(line)
        head = lines[i + 1]
        out.append(head)
        nup = int(head.split()[0])
        rows = [lines[i + 2 + k] for k in range(nup)]
        i += 2 + nup
        stats['events'] += 1
        fields = [r.split() for r in rows]
        pdg = [int(f[0]) for f in fields]
        st = [int(f[1]) for f in fields]
        mo = [int(f[2]) for f in fields]
        p4 = np.array([[float(f[6]), float(f[7]), float(f[8]), float(f[9])] for f in fields])
        taus = [k for k in range(nup) if abs(pdg[k]) == 15 and st[k] == 2]
        if len(taus) != 2:
            stats['skipped'] += 1
            out.extend(rows)
            continue
        children = {k: [] for k in range(nup)}
        for k in range(nup):
            if mo[k] >= 1:
                children[mo[k] - 1].append(k)

        def descendants(k):
            res = []
            for c in children[k]:
                res.append(c)
                res.extend(descendants(c))
            return res

        desc = [descendants(k) for k in taus]
        par = mo[taus[0]] - 1 if mo[taus[0]] == mo[taus[1]] and mo[taus[0]] >= 1 else None
        if par is not None and abs(pdg[par]) not in (23, 25):
            par = None
        P = p4[taus].sum(0)
        stats['m_old'].append(math.sqrt(max(P[3] ** 2 - float(P[:3] @ P[:3]), 0.0)))
        q = rescale_event(p4, taus, desc, par, m0)
        Q = q[taus].sum(0)
        stats['m_new'].append(math.sqrt(max(Q[3] ** 2 - float(Q[:3] @ Q[:3]), 0.0)))
        stats['rescaled'] += 1
        changed = set(taus) | set(desc[0]) | set(desc[1]) | ({par} if par is not None else set())
        for k in range(nup):
            if k not in changed:
                out.append(rows[k])
                continue
            f = fields[k]
            vals = list(q[k])
            mass = float(f[10])
            if k == par:
                mass = m0
            new = [f[0].rjust(9), f[1].rjust(2), f[2].rjust(4), f[3].rjust(4), f[4].rjust(4), f[5].rjust(4)]
            new += [fmt(v) for v in vals[:3]] + [f'{vals[3]:.10e}', f'{mass:.10e}'] + f[11:]
            out.append(' '.join(new))
    data = '\n'.join(out).encode('utf-8')
    os.makedirs(os.path.dirname(dst_tar), exist_ok=True)
    with tarfile.open(dst_tar, 'w:gz') as tf:
        info = tarfile.TarInfo(member.name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    m_old = np.array(stats['m_old'])
    m_new = np.array(stats['m_new'])
    return {'file': src_tar, 'events': stats['events'], 'rescaled': stats['rescaled'],
            'skipped': stats['skipped'],
            'm_old_q': np.quantile(m_old, [0.01, 0.5, 0.99]).tolist() if m_old.size else [],
            'm_new_maxdev': float(np.abs(m_new - m0).max()) if m_new.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--dsid', required=True)
    ap.add_argument('--mass', type=float, required=True)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    srcs = sorted(glob.glob(f'{a.src}/test_{a.dsid}_sub*_n2000/mc.TXT._00001.tar.gz'))
    jobs = [(s, os.path.join(a.dst, os.path.relpath(s, a.src)), a.mass) for s in srcs]
    with Pool(a.workers) as pool:
        res = pool.map(process, jobs)
    summ = {'src': a.src, 'dsid': a.dsid, 'mass': a.mass, 'files': len(res),
            'events': sum(r['events'] for r in res), 'rescaled': sum(r['rescaled'] for r in res),
            'skipped': sum(r['skipped'] for r in res),
            'max_mass_dev': max(r['m_new_maxdev'] or 0 for r in res),
            'per_file': res}
    with open(os.path.join(a.dst, f'rescale_summary_{a.dsid}.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'per_file'}))


if __name__ == '__main__':
    main()
