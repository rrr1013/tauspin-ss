"""Read raw (pre-filter) LHE events of pp -> Z j, Z -> tau tau -> hadrons into flat arrays.

Runs on ICEPP next to the samples; numpy only.  For each event it keeps the two
incoming partons, the outgoing QCD parton, and for each tau the charged pions,
the neutral pion and the neutrino (lab four-vectors, float64, (px,py,pz,E)).
Only pi / rho / 3-prong (pi pi pi) tau decays are kept; anything else is
counted and dropped.  No kinematic selection is applied.

usage: python extract_lhe.py OUT.npz FILE.lhe.gz [FILE.lhe.gz ...]
"""
import gzip
import sys
from collections import Counter

import numpy as np


def _open_text(path):
    if path.endswith('.tar.gz'):                  # filtered mc.TXT tarballs (v2 merged / v3 filtered)
        import io
        import tarfile
        tf = tarfile.open(path, 'r:gz')
        member = [m for m in tf.getmembers() if m.isfile()][0]
        return io.TextIOWrapper(tf.extractfile(member), encoding='utf-8', errors='replace')
    return gzip.open(path, 'rt')


def events(path):
    with _open_text(path) as f:
        block = None
        for line in f:
            s = line.strip()
            if s.startswith('<event'):
                block = []
            elif s.startswith('</event'):
                yield block
                block = None
            elif block is not None and s and not s.startswith('<') and not s.startswith('#'):
                block.append(s)


def parse(block):
    head = block[0].split()
    nup = int(head[0])
    rows = []
    for s in block[1:1 + nup]:
        t = s.split()
        rows.append((int(t[0]), int(t[1]), int(t[2]), int(t[3]),
                     float(t[6]), float(t[7]), float(t[8]), float(t[9])))
    return float(head[2]), rows


def descendants(rows, idx):
    """final-state descendants of 1-based particle idx."""
    out = []
    for j, r in enumerate(rows, start=1):
        if r[2] == idx or r[3] == idx:
            if r[1] == 1:
                out.append(j)
            else:
                out.extend(descendants(rows, j))
    return out


def main(out, paths):
    keep = {k: [] for k in ('w', 'beam_id', 'beam', 'jet_id', 'jet', 'mode', 'pions', 'charges', 'pi0', 'nu')}
    stats = Counter()
    for path in paths:
        for block in events(path):
            stats['events'] += 1
            w, rows = parse(block)
            inc = [r for r in rows if r[1] == -1]
            taus = {r[0]: i for i, r in enumerate(rows, start=1) if abs(r[0]) == 15}
            if len(inc) != 2 or set(taus) != {15, -15}:
                stats['bad_topology'] += 1
                continue
            tau_desc = set()
            sides = []
            ok = True
            for tid in (15, -15):                      # side 0 = tau-, side 1 = tau+
                d = descendants(rows, taus[tid])
                tau_desc.update(d)
                pis = [rows[j - 1] for j in d if abs(rows[j - 1][0]) == 211]
                pi0 = [rows[j - 1] for j in d if rows[j - 1][0] == 111]
                nus = [rows[j - 1] for j in d if abs(rows[j - 1][0]) == 16]
                other = len(d) - len(pis) - len(pi0) - len(nus)
                sig = (len(pis), len(pi0), len(nus), other)
                mode = {(1, 0, 1, 0): 0, (1, 1, 1, 0): 1, (3, 0, 1, 0): 2}.get(sig)
                if mode is None:
                    stats['other_mode_%d%d%d%d' % sig] += 1
                    ok = False
                    break
                sides.append((mode, pis, pi0, nus[0]))
            if not ok:
                continue
            fin = [r for j, r in enumerate(rows, start=1) if r[1] == 1 and j not in tau_desc]
            if len(fin) != 1:
                stats['jets_%d' % len(fin)] += 1
                continue
            p4 = lambda r: (r[4], r[5], r[6], r[7])
            keep['w'].append(w)
            keep['beam_id'].append([inc[0][0], inc[1][0]])
            keep['beam'].append([p4(inc[0]), p4(inc[1])])
            keep['jet_id'].append(fin[0][0])
            keep['jet'].append(p4(fin[0]))
            keep['mode'].append([s[0] for s in sides])
            pions = np.zeros((2, 3, 4))
            charges = np.zeros((2, 3), int)
            pi0 = np.zeros((2, 4))
            nu = np.zeros((2, 4))
            for k, (mode, pis, p0, n) in enumerate(sides):
                for i, r in enumerate(pis):
                    pions[k, i] = p4(r)
                    charges[k, i] = np.sign(r[0])
                if p0:
                    pi0[k] = p4(p0[0])
                nu[k] = p4(n)
            keep['pions'].append(pions)
            keep['charges'].append(charges)
            keep['pi0'].append(pi0)
            keep['nu'].append(nu)
            stats['kept'] += 1
        stats['files'] += 1
    arrays = {k: np.asarray(v) for k, v in keep.items()}
    np.savez_compressed(out, **arrays, stats_keys=np.array(list(stats.keys())),
                        stats_vals=np.array(list(stats.values())))
    print(out, dict(stats))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
