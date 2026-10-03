"""Evaluate CLEO (vendored C++) and generator 3pi currents through the same
HybridPolarimeter code path on the truth surface with truth neutrinos.

Output per event and side: h for both models, omega for both models, and the
ordered tau-rest-frame pions passed to the current (3pi sides only).
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from currents import generator_current_with_rate  # noqa: E402

NN_ROOT = Path('/home/rbaba/tauspin-truth-neutrino-20260906/NN')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(NN_ROOT))
    mod = importlib.import_module('neutrino_statistics_20260907.hybrid_polarimeter')
    pol = mod.HybridPolarimeter()
    cpp = pol.current
    s = dict(np.load(a.surface))
    rec = {}

    def recorder(fn, tag):
        def current(pions, masses, charges):
            out = fn(pions, masses, charges)
            rec.setdefault(tag, []).append((np.array(pions), np.array(masses), np.array(charges), out.copy()))
            return out
        return current

    args = (s['pions'], s['pion_charges'], s['pion_counts'], s['pi0'], s['pi0_counts'],
            s['modes'], s['nu4'][..., :3])
    res = {}
    for tag, fn in (('cleo', cpp), ('gen', generator_current_with_rate)):
        pol.current = recorder(fn, tag)
        r = pol(*args)
        res[tag] = r
        print(tag, 'valid', int(r['valid'].sum()), 'of', len(r['valid']), flush=True)
    # The 3pi current is called once per side (tau- then tau+); recover the mapping.
    out = {}
    for tag in ('cleo', 'gen'):
        out[f'h_{tag}'] = res[tag]['h']
        out[f'omega_{tag}'] = res[tag]['omega']
        out[f'valid_{tag}'] = res[tag]['valid']
    calls_c, calls_g = rec['cleo'], rec['gen']
    assert len(calls_c) == len(calls_g) == 2
    for side, ((pc, mc, qc, oc), (pg, mg, qg, og)) in enumerate(zip(calls_c, calls_g)):
        assert np.array_equal(pc, pg) and np.array_equal(mc, mg) and np.array_equal(qc, qg)
        out[f'ordered_side{side}'] = pc
        out[f'cmass_side{side}'] = mc
        out[f'charge_side{side}'] = qc
    out['global_indices'] = s['global_indices']
    out['h_ref'] = s['h_ref']
    np.savez_compressed(a.output, **out)
    m = s['modes'] == 3
    info = {
        'surface_sha256': hashlib.sha256(a.surface.read_bytes()).hexdigest(),
        'events': int(len(s['modes'])),
        'max_abs_h_cleo_minus_ref': float(np.nanmax(np.abs(out['h_cleo'] - s['h_ref']))),
        'nonfinite_h_cleo': int((~np.isfinite(out['h_cleo'])).any(-1).sum()),
        'nonfinite_h_gen': int((~np.isfinite(out['h_gen'])).any(-1).sum()),
        'omega_cleo_3pi_min': float(np.nanmin(out['omega_cleo'][m])),
        'omega_gen_3pi_min': float(np.nanmin(out['omega_gen'][m])),
        'threepi_sides': int(m.sum()),
        'calls_per_side': [len(c[0]) for c in calls_c],
        'seconds': time.time() - t0,
    }
    Path(str(a.output) + '.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
