"""Evaluate CLEO (C++) and generator currents on the toy tau- decays."""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from currents import generator_current_with_rate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--toy', required=True)
ap.add_argument('--output', required=True)
a = ap.parse_args()
sys.path.insert(0, '/home/rbaba/tauspin-truth-neutrino-20260906/NN')
cpp = importlib.import_module('neutrino_statistics_20260907.hybrid_polarimeter').compile_current()
t = np.load(a.toy)
p = t['pions']
m = np.full(len(p), 1.777)
q = np.full(len(p), -1)
oc = np.concatenate([cpp(p[i:i + 200000], m[i:i + 200000], q[i:i + 200000]) for i in range(0, len(p), 200000)])
og = generator_current_with_rate(p, m, q)
np.savez_compressed(a.output, out_cleo=oc, out_gen=og)
print(json.dumps({'n': len(p), 'nonfinite_cleo': int((~np.isfinite(oc)).any(1).sum()),
                  'nonfinite_gen': int((~np.isfinite(og)).any(1).sum())}))
