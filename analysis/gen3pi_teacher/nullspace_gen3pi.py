"""Run hz_beyond_ceiling p12 (geometry-weighted null space) with the generator 3pi current.

The only change is the 3pi current inside the HybridPolarimeter instance that
p12 loads; pass --surface with h_ref from the same current (build_targets.py
--surface) so that p12's truth closure is a test of the replacement.

env: P12_DIR (directory with p12_geometry_nullspace.py, metspace.py, nullspace.py)
usage: python nullspace_gen3pi.py <p12 arguments>
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ['P12_DIR'])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p12_geometry_nullspace as P12  # noqa: E402
from generator_current import install  # noqa: E402

_load = P12.ns.load_polarimeter


def load_patched(root):
    polarimeter, origin = _load(root)
    install(polarimeter)
    return polarimeter, origin


P12.ns.load_polarimeter = load_patched

if __name__ == '__main__':
    P12.main()
