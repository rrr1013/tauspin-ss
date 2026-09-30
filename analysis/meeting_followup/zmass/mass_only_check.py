"""Mass-only H/Z separation and spin densities for v2, v3 and their mass-rescaled versions.

Thin wrapper around the 2026-09-21 zspin-v3 script (NN/zspin_v3_20260921/spin_density_mass.py,
unchanged) with the two rescaled samples added to its SOURCES.
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/rbaba/tauspin-zspin-v3-20260921/NN/zspin_v3_20260921')
import spin_density_mass as sdm

D = Path('/home/rbaba/meeting-followup-20260930/zmass')
sdm.SOURCES = {
    'v2': Path('/home/rbaba/simple-smearing-20260910/data/diagnostics.npz'),
    'v3': Path('/home/rbaba/zspin-v3-20260921/data/diagnostics.npz'),
    'v2r': D / 'data_v2r/diagnostics.npz',
    'v3r': D / 'data_v3r/diagnostics.npz',
}
sys.argv = [sys.argv[0], '--output', str(D / 'mass_only')]
sdm.main()
