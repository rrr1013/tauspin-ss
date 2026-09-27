# Where along the tau boost does the H/Z spin information live?

Run `tauspin-spin-boost-map-20260927` (Vault: `ARIADNE/Runs/`).  No network is
retrained and no new sample is generated: everything reads the aligned
validation cohort (59,390 events, overlap weights) that the 2026-09-25 CP run
copied into `../cp_mixing/data/` (not tracked, see `../cp_mixing/README.md`
on the `ariadne/auto-2026-09-25-cp-mixing` branch for provenance).  The test
split is never loaded.

Readout: for every arm (exact `h`, and each reco `h_pred`) a weighted logistic
regression on the nine bilinears `h-_i h+_j`, cross-fitted on two halves of the
validation cohort (global-index parity).  It is one fixed function per arm,
applied unchanged in every bin.  Its global AUCs sit ~0.004 below the p11
readout of `hz_beyond_ceiling` for every arm; ordering and paired differences
agree.

| script | what it does |
|---|---|
| `boost_map.py` | cross-fitted scores; AUC and paired AUC differences with bootstrap in 1D bins of beta*gamma (pair geometric mean, min, max), visible pT sum, per mode class |
| `boost_x_grid.py` | beta*gamma vs visible energy fraction x: per-side h-quality grid, event AUC on (x tercile x beta*gamma quartile) with slope fits, AUC vs truth x and collinear-approximation x |
| `mode_profile.py` | per-side transverse h quality vs beta*gamma per truth mode, x reweighted to each mode's inclusive x; visible-axis error / vis-tau angle |
| `ip_boost.py` | PV-referenced IP: |IP| and IP-azimuth error vs beta*gamma for pi and rho 1-prong sides |
| `boost_shift.py` | stress scenario: reweight ln(beta*gamma) by a factor 1.15 / 1.37 (91 -> 125 GeV extreme) |
| `make_figures.py` | figures in `figures/` |

```sh
../../.venv/bin/python boost_map.py && ../../.venv/bin/python boost_x_grid.py
../../.venv/bin/python mode_profile.py && ../../.venv/bin/python ip_boost.py
../../.venv/bin/python boost_shift.py && ../../.venv/bin/python make_figures.py
```

`scores.npz` (per-event cross-fitted scores) is written but not tracked.
