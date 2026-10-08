# Calibration response of the learned polarimeter

Autonomous ARIADNE run `tauspin-calibration-response-20261003` (branch
`ariadne/auto-20261003-calib-response`).  The fixed 9/30 point-h networks (ATLAS sample,
seed 42, arms `none` and `local`) and the H/Z readouts frozen by the 2026-10-02 azimuth run are
evaluated on validation inputs with physically consistent calibration shifts
(`calib_transform.py`): pi0/PFO energy scale, charge-symmetric track momentum scale,
sagitta-type charge-dependent track bias, and a MET-only scale.  No retraining; the test split is
never loaded.

Run order:

1. `probe_remote.py` (ICEPP): closure of relative/event features and visible-tau composition.
2. `evaluate_remote.py --arm none|local` (ICEPP GPU): predictions for the fixed variant grid.
3. Offline analysis and figures on the Mac (`analyze.py`, `make_figures.py`).
