# H/Z spin separation: synthesis of all runs and additional checks

Run `tauspin-hz-spin-synthesis-20260923` (Vault: `ARIADNE/Runs/`).  Large
artifacts stay on lxgpu02 under `~/hz-spin-synthesis-20260923/`.  The test split
is never loaded; every number is on the development validation split.

| script | what it does |
|---|---|
| `s1_readout.py` | fixed-recipe H/Z readout on point-h predictions; hard label or analytic spin-weight soft target; seed ensembles; extra per-event inputs |
| `s2_auc.py` | weighted/unweighted AUCs of arbitrary score files on a common cohort, per decay-mode pair, paired bootstrap |
| `s3_reco_nullspace.py` | kinematic solution set with truth or reco visible momenta (per decay mode) and exact or reco MET (train-fitted resolution); likelihood ratio with IP azimuth, SV, parent mass and the full IP likelihood (`ipg`: anisotropic errors, shared decay length integrated analytically); `--ip-sim-scale` simulates the IP from the truth tau direction |
| `s4_merge.py` | merge row chunks of s3, export per-measure score files |
| `s5_stack_cv.py` | 5-fold cross-fitted stacking of the learned readout with solution-set likelihood ratios |
| `s6_ip_resolution.py` | transverse / longitudinal IP error per (pT, abs eta) from the component orthogonal to the truth tau direction (TRAIN) |
| `s7_additive_readout.py` | additive readout f(h-)+g(h+): single-tau (marginal) information only |
| `s8_geometry_v3.py` | v3 learner geometry: IP with magnitude in the local frame plus its error ellipse, track offset, SV (31 channels) |
| `s9_figures.py` | figures in `figures/` from `results/*.json` |
| `metspace.py`, `nullspace.py` | unchanged copies from the 09-19 MET null-space run |
| `launch/` | the shell launchers used on lxgpu02 |
