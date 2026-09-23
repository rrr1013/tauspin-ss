# H/Z spin separation beyond the visible+MET kinematic ceiling

Run `tauspin-hz-beyond-ceiling-20260923` (Vault: `ARIADNE/Runs/`).  Large
artifacts stay on lxgpu02 under `~/hz-beyond-ceiling-20260923/`.  The test
split is never loaded.

The ntuple stores xAOD perigee `d0`, `z0` relative to the beam spot.  Earlier
geometry studies used `(-d0 sin phi, d0 cos phi, z0 sin theta)` as a
"directional IP" without subtracting the primary-vertex position, so its
direction was dominated by the PV z coordinate.  This directory rebuilds the
PV-referenced 3-D impact parameter and measures what it adds.

| script | what it does |
|---|---|
| `p1_collect_ip.py` | core-track perigee + PV for all 892,304 canonical rows, row-aligned with the 09-20 geometry audit |
| `p2_ip_azimuth_audit.py` | azimuth of the PV-referenced IP around the track vs the truth tau direction (legacy IP as control) |
| `p3_geometry_features.py` | per-tau local-frame (reco n,r,k) features: PV IP, legacy IP, matched shuffle, truth-direction oracle |
| `p4_train_geometry_local.py` | point-h Transformer, reduced legacy input (tau slots 6-9 replaced by the 4 local features) |
| `p5_ip_nullspace.py` | MET null space of the 09-19 run with an IP azimuth likelihood (train-fitted response), legacy/shuffle controls and a common parent-mass constraint |
| `p6_nullspace_auc.py` | likelihood-ratio AUCs of p5 with paired bootstrap |
| `p7_train_direct_soft.py` | direct H/Z classifier, hard label vs analytic spin-weight soft target |
| `p8_auc_compare.py` | weighted AUCs of fixed h readouts and direct classifiers, paired bootstrap |
| `p9_train_parity.py` | parity-preserving arm: unchanged `train_original_geometry.py`, appended slots filled with the local features |
| `p10_geometry_full.py` | per-tau local-frame geometry: 3 core-track IPs + SV (v1, 16 ch); `--with-track-offset` adds the track offset from the visible axis (v2, 22 ch); `--ideal-ip` oracle |
| `p11_train_full_geometry.py` | parity trainer with GEO_DIM appended channels, GEO_KEY variant, SEED |
| `p12_geometry_nullspace.py` | null space with all-track IP and SV likelihoods (+ parent mass) |
| `p13_readout_variants.py` | fixed-recipe readout on seed-ensembled h and/or per-event geometry quality |
| `plot_*.py` | figures |
