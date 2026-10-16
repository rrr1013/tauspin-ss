# Classical posterior polarimeter on the ATLAS full-reco H/Z cohort

Run `tauspin-atlas-classical-posterior-20261017` (Vault: `ARIADNE/Runs/`, automatic run, branch
`ariadne/auto-20261017-atlas-classical-sv`).  Large artifacts on lxgpu02 under
`~/atlas-classical-posterior-20261017/`.  Validation cohort only (59,390); test never loaded.
Responses (IP azimuth, SV angle, MET, x prior) are fitted on TRAIN.

| file | what it does |
|---|---|
| `posterior.py` | spin-flat prior sampling around the reco visible taus, IP (3 core tracks, p12 response) / SV / MET (iso, aniso, exact oracle) / optional x prior and parent mass; exact generator-current polarimeter on retained pairs; H/Z likelihood ratio and moments |
| `evaluate.py` | weighted AUCs by truth / reco mode pair, paired bootstrap, cross-fit combination with the main-line learner |
| `hquality.py` | per-side correlation of each h component with the exact h |
| `make_figures.py` | three figures in `results/` |
| `rhosplit.py`, `yres.py`, `rhok.py` | rho diagnostics (mode migration split, energy-asymmetry resolution, h_k vs reco visible mass) |

Result (corrected version: |M|^2_unpol per side draw for rho/3pi, rho shell mass capped at the TRAIN
95% quantile 1.11 GeV): the classical posterior reaches AUC 0.581 (IP+SV) / 0.590 (+parent mass) vs
0.629 for the main-line learner (first version without |M|^2: 0.577 / 0.581).  Exact MET (oracle)
gives 0.595, an anisotropic MET model and an empirical x prior do not change the picture, and a
cross-fit combination adds +0.0006 to the learner.  The gap sits in pairs with rho (rho x rho -0.047):
there the exact formula evaluated on reco visible rho with the truth neutrinos already caps at 0.606
(learner 0.621), and the classical neutrino posterior loses most of the rest.  With exact MET the
classical matches the learner in 3pi x 3pi and uses the SV more strongly (+0.03 vs +0.009).
`review_checks.py` and `results/review_checks_and_rho_unpol.txt`: review follow-up checks.
