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

Result: the classical posterior reaches AUC 0.577 (IP+SV) / 0.581 (+parent mass) vs 0.629 for the
main-line learner; exact MET, an anisotropic MET model and an empirical x prior do not change this.
The gap sits in rho: on reco rho sides with an unphysical reco visible mass (>1.2 GeV, 17% of
correct-rho sides) the exact-formula polarimeter loses its longitudinal component, while the learner
does not.  For pi the classical estimator matches the learner given the same geometry.
