# What the learned polarimeter uses on reco rho, and where its cross-side term comes from

Automatic ARIADNE run `tauspin-rho-beyond-exact-20261018` (Vault `ARIADNE/Runs/`), branch
`ariadne/auto-20261018-rho-denoise`.  Large artifacts on lxgpu02 under `~/rho-beyond-exact-20261018/`
(`out/`, `logs/`, one `code-<commit>` copy per deployment).  ATLAS full-reco H/Z cohort, VALIDATION only for every
number (59,390 events, parent-overlap weights); TRAIN for every fitted thing; test never loaded.

Question (from the 10/17 classical-posterior run): on rho x rho the main-line learner (gen3pi p11 full22, AUC 0.621)
beats the exact polarimeter evaluated on the reco visible rho with the TRUTH neutrinos (0.606).  Why?

| file | what it does |
|---|---|
| `s1_reference_ladder.py` | exact formula (generator-current HybridPolarimeter) on reco visible + truth nu, TRAIN and VALIDATION; on truth-rho reco-rho sides the pi0 direction / pi0 energy / pi0 / charged pion replaced by truth one at a time; closure to the exact h when both are truth |
| `s2_readouts.py` | H/Z AUC by truth pair with the exact-LR formula and with the unchanged fixed MLP readout recipe trained on TRAIN; learner rows; paired bootstrap |
| `s3_side_quality.py` | rho-side correlation with the exact h vs the reco error (pi0 angle, E ratio, y, m_vis, 1p1n/1p2n); cross-side regression h_pred[side] ~ a h[side] + b h[other] within H and within Z |
| `s4_denoised_reference.py`, `s4_eval.py` | E[h | reco rho, truth nu] learned on TRAIN (2-fold cross-fit, TRAIN-holdout early stopping); its quality and AUC |
| `s7_tau_shell_refit.py` | classical pi0 refit with the truth nu: minimum resolution-weighted move onto (pi + pi0 + nu)^2 = m_tau^2 (resolutions from a TRAIN subset) |
| `s5_make_subset_targets.py`, `s5_train.py` | training-composition control: H only / Z only / random half (base16 arm, seed 43, unchanged recipe); shims for the row-aligned oracle truth file and for single-class sampling |
| `s5_train_flatw.py` | same recipe on the full TRAIN with the per-event MSE weighted to a spin-flat population, w = clip(1/(1 + h-^T C h+), 0, 10) |
| `s5b_make_flat_subset.py` | first, failed attempt at a spin-flat subset by accept-reject (clipping leaves 1/3 of the H correlation); not trained |
| `s6_composition_eval.py` | per arm: cross-side b, plug-in 9<h-h+> by label, the 10/01 static dphi harmonic, rho quality, formula AUC |
| `s8_cp_eval.py` | 10/17 classical posterior with the reco pi0 vs truth pi0 / truth pi0 direction (`../atlas_classical_posterior/posterior.py --pi0`) |
| `make_figures.py` | figures in `results/` |

Results are summarised in the Vault run note; JSONs in `results/`.
