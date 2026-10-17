# Review packet — rho_beyond_exact (2026-10-18)

Read-only review.  Do not modify files.  All code is in this directory (`analysis/rho_beyond_exact/`) plus
`analysis/atlas_classical_posterior/posterior.py` (`--pi0` option added in this run).  Result JSONs in `results/`,
figures `results/fig1_rho_auc_ladder.png`, `results/fig2_rho_side_quality.png`, `results/fig3_cross_side.png`.
Large artifacts on lxgpu02 (`ssh icepp-gpu`, `~/rho-beyond-exact-20261018/`) — read-only access is fine.

## Context

ATLAS full-reco H→ττ / Z→ττ τhτh sample, both generated at 91.19 GeV (same mass by construction), H spin state
C_H = diag(1,1,−1) and this Z (two-step generation) C_Z = diag(0,0,+1) in the (n,r,k) pair basis.  `h` = exact
polarimeter vector (generator current), unit length; the learner (“point-h”, Transformer on reco inputs) regresses
the exact h of both τ jointly with MSE on a balanced H/Z TRAIN stream; a fixed MLP readout on the 6 predicted numbers
gives the H/Z score.  All numbers: VALIDATION 59,390 events, parent-overlap weights; TRAIN for every fitted object;
test never loaded.  Learners and readouts other than those retrained here were selected on validation in earlier runs.

The previous run (10/17) found the learner beats “the exact formula evaluated on reco visible ρ + truth ν” on ρ×ρ
(0.621 vs 0.606) and called the latter an upper bound for the formula approach.

## Claims to review

C1. The 10/17 reference is not an upper bound.  With the same information (reco ρ + truth ν):
  - matched readout does not explain the gap (MLP readout on the reference +0.003; learner − reference = +0.013
    [+0.005, +0.023] on ρ×ρ; only ρ×ρ, not π×ρ or ρ×3π);
  - replacing the π⁰ direction by truth gives +0.047 (energy +0.009, charged π +0.004, full π⁰ +0.102);
  - a classical refit of the π⁰ onto the τ mass shell (π + π⁰ + ν_truth)² = m_τ² (resolution-weighted minimum
    move, resolutions from TRAIN) gives 0.664; a TRAIN-learned E[h | reco ρ, truth ν] gives 0.677; both above the
    learner.  The learner is flat vs the π⁰ error while the formula collapses (fig 2).
  Claim boundary: says nothing about *how* the learner achieves robustness; refit/denoiser use truth ν (diagnostic only).

C2. In the 10/17 classical posterior (no truth ν), the π⁰ measurement is the main ρ limitation: truth π⁰ on
  truth-ρ reco-ρ sides raises ρ×ρ 0.575 → 0.638 (learner 0.621), overall 0.590 → 0.626 (learner 0.629); direction
  only 0.600.  Oracle only; no realistic π⁰ marginalisation was implemented.

C3. The learner’s own-side h carries ≈ 0.09 × C_train × (other side’s exact h), where C_train is the spin correlation of
  its training population.  Evidence: retraining the same arm (geometry hidden, seed 43, unchanged recipe) on H only,
  Z only and a random half (same size), regression h_pred[side,c] = a h[side,c] + b h[other,c] + const within H and
  within Z test events (s6).  b(n,r) +0.09 / +0.045 / 0.00 and b(k) −0.07 / ≈+0.01 / +0.09 for H only / mixed / Z only,
  independent of the test label.  Consequences: the raw plug-in 9⟨h⁻h⁺⟩ of the mixed learner on Z→ρρ is
  (+0.29, +0.32, +0.35) vs exact (−0.02, −0.03, +0.93); the static Δφ first harmonic in Z (0.51, the unexplained 10/01
  observation) becomes 0.008 with Z-only training.  H−Z contrast and H/Z AUC change little (AUC 0.613 mixed-half vs
  0.610 single-class).  π sides carry an extra ≈ +0.035 in b(k) in every arm (proposed: same-mass kinematic link,
  untested).
  Implementation details to check: single-class arms use a shim that splits the one class into two pseudo-classes
  for the balanced sampler (label not an input, not in the loss); oracle-truth file row alignment shim.

C4. (spin-flat weighted training, `s5_train_flatw.py`) — see the results section appended below when present.

## What I want from you

- validity: are the estimators, comparisons, CIs, baselines and shims correct; is anything mis-specified
  (e.g. basis conventions in s4 features, refit constraint/linearisation, cross-side regression design, the weighted
  loss patch); do the figures show what the captions say; what view is missing that you would need to believe C1–C4.
- skeptical: alternative explanations for each claim, overstatement, hidden dependence (checkpoint selection on
  validation, single seed per composition arm, same-mass sample, the canonical basis), what result would falsify it.
Return findings ranked by severity with file/line or artifact references.  Do not edit anything.
