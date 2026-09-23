# Versioned result bundle

This directory contains the compact development-validation outputs for
`tauspin-sv-likelihood-posterior-20260922`.  Large posterior arrays and model
checkpoints remain in the ICEPP artifact store; only reviewable reports,
tables, and figures are versioned here.

Primary figures:

- `raw_sv_geometry.png`: continuous-candidate angular compatibility, including
  the full range, logarithmic tails, core zoom, and truth-nearest competitors.
- `posterior_weight_quality.png`: ESS, maximum weight, entropy, and collapse
  tails for the calibrated response and controls.
- `h_event_distributions.png`: event-level h errors, cosine, paired scatter,
  and error differences, with a separate 3p x 3p view.
- `posterior_sample_stability.png`: fixed-prefix 32/64/128/256-draw stability.
- `spin_discrimination.png`: fixed-readout ROC and score distributions.
- `summary_ladder.png`: baseline to calibrated SV likelihood to direction
  oracle for h MSE, h cosine, and weighted H/Z AUC.
- `readout_training.png`: fixed-endpoint train/development-validation BCE.
- `additional_controls_and_support.png`: offset-preserving and visible-axis
  controls plus frozen-posterior truth-direction support.
- `extra_control_auc.png`: deterministic-readout comparison against those
  controls and the no-SV placebo cohort.

Machine-readable outputs:

- `representation_report.json` and `readout_report.json` are the complete
  compact reports.
- `h_metrics_technical_closure.csv` and
  `h_metrics_canonical_diagnostic.csv` keep the two h target definitions
  separate.
- `auc_metrics.csv`, `bootstrap_auc.csv`, and `posterior_quality.csv` contain
  the principal scalar comparisons.
- `summary.json` records paired uncertainties, response provenance, the
  offset-preserving controls, and bootstrap-bounded response-relative gaps.
- `train_extra_controls_report.json`, `validation_extra_controls_report.json`,
  and `extra_control_readout_report.json` contain the stricter collinearity
  controls, reconstructed-mode cohorts, exact-surface intersections, and
  no-SV placebo.
- `gap_recovery_bootstrap.json` gives bootstrap intervals for the
  response-relative truth-direction arm and the truth-neutrino-relative gap.

The final deterministic readouts use a same-run unweighted baseline and reset
every paired model after seeding.  The primary inference is the mean-h arm;
the full-posterior readout is supporting evidence because no matched full-set
shuffle was trained.  The truth-direction reference applies the same tempered
response and is therefore response-relative, not a detector-independent
information oracle.  The original matched random-direction shuffle is retained
only as a sanity check because its weights are nearly uniform; the
offset-preserving shuffle is the operative geometry null.  The final analysis
implementation is on commit `bae8cf3`; this directory versions its compact
outputs in a later result-bundle commit.  No test split was loaded.
