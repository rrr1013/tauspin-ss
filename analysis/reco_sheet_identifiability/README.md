# Reconstructed-geometry sheet identifiability

This directory contains the four work packets used to test whether detector
geometry that is independent of the tau momentum measurement can resolve the
four global tau-neutrino solution sheets.

- `wp_a_label_stability.py` audits whether the truth-surface sheet label remains
  meaningful after rebuilding the exact mass-shell surface from reconstructed
  visible taus and reconstructed MET.
- `wp_b_sv_information.py` fits secondary-vertex direction, impact-parameter
  plane, and secondary-vertex length responses on training data, calibrates
  their four-sheet posteriors on a disjoint training partition, and evaluates
  them on another disjoint training partition. It includes topology-only and
  matched-geometry-shuffle controls.
- `wp_c_spin_recovery.py` freezes the training-derived selector, applies it to
  the canonical validation cohort, and recombines the existing per-sheet spin
  moments. It reports joint, marginal, and likelihood-ratio AUCs with paired
  bootstrap attribution. The validation cohort is historically reused and is
  not treated as a blind test; the test split is never opened.
- `wp_d_reco_surface_sv.py` repeats the SV selector on the exact surface built
  from reconstructed visible taus and reconstructed MET.  It uses two further
  disjoint training hash partitions for posterior calibration and evaluation,
  defines the evaluation label as the reconstructed sheet nearest to the truth
  tau direction, and reports the hard-surface cut flow explicitly.  This packet
  quantifies what is possible on the surface-valid subset; it does not impute a
  four-sheet target for the off-shell majority.

The scripts depend on the `metspace.py` and `sheetspace.py` modules produced by
the earlier sheet-ambiguity analysis. Large input and output artifacts remain
on the analysis host and are intentionally not committed.
