review_type: `validity — second round`  
verdict: `revise`  
confidence: `high (0.90)`

Stage acceptance: bin-level result is usable as a conditional simulation estimate. The ATLAS/ATLAS+CMS transfer is not yet a paper-level projection.

## Major findings

1. **The fully defined-event reweighting is sound; missing-polarimeter events are not physically reweighted.**

   - Claim: `TauDecays:mode=3`, `tauPolarization=0` is an appropriate unpolarised, uncorrelated reference. This agrees with the [PYTHIA manual](https://pythia.org/latest-manual/ParticleDecays.html). For events with both exact polarimeters, the implemented densities are bounded, non-negative, and have adequate ESS.
   - Evidence: at the 20% region, ESS is H/Z/W/U = 9470/10951/7183/11761; no negative weights were found. However, [spin_gain_U.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/spin_gain_U.py:166) sets every hypothesis weight to 1 unless **both** polarimeters exist.
   - Consequence: this is an operational “erase spin information” prescription, not full marginal reweighting. With one implemented side, the marginal weights should retain \(1-0.147h_k\) for Z and \(1-h_k\) for W. The current normalization also gives unsupported-event fractions H/Z/W/U = 26.7/27.2/32.9/26.7%, so these events are not actually identical across templates.
   - Read-only probe: proper one-side marginalization changed spin \(R\) from 1.02279 to 1.02372; a fixed-cohort treatment gave 1.02262. The numerical effect appears small, but the measurement semantics must be corrected.
   - Required action: implement side-wise marginalization or explicitly define a both-implemented cohort with a hypothesis-independent missing cohort. Do not describe the present all-event sample as fully reweighted.

2. **The channel/experiment transfer remains assumption-driven, not a calibrated uncertainty envelope.**

   - Claim: the quoted +1.1–2.0%, +0.7–1.4%, and +0.5–1.0% transfers are exact arithmetic consequences of the selected assumptions.
   - Evidence: [final_v2.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/final_v2.py:23) assumes significance-squared additivity, \(f=0.62\), quadrature combinations, identical ATLAS/CMS improvement, and a stat-only top-rich scan. It does not use a bin-level likelihood or correlated systematics.
   - There is also an inconsistency: `top_fraction <= 0.6` excludes the 60.29% point, so the transfer uses 55.86% top with R=1.0346/1.1236, while the packet cites the excluded R=1.0393/1.1324 point.
   - Consequence: the range is a scenario bracket, not a statistical or systematic uncertainty band. Absolute significance values cannot be presented as a paper-level projection.
   - Required action: either derive \(f\), rest-bin gains and correlations from a likelihood-level model, or label the transfer explicitly as illustrative. Fix the 55.9%/60.3% mismatch.

3. **Closure and uncertainty treatment are improved but incomplete for a 2.3% effect.**

   - Evidence: inclusive closure is acceptable: \(D\) χ²=24.7/20. The actual figure was inspected and real-H points broadly follow the reweighted-H template. However, [closure_U.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/closure_U.py:56) trains on the same flat sample used for its reference template and does not test the central 20% K region.
   - A read-only 20%-region probe gave \(D\) χ²=29.2/20 and side-1 \(\hat h_k\) χ²=33.4/21; not a failure, but borderline enough to require the production closure.
   - The reported bootstrap keeps the trained classifier and bin edges fixed; it does not cover training/fold/seed uncertainty. The 25% shape nuisance has no documented derivation, and the final headline transfer uses nominal rather than shape-profiled R.
   - Consequence: ±0.0009 is only conditional template-MC uncertainty, not the total uncertainty on R.
   - Required action: perform out-of-fold closure in the 20% region, refit classifiers in bootstrap or repeated split/seed trials, justify the 25% prior, and propagate it consistently. Validate response versus tau kinematics using background-rich looser regions; all hypotheses currently inherit HH kinematics.

## First-round status

- Resolved: unstable density-ratio weighting/low ESS; explicit kinematic-leakage check; removal of the “exact ceiling” wording; exclusion of lephad from the headline; Z-transverse sensitivity check; C4’s direct-background-MC statistics limitation.
- Partially resolved: spin reweighting closure, missing-mode treatment, classifier statistical stability, spin-model systematic, and transfer uncertainty.
- Still open: fake-tau U model, background-specific kinematic dependence, full-likelihood transfer/correlations, independent detector closure for C3, and documented `stage objective`, `output_contract`, `plan_state`, and `plan_lock`.

## Permitted central wording

> In a factorised, common-HH-kinematics spin-only simulation using the ATLAS Run-2 highest-score τhadτhad composition, a 20%-signal-efficiency kinematic region, and profiled background-normalisation nuisances, tauspin \(\hat h\) binning gives \(R=1.0228\) for the bin-level Asimov discovery significance. With the assumed 25% correlated spin-contrast nuisance, \(R=1.0224\). The fixed-classifier event-bootstrap standard deviation is 0.0009. The exact-\(h\) reference for the implemented modes gives \(R=1.124\), or 1.115 with that nuisance. These are conditional simulation results, not an exact-polarimeter ceiling or a full ATLAS projection.

The transfer may additionally be stated only as:

> Under the illustrative \(f=0.62\) transfer model, the corresponding relative gains are approximately 1.1–2.0% for ATLAS \(bb\tau\tau\), 0.7–1.4% for ATLAS combined, and 0.5–1.0% for ATLAS+CMS.

It must be followed by “scenario range, not an uncertainty interval.” No quantitative \(\kappa_\lambda\) claim is permitted beyond crude significance scaling.