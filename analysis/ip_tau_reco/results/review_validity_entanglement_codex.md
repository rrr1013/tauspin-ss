review_type: validity  
verdict: revise  
confidence: high (0.95)

Stage acceptance: not met for a paper-level or unqualified entanglement claim. The result is usable as an exploratory, model-conditioned feasibility estimate.

## Major findings

1. Werner test is correct only inside its one-parameter model

- Claim: The likelihood establishes entanglement.
- Evidence: For \(p_0(x)[1+p\,g_s(x)]\), \(g_s=E[s|x]\) is sufficient for \(p\), and \(p=1/3\) is the least-favourable separable boundary within the Werner family. The Asimov free-\(S\), constrained-\(B\) likelihood is internally consistent for this model. However, [the implementation](/Users/ryunosuke/Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/entanglement_feasibility.py:77) compares fixed \(p=1\) and \(p=1/3\) templates; it does not profile arbitrary polarizations, off-diagonal correlations, or the full separable two-qubit state space.
- Consequence: \(3.78\sigma\) is discrimination of two Werner-family hypotheses, not model-independent rejection of all separable states.
- Required action: Fit at least independent \(C_{nn},C_{rr},C_{kk}\) and test the composite null \(C_{nn}+C_{rr}-C_{kk}\le1\), preferably profiling the full physical density matrix under the PPT constraint. Calibrate the boundary statistic with toys. Retain the “standard QM/QFT decay-as-spin-analyser” qualification; collider decay correlations are not an unconditional proof of entanglement ([interpretation discussion](https://arxiv.org/abs/2507.15949)).

2. Missing-polarimeter treatment does not close

- Claim: Unity weights for incomplete polarimeters give a valid, conservative signal model.
- Evidence: Only 71.81% of selected events have both exact polarimeters; 25.85% have one and 2.34% neither. Although inclusive H closure is good—\(13.3/20\) for \(g_s\), \(16.8/20\) for \(g_Z\)—a read-only stratified probe gives:

  - one-side H closure: \(g_s=46.6/20\), \(p=6.8\times10^{-4}\);
  - all missing-side H closure: \(g_s=44.6/20\), \(p=1.3\times10^{-3}\);
  - one-side \(g_Z=51.2/20\), \(p=1.5\times10^{-4}\).

  The missing side’s visible observables and selection are retained, so assigning unity does not physically marginalize that decay.

  There is also a concrete Z-weight bug: [weights()](/Users/ryunosuke/Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/entanglement_feasibility.py:49) zeroes both \(h\) vectors unless both are finite, making the intended one-side Z polarization weight identically one. Correcting only this bug changes \(3.778\to3.790\sigma\), so its numerical effect is small.
- Consequence: The inclusive closure hides nonclosure in 28.2% of the sample; it does not validate the full signal template.
- Required action: Fix the one-side bug and rerun. Either implement polarimeters for the missing modes or provide results excluding them. Perform final cross-fitted 2D closure by both/one/neither-polarimeter status, decay mode, and \(p_T\).

3. Background closure fails and the real-Z variant is not sufficient

- Claim: The spin-flat H sample reweighted to Z adequately models the background.
- Evidence: \(g_s\) closure against real \(Z(125)+j\) is \(187/20\), \(p\simeq3\times10^{-29}\); adding the quoted transverse terms does not improve it. Replacing the proxy by real Z shifts a comparable 10%-response setup from 4.05 to 3.79σ, suggesting an optimistic nominal bias in that comparison. But this variant uses an artificial on-shell \(Z(125)+j\) sample, not the physical \(Z/\gamma^\ast\) mass spectrum and reconstructed Higgs-mass tail, and it is not evaluated with the final cross-fit/category-\(p_T\) setup.
- Consequence: The dominant background template is unvalidated. The full bias direction is unresolved, although the available substitution is downward.
- Required action: Build final, category-specific templates from physical \(Z/\gamma^\ast+\)jets, fakes, top and other processes. Validate them in the joint \((g_s,g_Z)\) plane and propagate control-region constraints and shape nuisances.

4. Yield transfer and quadrature are exploratory approximations

- Claim: ATLAS Run-2 yields can be transferred directly into a paper-level HL-LHC likelihood.
- Evidence: The ATLAS table entries are transcribed correctly, but they are post-fit yields from an analysis with 32 signal regions, 36 control regions, and 31 Z-normalization factors—not independent counting inputs ([ATLAS analysis](https://arxiv.org/pdf/2201.08269)). The projection then applies:

  - common 1.13/1.15 signal/background factors to unlike processes;
  - one mass-acceptance vector in every category;
  - the same 60–200 GeV H+jet template for VBF, VH and ttH;
  - essentially one Z spin template for all backgrounds.

  Signal mass-bin fractions visibly vary with \(p_T\) in the saved H+jet sample. The likelihood also profiles separate \(S\), \(B\), and response nuisances in each category–mass cell before quadrature, i.e. 108 effective response nuisances rather than one global nuisance. A shared response nuisance gives 3.82σ versus 3.78σ, so that particular choice is slightly conservative; missing component correlations can move either way.
- Consequence: Process evolution, category resolution, topology, and nuisance correlations are not controlled. “ATLAS+CMS” is especially not supported: 6 ab\(^{-1}\) is only a doubled ATLAS-like luminosity scenario.
- Required action: Use process-specific 14/13 ratios, category/process-specific mass acceptances and spin templates, and a single joint likelihood with declared correlations. Report a transfer envelope in significance.

5. Template precision and plan locking are insufficient

- Claim: The quoted significance is numerically stable.
- Evidence: With the saved templates and otherwise unchanged setup, the 10%-response result changes as follows:

  - \(3\times3:\ 3.27\sigma\)
  - \(4\times4:\ 3.52\sigma\)
  - \(6\times6:\ 3.78\sigma\)
  - \(8\times8:\ 3.91\sigma\)
  - \(10\times10:\ 3.97\sigma\)
  - \(12\times12:\ 4.02\sigma\), with cells containing as few as eight MC events.

  No template-statistical nuisance is included. No `plan_state`, `plan_lock`, pre-result acceptance rule, or recorded rationale for the binning, mass bins, 2% constraints, or response priors was provided.
- Consequence: Threshold statements depend materially on an unlocked analysis choice and finite-template structure.
- Required action: Predefine the binning using closure/statistical precision, run independent-template or bootstrap studies, include template MC uncertainty, and validate asymptotic significance and coverage with pseudoexperiments.

## Minor findings

- The inspected [closure figure](/Users/ryunosuke/Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/results/figures/ent_templates_closure.png) shows good real-H agreement, but panel (b) again overlays real H—not real Z. No 2D, category, decay-mode, or \(p_T\)-resolved closure is shown; the known Z failure is absent.
- The spin-flat dataset’s tau-ID/event weights are omitted from template construction. Including them in a read-only probe changes \(3.778\to3.840\sigma\), so this omission is mildly conservative but should be fixed consistently, including regression training.
- The response morph is an unsupported scalar toy, not a propagation of tau/\(\pi^0\) energy, IP/SV, decay-model, pileup, MET, or regressor-calibration variations. Positivity and optimizer success are not checked.
- IP/SV improves significance by 14.3% without the response nuisance and 17.3% with 10%; “about 15%” is acceptable only as an explicitly approximate statement.

## Permitted claims

> In an ATLAS-like Asimov feasibility model, a cross-fitted tauspin-level 2D statistic gives a median expected \(3.78\sigma\) separation between the CP-even \(p=1\) Werner hypothesis and its \(p=1/3\) separable boundary at 3 ab\(^{-1}\), conditional on the stated yield transfer, independent 2% background constraints, and 10% spin-contrast nuisance assumptions.

Also permitted:

- \(g_s\) is sufficient for \(p\) within the assumed linear Werner family.
- Inclusive one-dimensional independent-H closure is compatible with the reweighting model, while mode-resolved closure is not yet established.
- Removing IP/SV changes \(3.78\to3.22\sigma\) in this simulation.
- Doubling the ATLAS-like yield gives 5.18σ.

Not permitted:

- “The HL-LHC can establish/observe tau-pair entanglement.”
- “ATLAS+CMS will reach 5.2σ.”
- “The signal or background model is closed.”
- “The quoted response uncertainty represents experimental systematics.”

## Approval conditions

Approval requires:

1. Fixing the one-side Z weight and resolving the missing-polarimeter closure failure.
2. Testing a composite separability null with toy-calibrated coverage.
3. Cross-fitted 2D signal and physical-\(Z/\gamma^\ast\) closure in the dominant categories.
4. Process/category-specific yield and template transfer in a correlated likelihood.
5. Detector-shape and template-MC systematics with a locked binning choice.
6. Restricting the claim to a model-conditioned QM entanglement-witness projection until those conditions are met.

Read-only review completed; no files or external state were changed.