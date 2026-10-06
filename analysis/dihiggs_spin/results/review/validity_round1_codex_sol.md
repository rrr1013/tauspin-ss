review_type: validity  
verdict: revise  
confidence: high (0.92)  
stage_acceptance: not accepted for paper-level claims or projected ATLAS/ATLAS+CMS sensitivity. Acceptable only as an exploratory, assumption-conditional estimate.

## Claim assessment

| Claim | Verdict | Assessment |
|---|---|---|
| C1: realistic gain of 1.7–3.8% and exact ceiling of 9.6–12.6% | Partial / revise | The code reproduces these bin-level numbers under its stated templates and nuisance assumptions. The range is not internally uniform, and transferring it to an entire channel is unsupported. “Exact ceiling” is also too strong because exact polarimetry exists for only 71.2% of the phase-2 events. |
| C2: phase-2 loss follows from the 44% Z and 28% single-H composition | Partial / revise | The quoted fractions and qualitative ordering are reproduced for the old Table-5 proxy. The causal wording is too strong: unsupported decay modes, simplified Z polarization, fake/single-top hypotheses and matching choices also affect the loss. |
| C3: parametric detector closure validates the phase-2 response | Partial / revise | The valid-mode AUC closes, 0.627 versus 0.626, but the all-event AUC is 0.606 and the response was calibrated against the same ATLAS cohort. This is conditional calibration closure, not an independent detector validation. |
| C4: the direct classifier is MC-limited in the high-score region | Supported with qualification | The dominant non-single-H backgrounds have only 0–2 events in the highest-score bins, occasionally seven in the broader top-30% region. Single-H remains populated, so the statement must not be phrased as “zero background.” |

The W and Z longitudinal sign conventions pass an independent physics check. In the source samples, the W sample has \(\langle k_-\rangle\simeq-0.982\), \(\langle k_+\rangle\simeq-0.948\), while the Z sample has means near \(-0.16\), consistent with the implemented signs in [phase1_reweight.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/phase1_reweight.py:47). This does not validate the omitted transverse Z correlations.

## Major findings

### M1 — The projection applies a top-bin gain to the whole channel

- **Claim:** The reported bin-level \(R\) can be propagated to the projected \(b\bar b\tau\tau\), ATLAS and ATLAS+CMS sensitivities.
- **Evidence:** [projection.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/projection.py:25) multiplies the full \(3.1\sigma\) hadronic channel by the top-bin \(R\). This is equivalent to assuming that the affected bin contains all of the channel’s Fisher information. If its information fraction is \(f\), the channel factor should instead be
  \[
  R_{\rm channel}=\sqrt{1+f(R_{\rm bin}^{2}-1)}.
  \]
  For the realistic phase-2 result, varying \(f=0.25,0.5,0.75,1\) gives \(Z_{bb\tau\tau}=3.511,3.522,3.533,3.544\) before the separate lepton-hadron assumption, rather than the reported 3.558. The lepton-hadron increment itself is derived from phase 1 and is combined using an unexplained 0.75/0.25 linear average at [projection.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/projection.py:43).
- **Consequence:** The final projected significances are optimistic upper-scaling scenarios, not likelihood-derived expected sensitivities.
- **Required action:** Obtain the bin-wise signal/background and covariance or public statistical workspace for the analysis underlying the projection, inject the spin templates only into affected bins, and recompute the profile likelihood. If unavailable, publish the result as an \(f\)-dependent envelope and omit the lepton-hadron gain until it has its own realistic-mode evaluation.

There is also an external-validity mismatch: the composition comes from the older [ATLAS Table-5 analysis](https://arxiv.org/html/2209.10910), whereas the HL-LHC projection uses the newer analysis documented in [ATL-PHYS-PUB-2024-016](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff), based on the superseding [2024 \(b\bar b\tau\tau\) analysis](https://arxiv.org/html/2404.12660).

### M2 — The pooled-density reweighting premise has not closed

- **Claim:** `matched_rw.py` creates a valid common-kinematics pooled density and isolates spin information.
- **Evidence:** The algebraic classifier ratio \(p/(1-p)\) is correct because class priors are equalized, and the count-weighted pool is correct conditional on a common underlying kinematic density. However:

  - The moment-inferred Z density has a diagonal correlation component of 1.071 and becomes negative for 107 of 105,545 events; [matched_rw.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/matched_rw.py:31) silently clips it to zero.
  - After matching, the actual H sample has approximate diagonal spin moments \((0.911,1.002,-0.857)\), while the pooled construction reweighted to H gives \((1.089,1.044,-1.176)\). The analogous Z closure also fails appreciably.
  - Matching balances the listed one-dimensional variables reasonably well, with standardized mean differences below about 0.053, but it omits `kin_m_tautau` and `mmc_status`, which are inputs to the Higgs readout in [train_h.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/train_h.py:24).
  - The matcher weights are skewed: median about 0.017, 90th percentile 3.6 and maximum 13.3. No artifact displays multidimensional closure or score independence after reweighting.

- **Consequence:** The measured gain can still contain residual kinematic/source information, while clipping and pooling can also suppress genuine spin information. Its net bias direction is not established.
- **Required action:** On untouched folds, show:

  1. before/after distributions and correlations for every readout-conditioning variable, including \(m_{\tau\tau}\) and reconstruction status;
  2. closure of known H and Z spin moments after pool construction;
  3. cross-fitted density-ratio and matching estimates;
  4. weight clipping/regularization scans with ESS and resulting \(\Delta R\);
  5. the final score versus each matching variable and versus source identity.

A dedicated common-kinematics generated sample would be the cleanest small production probe if the empirical closure continues to fail.

### M3 — The significance definition and nuisance models are inconsistent

- **Claim:** The reported \(R_{\rm syst}\) values are comparable expected-significance gains.
- **Evidence:** [classify.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/classify.py:39) implements the standard known-background Asimov expression, but discards bins with \(b=0\). Those empty bins cannot simply be removed when estimating sensitivity from finite MC. `fisher_z` is a local Wald/Fisher calculation at \(\mu=1\), not the discovery \(q_0\) likelihood used for published expected significances. Moreover, phase 1 assigns nuisances per process, while phase 2 groups them into Z/top/fakes/single-H. Recomputing phase 1 with the phase-2 grouping changes:

  - exact: \(R=1.1258\rightarrow1.1430\);
  - reco-IPSV: \(R=1.0379\rightarrow1.0434\).

  The assumed 10%, 10%, 20% and 15% normalizations have no derivation, correlation model or accompanying shape uncertainties.
- **Consequence:** The 1.7–3.8% interval is not an apples-to-apples method comparison, and the systematic gain is model-dependent.
- **Required action:** Use one likelihood implementation throughout, with discovery \(q_0\), consistent process grouping and correlations, weighted-template statistical uncertainties, and minimum-background/merging rules. Add at least normalization and shape variations for spin response, matching, decay-mode composition, Z modeling and fake templates. Report the change in \(R\) under each variation.

The 20-bin choice itself is reasonably stable near the tested plateau, but that does not replace template-statistical treatment.

### M4 — “Exact ceiling” and several background spin hypotheses are not physical ceilings

- **Claim:** The exact result bounds the available spin sensitivity.
- **Evidence:** The exact polarimeter is defined for 71.17% of the phase-2 sample. Invalid modes are assigned unity/isotropic weights, so 28.83% are assumed to carry no information. The Z model retains only a longitudinal polarization term and \(k_-k_+\), while `other` is assigned the Z hypothesis, single top the fully polarized W-pair hypothesis, and fakes an unpolarized hypothesis without truth/control-region validation.
- **Consequence:** The reported 9.6% is a ceiling only for the implemented-mode and background model, not a truth-level or detector-level physical ceiling. Both upward and downward changes are plausible.
- **Required action:** Perform minimal mode and process probes:

  - quote the gain separately for supported and unsupported decay modes and bracket the latter between zero and exact-mode performance;
  - compare the simplified Z density against a full TauSpinner/Pythia spin density in selected mass and \(p_T\) bins;
  - measure the true-\(\tau\) origin and decay composition of single-top and “other” processes;
  - derive or bracket the fake-template response from a data control region or fake-enriched simulation.

### M5 — The closure evidence is conditional and visually incomplete

- **Claim:** The detector/readout model closes well enough for realistic phase-2 use.
- **Evidence:** [closure_hz.json](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/outputs/v1/closure_hz.json) reports AUC 0.627 only on the valid-polarimeter subset; the all-event AUC is 0.606. The response correlations are approximately 0.61–0.73, with visible nonlinear compression and broad tails in [phase2_closure_h_regression.png](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/outputs/figures/phase2_closure_h_regression.png). The parametric response in [reco.py](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/reco.py:4) was calibrated using the same ATLAS cohort used for comparison.
- **Consequence:** This verifies reproduction of one calibrated aggregate metric, not generalization across modes, kinematics, samples or detector/systematic conditions.
- **Required action:** Provide mode-by-mode and kinematic-bin response, bias, resolution, tail fraction, AUC and score-template comparisons. Reserve one independent sample or generator/configuration for validation and repeat after detector-response variations.

### M6 — The main phase-2 claim lacks a viewable primary template artifact

- **Claim:** The figures substantiate the phase-2 matched-template gain.
- **Evidence:** The phase-1 figure visibly shows the expected exact separation and diluted reconstructed separation. The closure figure shows the compressed response. But [final_spin_gain_projection.png](/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/outputs/figures/final_spin_gain_projection.png) contains only aggregate bars and scalar comparisons: it does not display the matched score templates, process composition, matching closure, weighted uncertainty, bin-wise \(s/b\), or nuisance impacts.
- **Consequence:** The central phase-2 result cannot be independently visually assessed for shape pathologies or localization of the gain.
- **Required action:** Add the matched phase-2 signal and per-process background templates, before/after matching ratios, weighted statistical bands/effective entries, score-versus-matching-variable views, and bin-wise information or \(s/b\).

## Minor findings

1. Table-5 entries and \(3000/139\) luminosity scaling are reproduced correctly. The blanket single-H factor 1.15 is only an approximation to the process-dependent factors in the HL-LHC note and should be documented or varied.

2. C4 is valid as a reason not to use the direct classifier for the headline estimate. Nevertheless, dropping \(b=0\) bins in `asimov_z` is not an acceptable workaround; the conclusion must remain “insufficient MC support.”

3. Rejecting `factorized.py` is justified: it shows kinematic leakage and even gives a reconstructed gain above its corresponding exact estimate. The pooled method also has much better ESS than direct H-only importance weighting. This justifies choosing `matched_rw.py` operationally, but does not validate it physically.

4. The direction of bias from `matched_rw.py` is unresolved. A read-only H-only importance comparison gives a somewhat larger gain but has very poor ESS and extreme weights; it is not evidence that pooled matching is conservative. Residual kinematic leakage can bias upward, while pooling, clipping, missing modes and simplified densities can bias downward.

5. No explicit `plan_state`, `plan_lock`, stage acceptance criterion or output contract was found. The packet’s boundary against quantitative \(\kappa_\lambda\) claims is appropriate, but permitted claims need to be recorded explicitly.

## Permitted use now

The artifacts support the following restricted statement:

> Under the old Table-5 composition proxy, the implemented supported-mode spin models, pooled matching procedure and chosen Fisher-normalization likelihood, the study obtains approximately 1.7% realistic and 9.6% implemented-exact improvement in the selected top-bin phase-2 setup; the phase-1 proxy gives approximately 3.8% and 12.6%. These are exploratory bin-level estimates, not projected full-analysis sensitivities.

The evidence does not currently support:

- calling the exact result a universal truth ceiling;
- presenting the absolute ATLAS or ATLAS+CMS projections as paper-level sensitivities;
- claiming independent detector closure;
- claiming the phase-2 loss arises only from Table-5 composition;
- quantitative \(\kappa_\lambda\) or coupling constraints.

## Unresolved questions

- What fraction \(f\) of the full channel’s likelihood information is carried by the affected category/bin?
- What are the composition and category definitions in the updated analysis underlying the HL-LHC projection?
- Does the pooled-density construction close when conditioned on the complete readout input set?
- How much do full Z spin correlations and unsupported decay modes change \(R\)?
- What spin hypotheses are appropriate for single-top, mixed “other,” and fake components?
- Would a CMS implementation have comparable reconstruction and spin-readout performance?

## Approval conditions

Paper-level approval requires:

1. likelihood-level rather than whole-channel scalar transfer;
2. a common significance/nuisance model with template statistics and systematics;
3. demonstrated multidimensional matching and pooled-density closure on held-out data;
4. corrected “implemented exact” terminology and validated process spin hypotheses;
5. independent, category-resolved detector/readout closure;
6. viewable phase-2 templates and matching/systematic diagnostics;
7. either updated-analysis composition or an explicit composition/information-fraction envelope.