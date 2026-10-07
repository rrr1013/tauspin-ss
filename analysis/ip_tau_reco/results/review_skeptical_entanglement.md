# Skeptical review (Claude skeptical-reviewer subagent), 2026-10-07

## Skeptical review: tau-pair entanglement in H→τhτh (claim of about 4σ per experiment, 5σ ATLAS+CMS)

**Verdict: revise.** The headline numbers depend on the background shape being known exactly and on a one-parameter null hypothesis. I re-ran the setup two ways (read-only, using the repo's saved outputs). Under a fair separable null the result drops. Under a realistic background mismatch it disappears completely.

The two scripts are `/private/tmp/claude-501/-Users-ryunosuke-Library-Mobile-Documents-com-apple-CloudDocs-Athena/b0903296-e8be-4bd3-a44c-bc01c02efa70/scratchpad/sepnull.py` and `.../scratchpad/mismod.py`. They take their inputs from `ip_tau_reco/outputs/ent/{dataset_HU,ent_final_G,closure_g}.npz`. I changed no repo files.

### Major findings

**M1. A known background shape drives the result, and a realistic mismatch reverses it.** (bias HIGH)
- **What is assumed:** the background g-template is reweighted spin-flat H(125)+j and treated as exact. The only freedom given to it is a 2% normalisation.
- **Evidence:**
  - The closure of spin-flat→Z against real Z(125)+j fails: g_s χ²=187/20. The mean shifts by 0.007, while the full H vs W13 contrast is only 0.030.
  - In the boosted, VH_0 and VBF_0 categories S/B is 0.8–4%. The signal contrast in those categories is 20–100 times smaller than that shift.
  - I built Asimov data with the real-Z(125)+j background and fitted it with the nominal flat→Z template. The preference inverted: −2lnL relative to a saturated fit was 9711 for p=1, 9665 for p=1/3 and 9635 for p=0. Significance is 0.
  - A control with MC noise only (two halves of real Z, fitted against each other) keeps the correct ordering. So the inversion is mismodelling, not MC statistics.
- **Why `ent_realZ.py` misses this:** it puts the same template in both the data and the model, so it cannot detect bias.
- **Required:** non-closure injection tests (data from real-Z / Z(91) / a fake template, fit with nominal). Add free background shape nuisances. Report the fitted p and its bias.

**M2. The null is a single Werner point, not the separable set.** (bias HIGH)
- I scanned the Bell-diagonal separable boundary (|c_n|+|c_r|+|c_k|=1), using the same binned statistic, nominal setup and no response nuisance.
- Tauspin: the minimum is about 3.5σ near C=(0.1,0.1,−0.8), compared with 4.08 for the Werner null.
- Textbook: 1.23σ at C=(0,0,−1), compared with 2.66.
- The textbook observables mostly measure C_kk. A classical longitudinal anticorrelation imitates the data.
- Polarised and off-diagonal separable states were not scanned, so the true minimum may be lower.
- **Required:** profile the null over the whole separable set, using one template per C-matrix component (an exact reweighting, so it is cheap).

**M3. The response systematic is too weak.** (bias HIGH, not quantified)
- **How it is modelled:** in `ent_closure_syst.z_ent_shape`, one scale k multiplies all spin components of both the signal and the Z template. It is profiled separately in each of the 108 category × mass bins, each with its own unit prior.
- **Effects:**
  - Separate per-bin priors penalise a common pull 108 times.
  - A single k lets the large Z background, which mainly has kk correlation, calibrate the transverse analysing power. The H Bell-state signal depends on that transverse part.
- This is why going from 10% to 20% costs only 0.2σ.
- **Required:** use separate transverse and longitudinal responses (per decay mode, including π⁰ scale), correlated across all bins. Show the limit where the transverse response is left free.

**M4. Template MC statistics are ignored.**
- Boosted bins have 61–900 MC events per template bin, against about 8k background data events per bin at 3 ab⁻¹. The template uncertainty is 3–10 times larger than the data uncertainty.
- About 27% of q comes from boosted categories with S/B≈1%. These need the background shape known to better than 0.5%.
- The one robust category is VBF_1 (S/B 0.29, 2.48σ). Together with ttH_1 it gives about 2.6σ before M1–M3 are applied.
- **Required:** Barlow–Beeston treatment. Report the numbers with categories of S/B<5% removed.

**M5. The textbook baseline is weak, so the comparison is unfair.** (bias LOW for textbook)
- The textbook baseline uses only per-side Υ, x and hybrid h from MMC neutrinos. It gets no pair-level products, while tauspin gets hh_pred.
- It also leaves out the φ_CP observables that the LHC actually uses: the impact-parameter method, ρ decay plane, combined IP–ρ, and a1 polarimeter.
- CMS (arXiv:2110.04836) and ATLAS (arXiv:2212.05833) use these. A rough scaling of the CMS expected CP-odd exclusion (about 2.6σ at 137 fb⁻¹, all channels) to 3 ab⁻¹ suggests about 3σ in the transverse-entanglement direction alone. This is an order-of-magnitude estimate only.
- "+15% from IP/SV" and the 1.5× gap over textbook are therefore relative to a weak reference.
- **Required:** a baseline that uses signed φ_CP (IP/ρ/a1) plus Υ products.

**M6. The 5.2σ combination is optimistic.**
- 6 ab⁻¹ of ATLAS-like categories is not the same as CMS plus ATLAS. Shared modelling systematics (Z spin and kinematics, fakes, τ-decay modelling) do not average down.
- Adding categories in quadrature assumes they share no systematics. M1 and M3 break that assumption.

### Minor findings
- **m1.** Fakes are modelled as H-kinematic events with Z or zero spin. Real jet fakes have a different h_pred response (no lifetime IP), so their apparent correlation could be nonzero. A data-driven template (same-sign or anti-ID region) is needed.
- **m2.** The same H(125) kinematic template is used in all nine m_ττ bins (70–200 GeV) and for VBF/VH/ttH, which use gg→H+j kinematics at 60–200 GeV. Z(91) events that migrate into the window are selected on mismeasured MET, which may correlate with the reconstructed h.
- **m3.** The closure and real-Z numbers (4.34 / 4.18) use in-sample regressors (`ent_closure_syst.py` fits and predicts on the same events).
- **m4.** Figure `ent_templates_closure.png` shows only the passing H closure. It should also show real Z against flat→Z (187/20). Figure `ent_needs_lumi_categories.png` should state "Asimov, background shape known, Werner null" in the panel (b) title.
- **m5.** The witness is tied to the CP-even orientation. A CP-mixed Higgs (still maximally entangled) would give a smaller witness, 2cos2φ+1.
- **m6.** Things that bias the result LOW: τhτh only (no τℓτh), a simple mass window, signal floated per bin, a 6×6 binning, and a regressor that was not optimised for this task. These cannot make up for M1.

### Can "entanglement" be claimed at all?
arXiv:2507.15949 (Abel et al.) and 2507.15947 (Bechtle et al.) argue that the measured momenta commute, so a local hidden-variable model can reproduce them. This test assumes quantum mechanics, with the τ decay acting as the spin measurement. The defensible claim is "QM-conditional exclusion of separable spin states", not "establishing entanglement". The packet's caveat says this, but the target-claim wording does not.

### Strongest wording the evidence supports now
"In an idealised Asimov study, the background spin and kinematic shapes are assumed known exactly. In that setting, tauspin-level h raises the expected significance of a QM-conditional entanglement witness in H→τhτh. Against the Werner null it goes from 2.7σ with per-side observables to 4.1σ. Against the full Bell-diagonal separable boundary it goes from about 1.2σ to about 3.5σ (per experiment, 3 ab⁻¹, no response systematic). The absolute significance is not yet established. It depends on sub-percent knowledge of the background g-shape, which the current Z closure does not provide."

### Single strongest alternative explanation for each key number, with a falsifiable check

| Number | Alternative explanation | Check |
|---|---|---|
| 4.1 / 3.8σ | Background-shape assumption plus Werner null | Injection test with real-Z or fakes, plus the separable-null scan. Claim survives if fitted p≈1 within its error and Z stays above 3. |
| 5.2σ | Systematics treated as uncorrelated | Correlated response and shape nuisances across all bins and both experiments |
| 2.7 (textbook) | Baseline without φ_CP | Add IP/ρ φ_CP |
| +15% from IP/SV | Feature asymmetry (hh_pred given only to tauspin) | Give both arms the same product features |

### Unresolved questions
- How large is the background MC sample needed to bring template statistics below data statistics?
- Can Z→ττ calibrate the transverse analysing power in situ?
- What is the separable-set minimum once polarised and off-diagonal states are included?
- How much do τℓτh and Run 3 data add?

Sources: [arXiv:2507.15949](https://web3.arxiv.org/abs/2507.15949), [arXiv:2507.15947](https://arxiv.org/pdf/2507.15947), [CMS arXiv:2110.04836](https://arxiv.org/pdf/2110.04836)