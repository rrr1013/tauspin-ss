# Skeptical review of the lep-had lepton-IP gain (Claude skeptical-reviewer subagent), 2026-10-07

**Verdict: overstated and not yet falsifiable as written.** The direction of the effect is plausible: tau-lepton lifetime does separate from prompt W leptons. But the size of the gain and its transfer to ATLAS rest on an idealised background, an optimistic resolution model and an unresolved baseline (whether ATLAS applies a lepton d0 cut). The factorised "confirmation" uses the same IP templates, so it is not an independent check.

**Major**

1. **Displaced-lepton backgrounds are missing.** In `reco_lephad.py` the candidate leptons are only tau daughters and W/Z leptons, and prompt leptons get a true IP of exactly zero (`np.zeros((n,2,3))`). Nothing models:
   - leptons from b/c decays (multijet, or non-isolated leptons in ttbar);
   - conversions;
   - wrong primary-vertex or pile-up track association.

   ATLAS 2209.10910 §6.1 says the lep-had fake background is a mix of multijet and ttbar (the r_MJ fraction), with the multijet part defined by anti-isolated leptons, i.e. mostly non-prompt. Here every fake gets a prompt lepton (`fakes prompt_frac 0.94`, factorized.json). This inflates exactly the background classes that the IP is supposed to remove.

2. **The "other" template is optimistic, and the code calls it conservative.** In `factorized_lephad.py`, "other" gets the prompt template. That makes it removable by the IP cut. ATLAS "other" includes Z→ττ+light jets and diboson ττ, which have tau-origin leptons. The 50/50 variant mentioned in the docstring is not in factorized.json.

3. **Whether ATLAS applies a d0 cut (TTVA) decides the baseline.** In this simulation, applying TTVA cuts the kinematics-only significance from 1.62 to 0.95 (cls.json). With TTVA the IP gain is only 1.117/0.951 = 1.17. The paper text I extracted does not state a d0 cut for signal-region leptons. Leptons pass "track-quality requirements", and the Z+HF control region requires leptons "compatible with originating from the primary vertex". So the status is ambiguous. If ATLAS does apply TTVA, the claimed gain mixes "remove TTVA" with "add IP" (the code's `R_total_vs_published` is 1.8–2.3). If it does not, a simulation where TTVA alone costs 40% of the significance suggests the tau-vs-prompt separation is too good.

4. **The resolution model is too good.** The nominal σ(d0) is 10 ⊕ 150/pT µm, with no beam-spot term. ATLAS measures d0 relative to the beamline, so the transverse beam-spot width (order 10 µm) adds in quadrature, giving about ×1.4 for pT ≳ 20 GeV. Other gaps:
   - no η dependence;
   - electrons get only ×1.2, with no bremsstrahlung tails;
   - one common 2% ×4 tail.

   The ×1.5 variant (R = 1.17) is closer to a realistic nominal than to a pessimistic bound. I did not verify exact Run-2 or ITk numbers.

5. **The method is unstable beyond the quoted seed spread.** K+IP is a superset of K+d0, so it should score at least as high. Instead K+d0 = 2.149 vs K+IP = 2.019 (LTT: 1.22 vs 0.88), while the seed sd is only 0.03–0.04. K+IP+Hspin has sd 0.686 (one seed gave a top bin with S = 113, B = 1073). The top bin holds about 64% of the signal (S ≈ 181 of 283). LTT fakes have an effective MC count of only 133. The seed loop only reshuffles folds; there is no bootstrap over MC events. So the 1.33 upper end mostly reflects LTT noise and should not be used as "nominal high".

6. **A non-IP feature confounds K+IP.** The IP feature list includes `lip_x` (E_lep/E_tau from MMC, a kinematic variable) and `is_e` (flavour). The baseline K has neither. Part of the ratio may not be lifetime information.

7. **The transfer arithmetic mixes inconsistent inputs.**
   - R comes from an Asimov test with only five normalisation nuisances and no IP-modelling systematics. It is applied to an ATLAS 1.8σ that includes systematics.
   - An IP analysis needs prompt- and tau-template uncertainties: alignment, beam spot, electron d0 tails. These were leading uncertainties in ATLAS R(τ/μ).
   - The HH combination assumes the channels add in quadrature with no correlations.
   - SLT and LTT ratios are averaged arithmetically, not weighted.
   - I did not check the 3.5 = 3.1 ⊕ 1.8 inputs from ATL-PHYS-PUB-2024-016.

**Minor**

- **Factorisation assumption is untested.** In the signal-like region, ttbar, single top, Z+HF and fakes fell back to "whole category". That assumes the ttbar prompt fraction in the most HH-like bin equals the inclusive one. HH-like ttbar may be enriched in tt→ττ, which would lower the gain.
- **The +2% from hadronic polarisation is noise.** 1.647 ± 0.028 vs 1.614 ± 0.024 is consistent with zero; it should say "no measurable gain".
- **Novelty and framing.** Lifetime tagging of tau leptons is established (LEP tau-lifetime work; ATLAS R(τ/μ), arXiv:2007.14040, muon |d0| in ttbar). I did not find or rule out lepton-IP use in HH or H→ττ searches. Novelty is at most the HH application. Calling it an "extension of tauspin" describes a lifetime tag, not polarimetry.

**Strongest supportable wording**

"In a parametric 14 TeV simulation with idealised prompt leptons (no non-prompt, multijet or conversion component) and track resolution without a beam-spot term, adding lepton d0 information to a kinematics-only classifier raises the τlepτhad significance (Asimov, normalisation uncertainties only) by about ×1.15–1.25, depending on resolution. Relative to a baseline that applies the standard lepton d0/z0 vertex cuts, the gain is ×1.17. Applied to the ATLAS HL-LHC projection, this is an illustrative upper range, not a projection."

Drop "confirmed", the ×1.33 figure, and the 4.62σ headline.

**Quick probes that could change the conclusion**

1. **Null control:** set every true IP to zero, or permute IP within (pT, flavour) bins. R must come out ≈ 1.00. Also run K + {x, is_e, σd0} without d0.
2. **Non-prompt injection:** give 30–50% of fakes a b-decay-like d0 (cτ ≈ 450 µm). Give "other" a tau-origin share. Rerun both the direct and factorised estimates.
3. **Realistic resolution:** add a 10 µm beam spot, η-dependent resolution and electron tails of 5–10% at ×3–5 width.
4. **MC-stat robustness:** Poisson-bootstrap the background events. Vary the effective-MC threshold (now 100) to 50, 200 and 400. Require K+IP ≥ K+d0.
5. **Factorisation check:** compute the ttbar and fake prompt fraction in quantiles of the kinematics-only score.
6. **ATLAS baseline:** settle whether ATLAS bbττ lep-had applies TTVA, using the support-note or framework defaults.
7. **IP systematics:** add a ±10–20% width nuisance on the prompt and tau d0 templates to the q0 fit.

**Files reviewed** (repo ~/Projects/tauspin-wt-dihiggs, commit 15fd28e):
- analysis/dihiggs_spin/reco_lephad.py
- analysis/dihiggs_spin/build_lephad.py
- analysis/dihiggs_spin/classify_lephad.py
- analysis/dihiggs_spin/factorized_lephad.py
- analysis/dihiggs_spin/make_figures_lephad.py
- analysis/dihiggs_spin/results/lephad/{cls,cls_hspin,factorized,summary}.json