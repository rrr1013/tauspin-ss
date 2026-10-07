# Review packet v2: tau-pair spin entanglement and the tau Yukawa CP angle in H → τhτh with tauspin-level polarimetry

2026-10-07. Revision of `review_packet_entanglement.md` after two independent reviews (`review_validity_entanglement_codex.md`, `review_skeptical_entanglement.md`; both: revise).

## What changed and why

| Review finding | Change (code) | Effect |
|---|---|---|
| The background spin shape was taken from a reweighted simulation as exact. Its closure against real Z fails (χ² 187/20), and a background mismatch can reverse the preference. | The genuine-τ background and the fake background each get a free shape per category in the spin statistic, shared across the nine m_ττ bins. Because signal and background composition differ between mass bins, the sidebands measure the shapes in situ. A second scenario adds an auxiliary measurement (embedding or fake-factor control sample) of τ × the signal-region yield (`ent_robust.py`). | The headline drops from 3.8σ to 1.8σ (sidebands only) or 3.0σ (τ = 1). |
| The null was a single Werner point, not the separable set. | The null is profiled over all separable Bell-diagonal states, \|c_n\|+\|c_r\|+\|c_k\| ≤ 1, using a generalized LR against all physical Bell-diagonal states. Both statistics, (g_T, g_L), are cross-fitted. The two H-facing faces of the octahedron and of the witness region coincide. | tauspin: Werner null 2.05σ, separable set 1.84σ. Textbook observables lose the most, because they mainly measure C_kk. |
| One response scale was applied per cell, i.e. 108 independent nuisances. | Global k_T and k_L are used, correlated over all bins. Priors are 0, 10 and 20 %, and k_T is also left free. | With a 0–20 % prior the change is ≤ 0.02σ. **With k_T free the entanglement test collapses to 0.15σ**: establishing entanglement needs the transverse analysing power to be calibrated. |
| Template MC statistics and binning. | The background templates are no longer MC. A bootstrap and half-sample check of the old setup gives ±0.03σ. The 6×6 binning is fixed, with 4×4 and 8×8 run as variants. | tauspin: 1.68 / 1.84 / 1.93σ. |
| The textbook baseline was weak (no φ_CP, no pair products). | Three baselines are added: `textbook_pair`; `phicp`, with lab-frame IP, ρ-plane and SV transverse directions; and `phicp_zmf`, the standard φ*_CP of the LHC CP analyses (IP method for 1p0n and 3-prong, ρ method with the y sign for 1pXn, ZMF of the charged tracks; `phicp_zmf.py`), combined with all of the above in one regressor. The φ* modulation is checked: ⟨cos φ*⟩ = −0.033 for CP-even and +0.036 for CP-odd in the reweighted flat sample. | This is the strongest baseline, and the reference for all comparisons below. |
| The 6 ab⁻¹ figure was presented as ATLAS+CMS. | It is now labelled "doubled ATLAS-like luminosity in one fit with global nuisances", not an ATLAS+CMS combination. | — |
| The one-side Z weight had a bug (both h were zeroed). | Fixed in `entanglement_feasibility.weights`. Event weights are now used in templates and regressions. | Codex read-only probe: 3.778 → 3.790σ. |
| A sufficiently realistic background mismodelling could fake entanglement. | Stress tests: the true genuine-τ background shape drifts across m_ττ, either from Z-spin to unpolarized or from H+jet kinematics to real Z(125)+jet kinematics, which the fit model cannot absorb. The injected signal is the best-fit separable state. | False significance: tauspin 0.40σ and 0.18σ; φ_CP 0.38σ; textbook 0.24σ and 0.84σ. With the H signal under kinematic drift, tauspin drops to 0.55σ. |

Events without both polarimeters keep weight 1. The residual correlation that real H carries in those events (Codex: one-side closure fails) is information this analysis does not use. Because data model and fit model are the same, this is conservative here. A real analysis would model those decays with full-mode simulation.

## New observable: the tau Yukawa CP-mixing angle

ρ_φ = 1 − h⁻_k h⁺_k + cos 2φ (h⁻_n h⁺_n + h⁻_r h⁺_r) + sin 2φ (h⁻_n h⁺_r − h⁻_r h⁺_n). This is maximally entangled for every φ. The statistic is (g_T, g_A), with the same backgrounds, categories and nuisances (`ent_cp.py`). The expected 68 % half-width comes from the small-angle quadratic of the Asimov profile likelihood.

- φ is a phase, so its precision is unchanged when k_T is free: 13.9° → 14.7°. Only the CP-odd vs CP-even sign test needs k_T.
- Calibration against data. With the strongest baseline (φ*_CP), our τhτh-only framework at 139 fb⁻¹ gives ±95° with sidebands only. Extrapolating the known-shape result (×4.98) gives about ±49°. ATLAS Run 2 (all channels, including τℓτh) expects ±28° (arXiv:2212.05833), and CMS expects about ±21° (arXiv:2110.04836). **Our absolute numbers are therefore conservative by roughly a factor 2–3**: cut-based categories, τhτh only, and a data-driven background shape. The robust content is the *ratio* between reconstruction levels within the same framework.

## Results (3 ab⁻¹, 14 TeV, one ATLAS-like experiment, τhτh only, Asimov)

| reconstruction | entanglement vs all separable states: sidebands only / τ=1 / τ=10 / known shape [σ] | σ(φ_τ): sidebands only / known shape [deg] |
|---|---|---|
| exact h (reference) | 5.2 / 8.0 / 8.8 / 8.9 | 4.5 / – |
| **tauspin ĥ (IP + SV)** | **1.84 / 3.02 / 3.44 / 3.51** | **13.9 / 7.2** |
| tauspin ĥ without IP/SV | 1.24 / 2.04 / 2.33 / 2.38 | 20.0 / 10.4 |
| LHC CP observables (φ*_CP ZMF + lab-frame + textbook) | 1.36 / 2.24 / 2.56 / 2.61 | 19.0 / 9.9 |
| textbook observables | 0.83 / 1.38 / 1.58 / 1.61 | 37.2 / 19.2 |

- Doubled luminosity (6 ab⁻¹, one fit): entanglement tauspin 2.6σ / 4.2σ / 4.8σ (sidebands / τ=1 / τ=10) vs φ*_CP 1.9σ / 3.2σ / 3.6σ. σ(φ_τ): tauspin 9.9° vs φ*_CP 13.5°.
- tauspin / best standard observables:
  - ratio 1.35–1.38 in every scenario (entanglement significance and φ_τ precision alike), i.e. the same sensitivity with about 1.9× less luminosity;
  - without IP/SV, tauspin is equal to or worse than the φ*_CP baseline, so the gain comes from the learned IP/SV local-frame representation;
  - the ratio is stable across binning (4×4: 1.68 vs 1.24; 8×8: 1.93 vs 1.42), response priors, Werner vs separable null, the high-S/B-only categories (1.21 vs 0.87) and every category individually (1.3–1.5).
- Fisher correlations with the exact statistics (g_T, g_A): tauspin 0.344 / 0.348; φ*_CP baseline 0.233 / 0.240; textbook 0.134 / 0.119.

## Proposed central claims (for review)

1. With polarimetry at the level of the tauspin regressor (PV-referenced IP and SV in the local τ frame, calibrated to ATLAS full simulation), the expected sensitivity to the H → ττ spin state improves by a factor 1.35–1.4 over the best combination of the observables used in the LHC CP analyses. This is equivalent to about 1.9× the integrated luminosity, and it holds for both the CP-mixing angle and the entanglement witness, under a data-driven background-shape treatment.
2. A QM-conditional exclusion of all separable Bell-diagonal spin states in H → τhτh at the HL-LHC is not achievable with sideband-only background knowledge (1.8σ per experiment). It needs both of the following:
   - background spin-shape control from an auxiliary sample at least as large as the signal region (3.0σ at τ = 1, about 4.2σ with two experiments' worth of luminosity);
   - an independently calibrated transverse analysing power.

   With standard observables, the same conditions give 2.2σ and about 3.2σ.
3. The precision on the CP-mixing angle φ_τ is robust against the analysing-power calibration.

## Not claimed

- Any absolute discovery-level statement. All numbers are Asimov, from cut-based ATLAS Run-2 categories transferred to 14 TeV, τhτh only, and from our own reco-level smearing simulation, which reproduces the ATLAS IP/SV resolutions.
- An ATLAS+CMS combination.
- Entanglement in a local-realist sense (arXiv:2507.15949, 2507.15947).

## Code and artefacts

- `ent_robust.py`, `ent_cp.py`, `phicp_zmf.py`, `ent_add_zmf.py`, `ent_mcstat.py`, `make_figures_robust.py`
- Results: `results/json/ent_robust*.json`, `ent_cp*.json`, `robust_summary.json`, `ent_mcstat.json`
- Figure: `results/figures/ent_cp_robust.png`

## Addendum after the re-review (`review_skeptical_entanglement_v2.md`, verdict: revise)

Probes run:

1. **Information vs regression machinery (M1).** The tauspin network was retrained without SV, the longitudinal IP component and MET (`train_h.py --drop sv_,ip_k,met_`; validation loss 0.01442 vs 0.01436 nominal and 0.01575 without IP/SV) and passed through the same second stage. It is indistinguishable from full tauspin:
   - correlations 0.343 / 0.347 vs 0.344 / 0.348;
   - entanglement at τ = 1: 2.96σ vs 3.02σ (sidebands only 1.81σ vs 1.84σ);
   - σ(φ_τ): 14.1° vs 13.9°.

   The gain is therefore carried by the **transverse impact-parameter information of all decay modes, used inside a learned per-τ polarimeter**. SV, longitudinal IP and MET do not matter. A gradient-boosted regressor on the same raw inputs does much worse (correlation 0.11, reviewer probe). The method claim is therefore "the learned polarimeter" (information plus architecture), not a pure information claim. The LHC CP analyses use the transverse IP only for 1p0n.
2. **Separate CP-odd response (M5).** The sin 2φ term gets its own response k_A:
   - 10 % prior: σ(φ_τ) 13.9° → 14.0° (φ*_CP 19.0° → 19.1°);
   - k_A free: the local width is lost (about 45°).

   The precision on φ_τ therefore needs the antisymmetric analysing power calibrated to about 10 %, but is insensitive to the common transverse scale. Only the local 68 % width is robust to k_T. The 95 % interval widens (29° → 42°) and the CP-odd exclusion needs k_T.
3. **Auxiliary background control τ = 1, CP angle:** tauspin 8.5° vs φ*_CP 11.5°.

Issues acknowledged and not resolved here:

- φ*_CP baseline: no ATLAS 3p0n decay-plane or CMS a1 polarimetric treatment for 3-prong decays.
- The Run-2 calibration of the absolute scale is only indicative: q(90°) < 1 at 139 fb⁻¹, so there is no 68 % interval and the ±95° figure is not one.
- Kinematic-drift stress test not run for the φ*_CP baseline (no hold-out features).
- Auxiliary samples measure the true shapes exactly (no transfer uncertainty).
- No identified in-situ source for the transverse analysing power.
- No IP/SV resolution-scaling test.
- Known-shape fits for `exact` and `textbook_pair` (CP) returned NaN.

Figure 1(b) now shows the actual profile q(φ) points.
