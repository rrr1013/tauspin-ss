# Skeptical re-review of packet v2 (Claude skeptical-reviewer subagent), 2026-10-07

**Verdict: revise.** The results are internally consistent. But claim 1 mixes up two things, the information content of the inputs and the regression machinery. Claim 2 is quoted under ideal transfer conditions. Claim 3 holds only by construction and only locally. I modified no files. The one probe I ran used scratch scripts only.

**Probe I ran** (local arrays, `regress()` from ent_robust.py, same folds and weights). Correlation with the exact g_T / g_L targets:
- `phicp` as-is: 0.192 / 0.401 (reproduces the JSON's 0.195 / 0.403)
- `phicp` plus raw IP/SV (n, r, k, L): 0.204 / 0.401
- HGB on **the same 73 raw inputs the tauspin NN uses**: **0.109** / 0.482
- tauspin (NN outputs, then HGB): 0.345 / 0.549

## Major

**M1. Regressor capacity is a demonstrated confound for claim 1.** The second-stage HGB cannot extract transverse cross-τ correlations from raw inputs: 0.109, which is below φ*_CP's 0.19–0.23. tauspin gets a dedicated NN trained on separate trainH/trainZ samples, with the nine h⁻h⁺ products as direct targets. The baseline gets only HGB on hand-crafted features. So the 1.35× compares two pipelines, not two information sets. The only same-architecture ablation is `tauspin_noIPSV`, and it is not a fair stand-in for the LHC observables: it removes IP even for 1p0n, which φ*_CP uses.

The attribution "gain comes from the learned IP/SV local-frame representation" is not isolated either. It could equally come from using IP for ρ and 3-prong decays, from SV, or from MET projections. Another inconsistency: noIPSV has *higher* correlations than phicp_zmf (0.247 / 0.544 vs 0.233 / 0.400) yet *lower* significance (1.24σ vs 1.36σ). So the quoted "Fisher correlations" do not support the ratio.

**M2. "Ratio 1.35–1.38 in every scenario" is overstated.**
- In the most damaging stress test, drift_kin_realZ (genuine-τ background drifting to real-Z kinematics), tauspin falls from 1.84σ to 0.55σ. Textbook rises to 1.34σ, of which 0.84σ is fake. phicp_zmf and phicp were never run there because there is no holdout Gz. So the scenario where the ordering is at risk is missing from the ratio table.
- drift_spin gives 2.06 / 1.625 = 1.27, not 1.35.
- The binning, prior, Werner, category and luminosity variants all reuse the same G arrays. They repeat one number; they are not independent checks.
- The smearing model is shared, but the comparison depends on it unequally. tauspin's gain sits entirely in IP/SV, while the baseline uses IP only for 1p0n and the 3-prong leading track. The ratio is therefore not protected against an optimistic IP/SV resolution or missing tails.

**M3. The φ*_CP baseline is weaker than the LHC analyses.**
- 3-prong decays use the leading-track IP. ATLAS (3p0n decay plane) and CMS (a1 methods) use stronger treatments.
- The y-shift is implemented correctly: the sign product runs over ρ sides only, and CP-even and CP-odd modulations have opposite signs. But the mean modulation is tiny: ⟨cos φ*⟩ = ±0.035.
- At 139 fb⁻¹ the baseline gives Z_CPodd = 0.60, against ATLAS's expected ~2.1σ in arXiv:2212.05833, all channels (my recollection; not re-checked).
- The "±95°" comes from a local quadratic beyond the physical range: q(90°) = 0.36 < 1, so there is no 68 % interval at all. The ×4.98 known-shape extrapolation to ±49° is ad hoc, so the "conservative by factor 2–3" calibration is not established. If the gap is specific to the baseline, the ratio is inflated.

**M4. Claim 2 holds only under ideal conditions.**
- The τ = 1 auxiliary sample measures exactly the true signal-region shapes (Asimov aux = Zsh / Ush). There is no embedding or fake-factor transfer uncertainty.
- Z, top and "other" share a single shape.
- The signal shape is the same in every mass bin.
- The fake shape is the H-flat kinematic shape, and data and fit model are the same.
- The k_T calibration needs a source that is not identified. Z→ττ has negligible transverse correlations, and calibrating on H itself would be circular. With k_T free the significance is 0.15σ. So 3.0σ is an upper bound, conditional on both QM and simulation.

**M5. Claim 3 is true by construction and only locally.**
- A single k_T multiplies both the cos 2φ and sin 2φ templates, so φ is a phase by assumption. A separate response for the antisymmetric statistic (anisotropic n/r smearing, separate regressors) is untested. If both responses float freely, φ is unconstrained.
- Even with one shared k_T free: q(45°) drops from 8.42 to 4.17 and q(90°) to 0. The 95 % half-width grows from about 29° to about 42° (interpolated from the q grid). Only the local 68 % width (13.9° → 14.7°) is robust.

## Minor

- Fig. 1(b) plots idealised parabolas (φ/σ)² out to 45°, labelled −2ΔlnL. The real q is lower: tauspin 8.4 vs 10.5 drawn at 45°, phicp_zmf 4.6 vs 5.6. Plot the actual points or label the curves as the quadratic approximation.
- The known-shape CP fits for `exact` and `textbook_pair` returned NaN. This is unreported fit instability at τ = 100.
- The separable null is fine for the (g_T, g_L) binning, which is blind to off-diagonal C. But polarisation terms B_k± can correlate with g_L and are absent from both hypotheses; add them as nuisances.
- The entanglement Asimov fixes φ = 0. In general C_nn + C_rr = 2 cos 2φ.
- The small-angle σ(φ) extraction (grid ≤ 20°) is valid at 3 ab⁻¹: tauspin q(20)/q(10) = 3.9. It is invalid at Run 2.

## Strongest supportable wording

1. "In our simulation, a statistic built on the tauspin NN polarimeter (PV-referenced IP, SV, MET, local frame) gives about 1.35× the expected Asimov sensitivity of a gradient-boosted regression on φ*_CP (ZMF), lab-frame and textbook features. This holds for both the entanglement witness and φ_τ, under mass-independent background shapes. How much of the gain comes from information content versus regression architecture is not separated."
2. "With sidebands only and ideal mass-independent shapes, we expect at most 1.8σ. An auxiliary sample of signal-region size with perfect shape transfer, plus a 10 % transverse analysing-power calibration (source unidentified), gives at most 3.0σ (φ*_CP: 2.2σ). Under realistic kinematic drift of the background, the tauspin sensitivity falls to 0.55σ."
3. "Assuming a common response for the symmetric and antisymmetric transverse correlations, the local 68 % precision on φ_τ is insensitive to the overall transverse scale. The 95 % interval and CP-odd exclusion are not."

## Probes that could change the conclusion

1. Retrain `dihiggs_spin/train_h.py` on inputs carrying only LHC-level CP information (no SV, no ρ/3p IP, no ip_k), then run the same second stage. If the ratio drops below ~1.15, claim 1 falls.
2. Compute holdout Gz for phicp_zmf and run drift_kin_realZ, with signal and with the null injected.
3. At τ = 1, take the aux shape from Xr (real-Z kinematics) while the signal region keeps Zsh.
4. In ent_cp.py, give the sin 2φ term its own response k_A (prior 10 % and free).
5. Scale IP/SV resolution by 1.3 and add non-Gaussian tails, for all levels.

Files: `~/Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/{ent_robust.py,ent_cp.py,phicp_zmf.py,ent_add_zmf.py,make_figures_robust.py,results/json/robust_summary.json,results/json/ent_cp*.json,results/figures/ent_cp_robust.png}` and `~/Projects/tauspin-wt-dihiggs/analysis/dihiggs_spin/train_h.py`.