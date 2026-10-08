# Review packet: HL-LHC HH→bbττ sensitivity with TauSpin information (2026-10-08)

Code: `analysis/hh_hllhc/` (this branch, commit 2a00e20 + later). Outputs (JSON, figures):
`lxgpu02.icepp.jp:/home/rbaba/dihiggs-hllhc-20261008/outputs/` (reachable with `ssh icepp-gpu`; read-only use please).
Figure PNG copies: `/private/tmp/claude-501/-Users-ryunosuke-Library-Mobile-Documents-com-apple-CloudDocs-Athena/6b82c1ca-f036-4baf-a79a-605e3b1875c1/scratchpad/figs/`.
All-variant table: `outputs/study/summary.md` on ICEPP.

## Question

How much would the ATLAS HL-LHC SM HH expected significance (ATL-PHYS-PUB-2025-006: combination 4.26σ, bbττ 3.54σ, baseline systematics, 3 ab⁻¹, 14 TeV) increase if the bbττ analysis added TauSpin information: (a) the regressed τ polarimeter ĥ (both τ_had in τhτh; the τ_had in τℓτh), (b) the PV-referenced transverse impact-parameter significance |d0|/σ of the light lepton in τℓτh (τ→ℓ vs prompt W→ℓ)?

## Method

1. **Baseline composition** — public per-bin, per-process yields of the final discriminant:
   - Run-2 legacy bbττ (arXiv:2209.10910) HEPData ins2155171 Fig 8a/b/c (hadhad 14, SLT 15, LTT 15 bins, post-fit). HEPData defect found: LTT "Z+HF" column duplicates the SLT column; re-derived as Total − others; last LTT bin from Table 5 (Z+HF 1.7). Signal normalised to Table 10 SM ggF+VBF yields.
   - Latest Run2+3 (arXiv:2607.26879): bin yields recovered from the vector PDFs of Fig 7/8 (12 SRs, stacked polygons + log-axis ticks). Closure vs auxiliary tables (region totals): major processes 0.1–0.5%, tt̄+tW+other 0.6%, ggF HH×20 overlay vs SM signal 0.2%. In τhτh the tt̄ stack contains lepton→τ fakes; split with aux-table fractions.
   - The official HL-LHC projection (PUB-2024-016 → PUB-2025-006) is built on arXiv:2404.12660 (no bin-level yields public). We therefore compute gain ratios R = Z(with info)/Z(baseline) in our likelihood and multiply ATLAS's numbers by R.
2. **HL-LHC scaling** (PUB-2024-016 §3): L′/L, xs 14/13 TeV (HH 1.18, single H 1.15, others 1.18; Run-3 composition 13.6→14 TeV ≈1.066). HEPData yields are post-fit, so the Z+HF ×1.3 is already included.
3. **Likelihood** (pyhf model, own minimiser): binned Poisson, Asimov μ=1; q0 = 2ΔNLL with the unconditional NLL at the generating point; μ=0 fit by L-BFGS-B from two starts (q0 differences ≈0.2%). Free top and Z+HF normalisations shared by channels. "Baseline" systematics: correlated single-H norm 18% (scanned to reproduce PUB-2024-016 Table 4 stat→baseline degradation: ours 0.74/0.80/0.87 vs ATLAS 0.76/0.775/0.78 for comb/hadhad/lephad), Z+HF 10%, fakes 10%, other 15%, top 5%.
4. **Spin** — spin-flat HH sample (58,805 events, 14 TeV MG5+Pythia8, parametric detector calibrated on ATLAS full sim; TauSpin ĥ regression reproduces ATLAS-sim H/Z fixed-readout AUC 0.627 vs 0.626) reweighted by physical spin densities: H (1+nn+rr−kk), Z (P=−0.147, C_kk=1, C_T=±0.5), W pair (1−k)(1−k), one W τ + fake, unpolarised. Multiclass XGBoost, 3-fold cross-fitting, early stopping on a held-out training fold, 3 seeds. Arms: textbook (Υ, x, m_vis, hybrid h per side), tauspin (ĥ6 + joint moments 9 + modes), exact (analytic likelihood of exact h). Each baseline bin b is split into n_sub=6 bins of D_b = log p_H − log Σ_p f_{p,b} p_{X(p)} (composition-aware, background fractions from the public yields; edges = signal quantiles on fold 0; templates filled on folds 1+2). τhτh templates conditioned on the kinematic-score window holding the same signal fraction as the ATLAS bin (and on the m_HH category for latest Hi/Lo). Process → spin: signal, single H → H; Z+HF → Z; top → W; MJ fakes, other → U; tt̄ fakes and lepton fakes → WU. τℓτh uses the τ_had side only (H/Z/W single-τ densities).
5. **Lepton lifetime** — 10/08 lep-had MC (584k raw, 247,608 selected; perigee-corrected d0; σ_d0 = 10⊕150/pT µm ×1.2 for e, 2% 4σ tails; 'real' variant degrades). |d0|/σ bins fixed [0,1,2,3,5,∞). τ→ℓ template = signal leptons in the kinematic window of each ATLAS bin; prompt and nonprompt (synthetic HF-like, mean 100 µm) pooled. Origin fractions: signal, Z+HF τ→ℓ; single H 0.92; top 0.08 (MC 0.05–0.09 per bin); fakes prompt; other prompt. Joint template = spin ⊗ d0 (independence given process assumed).
6. **Transfer** — Z_bbττ' = 3.54·R; Z_comb' = sqrt(4.26² + c(Z_bbττ'² − 3.54²)), c = 4.26²/ΣZ_i² = 0.880 (in the no-syst table the quadrature sum reproduces 5.98 exactly).

## Results (bbττ R; ATLAS bbττ; ATLAS combination)

| information | Run-2 legacy composition | latest Run-3 composition |
|---|---|---|
| textbook spin | 1.009; 3.57; 4.28 | 1.005; 3.56; 4.27 |
| TauSpin ĥ (spin only) | 1.015; 3.59; 4.30 | 1.012; 3.58; 4.29 |
| lepton lifetime only | 1.041; 3.69; 4.37 | 1.048; 3.71; 4.39 |
| TauSpin ĥ + lifetime | 1.061; 3.76; 4.42 | 1.063; 3.76; 4.42 |
| exact h + lifetime (ceiling) | 1.120; 3.96; 4.58 | 1.117; 3.95; 4.57 |

Robustness of "TauSpin ĥ + lifetime" (legacy): stat-only 1.063; degraded d0 resolution 1.043; TTVA cut 1.034; top τ→ℓ 25% 1.045; 30% nonprompt fakes 1.036; all simultaneously + 25% spin-contrast uncertainty 1.020 (comb 4.31). Spin-only variants: 1.010–1.020. Reference: ATLAS's own scenario "τ_had ID efficiency +5%" gives 4.34, "b-tag +5%" 4.44, both 4.52 (PUB-2025-006 Table 9).

## Draft claims

C1. With the official ATLAS HL-LHC projection as reference, TauSpin's reconstruction-level spin information alone raises the HH combination from 4.26σ to ≈4.29–4.30σ (+1%); the effect is small because the most signal-like τhτh bins are dominated by Z+HF, single H and fakes, for which reco-level spin separation is weak (H/Z AUC 0.578) or absent.
C2. Adding TauSpin's PV-referenced impact-parameter information for the light lepton in τℓτh gives the larger part: combined ≈4.42σ (+3.8%, ≈ +8% equivalent luminosity), stable between the 2022 legacy and the 2026 latest compositions; plausible degradations reduce it to 4.31–4.38σ.
C3. Perfect polarimetry (exact h) with lifetime would reach ≈4.57σ — the ceiling of these information channels in this analysis structure.

## Known limitations

No κλ interval (no κλ-dependent signal templates). Spin/lifetime templates from HH-like kinematics applied to all processes within a bin window. Fake-τ spin response and nonprompt lepton fraction are scenarios. Systematic model is a calibrated surrogate, not the ATLAS workspace. Lo-m_HH categories of the latest analysis have clipped signal overlays (tiny Z, not interpreted).
