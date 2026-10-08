# Validity review (Claude experimental-physicist subagent, read-only), 2026-10-08

Reviewed: review_packet.md, all modules, summary.md (v1), study/lifetime/latest JSONs, 7 PNGs; light
read-only probes on ICEPP (MINUIT refits, uniform-split null, spin-sign check, scenario reruns).

**Verdict: conditionally acceptable.** C1 supported once reworded; C2 and C3 not supported as worded.

## Findings
- **M1 (major)** Lifetime nominal may be inconsistent with the lepton selection already applied:
  no-cut |d0|/σ templates on post-selection yields; τ→ℓ has ~37% >3σ, ~23% >5σ (prompt 1.1% >3σ).
  Latest ATLAS explicitly applies no d0 selection (atlas_latest.txt:384–386); legacy definition
  silent. Tight electron LH uses d0 and |d0/σ| (1902.04655 Table 1) → e-channel gain overestimated.
  With TTVA the consistent estimate is R≈1.031 (comb 4.34). Asked: literature check, LH-sculpted
  electron variant, quote C2 as a range.
- **M2 (major)** τ→ℓ shares in signal-like τℓτh bins assumed: top 0.08 from 30 MC events; tt̄-fake
  lepton τ share set to 0 although ≈8–11% expected. Probe: fake τ share 8% → R 1.0606→1.0523.
- **M3 (major)** Systematic surrogate under-degrades τℓτh (0.865 vs ATLAS 0.783); raising top/fake
  norm syst does not help (absorbed); add a lep-had shape-type systematic and rerun.
- **M4 (major for C1 explanation)** Process→spin mapping partly scenario (fakes U/H/W; legacy top 100% W
  while 35–37% of latest hadhad tt̄ are lepton fakes — impact small; single H→H conservative).
- **m5** L-BFGS-B finite-difference precision: per-channel R can be off by ~0.5–1% (impossible entry
  lephad tauspin+lt 1.1688 < lt-only 1.1714); combined R agrees with MINUIT within 0.001.
- **m6** Uniform 6-way split reproduces baseline exactly → no spurious gain from sub-binning; template-MC
  noise bias < 10% of spin gain; per-channel R (e.g. LTT alone +6%) not physical, don't quote.
- **m7** c=1 instead of 0.88 gives 4.44 instead of 4.42 (quote ±0.02); "+8% equivalent luminosity"
  assumes Z∝√L — rephrase.
- **m8** d0 resolution 10⊕150/pT µm consistent with Run 2; ITk up to ×2 better at high pT (2412.15090);
  result fairly robust to prompt tail.
- **m9** spin⊗d0 ignores τ_lep–τ_had spin correlation (lost information, not double counting); d0 not an
  input of the latest ATLAS MVA; Fig 2 shows 10 sub-bins vs nominal 6.

## Verified correct
Spin sign conventions (π-mode corr(h_k,x)=+0.50/+0.52; W weight ⟨x⟩ 0.566 vs 0.629; H/Z kk signs
opposite; C_T=0.5 angular average); Asimov q0 construction; LTT HEPData fix; figure extraction closure.

## Approval conditions
1. Resolve M1 by literature or quote C2 as a range. 2. tt̄-fake τ→ℓ ≈ top share, justify top share.
3. Recalibrate lep-had systematics to ~0.78 and rerun. 4. Regenerate per-channel numbers with MINUIT.
