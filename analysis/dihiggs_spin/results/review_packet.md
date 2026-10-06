# Review packet: tau-spin background rejection for HL-LHC HH -> bb tau tau

All paths are relative to `~/Projects/tauspin/analysis/dihiggs_spin/` unless absolute. Read-only review: do not modify any file.

## Question asked by the user

The ATLAS HL-LHC projection of SM di-Higgs is 4.3 sigma (ATLAS alone, 3 ab^-1, scenario S2; ATL-PHYS-PUB-2025-018 / arXiv:2504.00672 Table 2; bb tautau contributes 3.5 sigma = tau_had tau_had 3.1 (+) tau_lep tau_had 1.8, ATL-PHYS-PUB-2024-016 Table 4). How much can adding tau-spin information, at the reconstruction quality reached by the user's "tauspin" project (H vs Z -> tau tau separation at equal mass with a learned polarimeter regressor; ATLAS full sim: fixed-readout AUC 0.626 with PV-referenced IP + SV, exact-polarimeter ceiling ~0.72-0.73), raise this significance? The answer should be at a level that could support a paper.

## Target claims (to be reviewed)

C1. With tauspin-level reconstruction, spin binning added to the most signal-like tau_had tau_had bin raises its significance by about +1.7 % (realistic decay-mode mix, phase 2) to +3.8 % (pi/rho/3pi-only cohort, phase 1); the transfer gives ATLAS bb tautau 3.5 -> 3.56-3.61 sigma, ATLAS combined 4.3 -> 4.35-4.39 sigma, ATLAS+CMS 7.2 -> 7.26-7.31 sigma.
C2. Even the exact (truth) polarimeter gives only +9.6 to +12.6 % on that bin (ATLAS combined 4.53-4.59 sigma), because about 28 % of that bin is single Higgs (spin-identical to the signal) and 44 % is Z+HF, against which spin separation is intrinsically weak; spin helps most against true-tau top (W -> tau nu, fully polarised).
C3. The parametric HL-LHC detector simulation plus a re-trained regressor reproduces the tauspin ATLAS full-simulation performance (closure).
C4. A direct "kinematic BDT vs kinematic+spin BDT" comparison on the simulated backgrounds is not usable: background MC statistics in the signal-like region are ~0-2 events.
Do NOT accept claims about kappa_lambda beyond "scales like the combined significance" (crude approximation).

## Method

Phase 1 (`phase1_reweight.py`, `phase1_significance.py`, `phase1_lephad.py`): the user's ATLAS full-sim validation cohort (59,390 events; H(91 GeV)+j and Z+j, tau decay modes pi, rho, 3pi only), exact h and regressed h_pred per event. TauSpinner-style reweighting with the pooled generator density (rho_H + rho_Zgen)/2 to hypotheses H, Z (C_kk=+1, P_tau=-0.147), W pair ((1-h-_k)(1-h+_k), sign checked on data), U (no spin; fake proxy), WU. Multiclass GBDT on h_pred + reco modes, 2-fold. Composition: arXiv:2209.10910 Table 5 most signal-like tau_had tau_had bin (139 fb^-1), scaled x3000/139 and by 14 TeV factors (S=40.8, B=155). Significance: Asimov (stat) and profiled Fisher with normalisation nuisances (Z+HF 10 %, top 10 %, fakes 20 %, single H 15 %), 20 quantile bins of D = p_H / sum f_b p_b. Results: `outputs/phase1/*.json`, figure `outputs/figures/phase1_spin_templates_and_gain.png`.

Phase 2 MC: MG5 3.5.9 LO 14 TeV + Pythia 8.315 (gen/procs.sh, gen/pythia_shower.py): gg->hh [noborn=QCD] 400k, ttbar->b tau nu b tau nu 2M, ttbar->tau nu + jj 1M (fakes), Z(->tau tau)bb 1.5M, ZH(bb) 0.3M, ttH 0.3M. TauDecays:mode = 4 (mode 1 decorrelated H->tau tau from MG LHE; checked on samples, see 10_Log). Exact h in `polarimeter_pythia.py` (Pythia's CLEO 3pi current ported). Detector response `reco.py` calibrated on the user's ATLAS cohort (pi0 energy/angle resolution vs energy, track pT 4 %, mode migration, IP, SV, MET). tau_had tau_had selection (pT>40/30, 2 b-tags). MMC ported from the user's code (`mmc.py`, `build_dataset.py`), with per-tau azimuth range and m_vis cap (98 % success, +-15 % resolution).
Regressor (`train_h.py`): MLP on per-tau substructure, PV-referenced IP and SV in the local basis, MET projections, hybrid h; targets h-, h+ and h- (x) h+; trained on a dedicated equal-mass H(125)+j / Z(125)+j Pythia sample (527k selected; 20 % held out).
Closure (`closure_hz.py`, `outputs/v1/closure_hz.json`, figure `outputs/figures/phase2_closure_h_regression.png`).
Direct classifier attempt (`classify.py`, `outputs/v1/classify_s*.json`): failed due to MC statistics (C4).
Factorised attempts: `factorized.py` (spin-only classifier on the real background MC; `outputs/v1/factorized.json`) showed the regressed-h discriminant on signal shifts with the kinematic score (kinematic leakage), so it was not used; `factorized_rw.py` (reweighting HH events themselves) had too small effective sample sizes (ESS of W hypothesis 43-339).
Adopted: `matched_rw.py` (`outputs/v1/matched_rw.json`): hold-out H/Z(125) events, classifier density-ratio matched to the tau kinematics (pT, eta of both taus, m_vis, MET, dR, reco modes) of HH events above a cut on the kinematic BDT score (signal efficiency 100/50/20 %), reweighted to H/Z/W/U with the pooled density (rho_Zgen measured on the Z hold-out), spin classifier on h_pred, h-h+ products, reco modes. `diag_modes.py` (mode-mix explanation), `scan_composition.py` (pure-background gains), `final_projection.py` (transfer; figure `outputs/figures/final_spin_gain_projection.png`, table `outputs/v1/final_projection.json`).
Transfer (`projection.py`): channels and experiments in quadrature; ATLAS+CMS uses the ATLAS bb tautau projection for both experiments as in the published combination.

## Key numbers

Phase 1 (ATLAS full-sim cohort), R with systematics: reco no IP/SV 1.033, reco 3 IP+SV 1.038, ideal IP 1.049, exact 1.126; tau_lep tau_had SLT/LTT reco 1.010/1.025, exact 1.017/1.042.
Closure (hold-out H/Z(125)): corr(h,hhat) n/r/k 0.61/0.62/0.73 (ATLAS 0.62/0.58/0.765); H/Z readout AUC 0.627 (ATLAS 0.626); exact LR 0.730 (ATLAS 0.719-0.729).
Phase 2 matched (sig. eff 20 %): exact 1.096, tauspin 1.017, textbook observables 1.007; ESS H/Z/W/U = 12.3k/11.4k/7.8k/12.5k. Restricting to pi/rho/3pi on both sides (65 % of events): tauspin 1.032, exact 1.123.
Pure-background stat-only gains (tauspin / exact): Z+HF 3.1/19.4 %, top 12.6/41.7 %, fakes 0.7/8.3 %, ATLAS mix 1.5/8.5 %.

## Known open points

- Fakes are modelled as "no spin information" (U); real fake taus in ATLAS are data driven.
- tau_lep tau_had only from phase 1 (single-side polarisation; no decay-mode dilution applied -> upper estimate).
- The ATLAS analysis has many BDT bins and 3 categories; only the most signal-like tau_had tau_had bin composition is public (Table 5); its gain is applied to the whole channel.
- Z production polarisation (transverse correlations) not included in the Z hypothesis.
- kappa_lambda interval only scaled with the combined significance.
- No systematic uncertainty on the spin modelling itself (tau decay model, pi0 scale).

---

# Revision after the first review round (2026-10-07)

The previous primary method (matched_rw.py) is superseded. New primary method, addressing the validity (revise) and skeptical (approve-with-conditions) findings:

- Spin-flat HH sample: the same MadGraph HH LHE files (2.0M events, 6 runs) showered with Pythia `TauDecays:mode = 3`, `tauPolarization = 0` (unpolarised, uncorrelated taus; checked C ~ 0, <h> ~ 0 on 8k events). Generator spin density = 1, so the weight to hypothesis X is rho_X(h_exact) itself (bounded; no density model, no kinematic matching). 58,805 selected tau_had tau_had events (`gen/pythia_shower.py --spinflat`, `run_v2.sh`, `outputs/v2/`).
- Same regressor (trained once on the equal-mass H/Z(125) sample) and the same saved kinematic BDT (`k_model.py`, retrained identically to classify.py set K) applied to this sample.
- `spin_gain_U.py`: hypotheses H, Z_long (C_kk=1, P=-0.147), Z_full (+C_nn=+0.49, C_rr=-0.46), W pair, U. Regions above the K score cut at 100/50/20/5 % H-weighted signal efficiency. 2-fold multiclass classifiers on spin (h_pred, products, modes), kin (all K inputs), kin+spin, obs; exact analytic LR. Statistics: Asimov discovery q0 (`q0_asimov`) with profiled Gaussian normalisation nuisances (Z+HF 10, top 10, fakes 20, single H 15 %) and one spin-template contrast shape nuisance (25 %), used for all numbers now. Event bootstrap (100) for the spin set.
- `closure_U.py`: spin-flat reweighted to H vs the real (spin-correlated) HH sample from the same LHE: D template chi2/ndf 24.7/20; h_pred components chi2 14-37 for 23-24 bins; exact-h moments C diag flat->H (1.046, 1.053, -0.905) vs real (0.945, 0.983, -0.905) (stat ~0.035 on real). Variation with H transverse correlations x0.92: spin R unchanged (1.0231), exact 1.124 -> 1.117 (`var_htrans.py`).
- Results (Z_long, K region 20 %): exact R 1.124; spin R 1.0228 (+-0.0009 boot), with shape nuisance 1.0224; kin only 1.0002; kin+spin over kin (increment) 1.0213; obs 1.0068. Z_full: spin 1.0229, exact 1.128. Region dependence: spin 1.023/1.024/1.023/1.017 for 100/50/20/5 %.
- Transfer (`final_v2.py`): R_ch = sqrt(f R_bin^2 + (1-f) R_rest^2), f in [0.5, 1] (0.62 = stat-only share of the Run-2 top bin: 3.15^2/4.0^2), R_rest between 1 and the gain of a top-rich composition (composition scan up to 60 % top: spin 1.039, exact 1.132). tau_lep tau_had excluded from the headline. Relative gains at f = 0.62: tauspin bb tautau +1.1...+2.0 %, ATLAS +0.7...+1.4 %, ATLAS+CMS +0.5...+1.0 %; exact (implemented modes) bb tautau +6.0...+9.4 %, ATLAS +4.0...+6.3 %, ATLAS+CMS +2.9...+4.6 %.
- Figures: `outputs/figures/v2_spin_templates_closure_region.png`, `outputs/figures/v2_final_gain_projection.png`.
- Wording now used: "exact h for the implemented modes (pi, rho, pi 2pi0, 3pi; 84 % of hadronic decays per tau, 71 % of events with both)", not "ceiling"; absolute significances only as illustration.
