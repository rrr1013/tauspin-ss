# Review packet: observing spin entanglement in H -> tau_had tau_had at the HL-LHC with tauspin-level polarimetry

Paths relative to `~/Projects/tauspin-wt-dihiggs/analysis/` (git worktree of branch `ariadne/dihiggs-spin-20261006`). Inputs not in git are under `ip_tau_reco/outputs/ent/` and `dihiggs_spin/outputs/` (symlink). Read-only review.

## Context
The user's tauspin project reconstructs the tau polarimeter vector h in H vs Z -> tau tau with a learned point-h regressor using PV-referenced impact parameters (IP) and SV in a local frame (ATLAS full simulation: fixed-readout H/Z AUC 0.626, exact-h ceiling ~0.72-0.73). A previous run found that tau spin adds only ~1-2 % to HL-LHC di-Higgs sensitivity. This run explored other uses. Two results:

1. IP/SV do not improve m_tautau (go/no-go probe, `ip_tau_reco/probe.py`, `ipmmc.py`, `ipmmc2.py`, result `ip_tau_reco/results/json/ip_mass_probe_v2.json`): HH 68 % half-width 12.1 % (physical-prior likelihood, no IP) vs 12.4 % (with IP likelihood); IP measures the tau azimuth, not the energy fraction.
2. Main claim (below).

## Target claim
With tauspin-level reconstruction, the HL-LHC can establish spin entanglement of the tau pair in H -> tau_had tau_had: expected significance per experiment at 3 ab^-1 ~4.1 sigma (3.8 / 3.6 sigma with a 10 / 20 % response-scale uncertainty), ~5.2 sigma for ATLAS+CMS-equivalent 6 ab^-1 (10 % response uncertainty), while textbook observables (Upsilon, x, hybrid h from reco visibles + MMC neutrinos) give 2.7 (2.4) sigma per experiment and 3.3 sigma combined. IP/SV in the regressor are worth +15 % in significance (3.6 sigma without). The European Strategy QI input (arXiv:2504.00086, sec. on limitations) states that neutrinos "preclude or significantly complicate" such studies in H -> tau tau -> pi nu pi nu. Prior LHC tau-pair entanglement work (arXiv:2504.01496) treats Drell-Yan, not H -> tau tau.

## Method
- Spin-flat H(125)+jet sample: Pythia 8.315, gg/qg/qqbar -> H + jet, pTHat > 40 GeV, H -> tau tau with TauDecays:mode = 3, tauPolarization = 0 (unpolarised, uncorrelated taus; generator spin density 1), 1.5 M events, 158,584 selected tau_had tau_had (reco.py --no-btag: tau pT > 40/30 GeV, |eta| < 2.5, OS). Detector response as in the HH study (calibrated on the ATLAS tauspin validation cohort). Regressor = the one trained on equal-mass H/Z(125)+jet (closure vs ATLAS: H/Z AUC 0.627 vs 0.626), applied unchanged (`dihiggs_spin/train_h.py --load`); a variant regressor without IP/SV inputs (`--drop ip_,sv_`).
- Spin model: Werner family rho_p = 1 + p (h-_n h+_n + h-_r h+_r - h-_k h+_k); p = 1 is the CP-even Higgs (Bell state), p <= 1/3 separable; testing p > 1/3 is the witness C_nn + C_rr - C_kk > 1. Background spin state: Z -> tau tau (C_kk = +1, P_tau = -0.147; variants with transverse terms, fakes unpolarised, real Z(125)+j template). Events without a polarimeter on both sides keep weight 1 (one-side marginal for Z).
- Statistic: cross-fitted regressions g_s = E_flat[s|x], g_z = E_flat[rho_Z - 1|x] on (h_pred, h-h+ products, reco modes) — g_s is sufficient for p because rho_p is linear in p. 2D quantile templates (6x6 or 8x8).
- Test: Asimov binned likelihood, H0: p = 1/3 vs data p = 1; signal yield free, background normalisation Gaussian-constrained 2 % (5 % variant), optional response-scale nuisance scaling the spin-dependent part of all templates (10/20/30 %).
- Yields: ATLAS H -> tau tau 139 fb^-1 (arXiv:2201.08269 Tables 8, 11) tau_had tau_had 12 categories, x 3000/139, 14 TeV factors 1.13 (signal) / 1.15 (bkg), split into 9 m_tautau bins with acceptances from our MC (H from hh/zh/tth, Z from Z(tautau)bb, fakes from ttbar tau+jets, top from ttbar). Per-category templates restricted to the category's pT(tautau) range (information per event falls with boost: 0.078 at 60-100 GeV, 0.029 above 300 GeV). Categories combined in quadrature.
- Code: `ip_tau_reco/entanglement_feasibility.py`, `ent_hllhc.py`, `ent_closure_syst.py`, `ent_realZ.py`, `ent_final.py`, `ent_lumi.py`, `make_figures_ent.py`; results `ip_tau_reco/results/json/*.json`; figures `ip_tau_reco/results/figures/ent_*.png`.

## Key numbers
- Events needed for 5 sigma (signal only / S/B=1 / 0.3 / 0.1, 2 %): exact 184 / 468 / 1123 / 4995; tauspin 1255 / 2809 / 7210 / 41098; no IP/SV 1639 / 3707 / 10114 / 73334; textbook 3469 / 7502 / 20740 / 163040.
- Final per experiment, 3 ab^-1 (resp 0 / 10 / 20 %): exact 9.8 / 9.4 / 9.3; tauspin 4.1 / 3.8 / 3.6; no IP/SV 3.6 / 3.2 / 3.0; textbook 2.7 / 2.5 / 2.3. Variants (window version): fakes unpolarised 3.59 vs 3.56; Z transverse 3.57; real-Z background template 4.18 vs 4.34 (in-sample setup), 3.79 with 10 % response.
- Luminosity scan (resp 10 %): tauspin 1.3 (300 fb^-1), 1.6 (450), 2.3 (1000), 3.1 (2000), 3.8 (3000), 5.2 (6000).
- Closure: spin-flat -> H vs real H(125)+j hold-out: g_s chi2/ndf 13.3/20, g_z 16.8/20, means 0.0303 vs 0.0302. Spin-flat -> Z vs real Z(125)+j: g_s 187/20 (means -0.034 vs -0.041; kinematics/production of Z+jet differ), hence the real-Z variant.

## Known limitations
- tau_had tau_had only; ATLAS Run-2 categories with a simple mass binning, not an optimised entanglement analysis; acceptances from our simulation.
- Fakes modelled with the Z spin state (or unpolarised); no data-driven fake template.
- Response systematics only as a global scale on the spin contrast; no pi0 energy scale/decay-modelling shape variations; no MC-statistics treatment of templates.
- Werner-family test (one-parameter); the general separability test of a full tomographic C matrix is not done.
- Interpretation caveats of "entanglement at colliders" (arXiv:2507.15949, 2507.15947): the claim is a witness under standard QM decay-as-measurement.
