# Review packet: can Z -> tau tau calibrate the spin response needed for H -> tau tau entanglement?

Code: `analysis/z_calibration/` (this branch `ariadne/zcalib-20261009`). Results: `results/*.json`, figures `results/fig*.png`.
Prior studies reused unchanged: `analysis/a1_current_model/` (3pi current reweighting, CLEO vs generator current),
`analysis/ip_tau_reco/ent_robust.py` (HL-LHC H->tau_h tau_h entanglement fit, 2026-10-07),
`analysis/calibration_response/evaluate_remote.py` (fixed point-h networks on calibration-shifted inputs).

## Question

Earlier work: the H->tau tau entanglement test collapses (1.84 sigma -> 0.15 sigma) when the transverse
analysing-power response k_T is free; the 3pi hadronic current model changes the learned polarimeter's
response by 6-15 %. Idea under test: use Z -> tau tau (known SM spin state) to calibrate the response in data,
which would also absorb the 3pi current uncertainty.

## Claims to check

C1 (structure). The tau-pair transverse spin correlation of any spin-1 source is traceless (C_nn = -C_rr):
it lives in the m = +-2 representation of rotations about the tau axis, while the Higgs (C_nn = C_rr) and the
CP-odd term (C_nr - C_rn) live in m = 0. A Z calibration therefore measures the m = 2 response, not the m = 0
response the Higgs needs. With per-tau errors hhat_perp = kappa e^{i eps} h_perp, the m = 0 response is
<kappa- kappa+ cos(eps+ - eps-)> and the m = 2 response <kappa- kappa+ cos(eps+ + eps-)>: equal when the
per-tau errors are independent and azimuthally symmetric; different for tau-correlated errors.

C2 (3pi current). On the ATLAS full-reco validation cohort (59,390 events), moving nature from the generator
3pi current to the CLEO current changes the m = 0 response of the generator-teacher +IP/SV network by
x0.938 (any-3pi events) and x0.86 (3pi x 3pi), but the m = 2 response moves by the same factor: the
Z-calibrated residual rho_T / rho_Q = 0.998 [0.979, 1.015] (any 3pi), 0.98 [0.91, 1.06] (3pi x 3pi).
Response r_a = Cov_flat(O_a(hhat), S_a(h_true)); flat world = cohort weight / (pi_H f_H + pi_Z f_Z);
CLEO world multiplies by the Dalitz weight u and uses h_CLEO as truth.  (`transfer_a1.py`, fig1)

C3 (nominal MC transfer). r_T / r_Q = 1.00 +- 0.01 overall; +-3 % across softer-tau energy quartiles;
up to ~8 % by decay-mode pair (3pi x 3pi 1.08 +- 0.04, pi x 3pi 0.96). (`transfer_kin.py`, fig3)

C4 (requirement). In the 10/07 robust model, the significance is unchanged for a k_T prior up to 30 %
(1.85 -> 1.84 sigma side-bands, 3.02 -> 2.98 sigma with tau = 1 auxiliary control), keeps >= 95 % at 50 %,
and collapses only when k_T is unbounded (null fit k_T = 30). A calibration-free Cauchy-Schwarz bound
(k_T <= 4.1) is not enough (0.97 / 1.56 sigma). (`kt_requirement.py`, fig2)

C5 (statistical power). With an approximate Z tensor polarisation (CS frame, A0 = A2 = qT^2/(qT^2+65^2)),
the tauspin estimate and a reco CS frame from visible + collinear MET, sigma(kappa) sqrt(N) = 16;
with ~1.09 M Z -> tau_h tau_h in the H categories at 3 ab^-1 (72 % with both polarimeters),
sigma(kappa) ~ 1.8 % (3 ab^-1), 8.5 % (140 fb^-1). (`z_state.py`)

C6 (detector shifts). Fixed 9/30 point-h networks rerun on shifted validation inputs (pi0 energy scale
+-1,2,5 %, track momentum scale +-1,2 %, MET scale +-5 %; sagitta variants dropped). +IP/SV (local) network:
single transverse response moves by <= 0.5 %, Z-calibrated residual within 0.1 %. No-IP/SV network: single
moves up to 3 % (pi0 -5 %: 0.968), residual <= 1.1 % (pi0 -5 %: 1.011 [1.003, 1.019]). These scale shifts are
weak stress tests for the transverse response; resolution mis-modelling and tau-correlated effects (PV
position shared by both IPs) were not tested. (`transfer_calib.py`, fig4)

Overall verdict proposed by main: physically sound and statistically easy, but calibration is not the
sensitivity bottleneck; the Z calibration turns an assumed ~30 % prior into a data-driven validation.
Not a stand-alone high-impact result.

## What not to claim

No real-data result. No claim on the absolute HL-LHC significance beyond the 10/07 model. The Z state model
in C5 is a statistical-power model (A_i parametrised), not a precision prediction. The 3pi "CLEO world" is a
model-distance benchmark, not nature.
