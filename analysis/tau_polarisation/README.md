# Can the reconstructed polarimeter measure the tau polarisation?

Run `tauspin-tau-polarisation-20260927` (Vault: `ARIADNE/Runs/`).  No network is
retrained and no sample is generated: every number comes from artifacts of
earlier runs (`analysis/cp_mixing/data/`), evaluated with numpy.  `q2` fits small
closed-form linear readouts of visible variables, cross-fitted on halves of the
validation Z rows; the test split is never loaded.

The ditau spin density in the (n, r, k) basis is

    rho  ~  1 + B- . h- + B+ . h+ + h-^T C h+ .

Earlier runs all measured something about `C`: the mainline H/Z separation, the
2026-09-25 CP-mixing angle (the direction of the transverse block) and the
2026-09-26 entanglement study (its magnitude).  This run measures `B`, the
single-tau polarisation.

  * H -> tau tau is a scalar, so `B = 0` **exactly** -- an in-situ null that has
    passed through the same selection, reconstruction and network.
  * an unpolarised Z gives `B-_can = B+_can = -P_tau k_hat` with
    `P_tau = -2va/(v^2+a^2) = -0.147037` for `sin^2 theta_W = 0.23152`, and
    `dP_tau/dsin^2 theta = 7.870`.

Note the sign: the project's canonical `h` is -1 times the physical polarimeter
on both sides.  That cancels in `h- C h+`, so no earlier run was affected, but it
flips the first moment.  Written for the stored canonical h,
`f = 1 + P (h-_k + h+_k) + h-^T C h+`.

| script | what it does |
|---|---|
| `pol_density.py` | `B` and `C` of both samples from the polarised Dirac trace; `P_tau(sin^2 theta)` and its derivative |
| `pol_tools.py` | density, score, reweighting, linear-response sensitivity, and the control-region moment estimator |
| `pol_visible.py` | Upsilon, `cos(theta*)`, masses and the collinear-approximation energy fractions |
| `pol_data.py` | shared loading (wraps the entanglement run's loader) |
| `q0_structure.py` | what the raw first moment is made of; `p_0` moments overall and per mode; closure of the product-`p_0` model |
| `q1_sensitivity.py` | `sigma(P_tau)` per Z event for exact `h` and every reco arm, the Cramer-Rao bound, an explicit row-permutation null, and the per-side analysing power by decay mode |
| `q2_classical.py` | the same for `E_vis/E_tau` with the truth `E_tau`, Upsilon, `cos(theta*)`, and the best cross-fitted visible-only and visible+MET readouts |
| `q3_measure.py` | measure `P_tau` on the Z rows with the H rows as the control region; injection closure; per mode; with the network output |
| `q4_transport.py` | estimator closure on disjoint halves, the H+jet -> Z+jet transport bias and its kinematic reweighting, and how accurately `E_0[h_k]` must be known |
| `q5_translate.py` | `sigma(sin^2 theta_eff)`, the events needed to match LEP, and the three measurements in one currency |
| `make_figures.py` | figures (needs matplotlib) |

## Headline

`sigma(P_tau)` per 1e5 selected Z events, shape only, signal only:

| level | sigma(P_tau) | sigma(sin^2 theta_eff) |
|---|---:|---:|
| exact `h`, Cramer-Rao | 0.00438 | 5.6e-4 |
| reco + ideal-IP oracle | 0.00531 | 6.8e-4 |
| reco + 3 IP + SV | 0.00546 | 6.9e-4 |
| reco, no geometry | 0.00552 | 7.0e-4 |
| best visible + MET readout | 0.00666 | 8.5e-4 |
| `E_vis/E_tau` with the truth `E_tau` | 0.00903 | 1.15e-3 |

Reconstruction costs a factor 1.22 here, against 2.71 for the CP angle and 2.73
for the entanglement witness; IP and SV buy 1.01 against 1.25 and 1.20.  The
polarisation lives in the longitudinal component, which is the one the network
already measures well and the one the impact parameter does not touch.

The limit is not statistics.  The acceptance-induced first moment
`E_0[h_k] = +0.049` is of the same size as the polarisation signal, and using the
H rows as the control region leaves a bias of -0.042 on `P_tau`, reduced to
-0.018 by reweighting to the Z per-side kinematics.  That already equals the
statistical error of ~9e3 Z events.

## Inputs

Same files as `analysis/cp_mixing/data/` (not tracked); see that run's README for
their provenance.  `ladder_validation.npz` supplies the visible four-vectors,
reco decay products and MET used by `q2`.

## Reproducing

```sh
python3 q0_structure.py && python3 q1_sensitivity.py && python3 q2_classical.py
python3 q3_measure.py && python3 q4_transport.py && python3 q5_translate.py
python3 make_figures.py
```

Needs numpy; the figures need matplotlib.  About two minutes on a laptop.
