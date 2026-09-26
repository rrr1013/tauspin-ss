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
  * an unpolarised Z gives `B-_phys = B+_phys = -P_tau k_hat = +0.14715 k_hat`
    with `P_tau = -2va/(v^2+a^2) = -0.147037` for `sin^2 theta_W = 0.23152`, and
    `dP_tau/dsin^2 theta = 7.870`.

Note the sign: the project's canonical `h` is -1 times the physical polarimeter
on both sides.  That cancels in `h- C h+`, so no earlier run was affected, but it
flips the first moment, so `B_can = -B_phys = +P_tau k_hat` and the density
written for the stored h is `f = 1 + P (h-_k + h+_k) + h-^T C h+`.  Both sides
carry the same B in this common basis: `P_tau` is the tau-minus helicity
polarisation and its momentum is `-k_hat`.  (`analysis/entanglement/z_density.py`
has a docstring saying `B- = -B+`, which contradicts its own trace output; that
run only used `|B|`, so its results are unaffected, and the file is left alone.)

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
| `q5_translate.py` | `sigma(sin^2 theta_eff)`, the selected-signal counts at which it equals the LEP uncertainty, and the three measurements in one currency |
| `q6_followups.py` | is `h_pred` a sufficient statistic; `E_0[h_k]` against the truth tau pT and its one-parameter model |
| `q7_bias_origin.py` | transport or a wrong `P_0`?  substitution, mode ordering, and what `sin^2 theta_W` would have had to be |
| `q8_review.py` | the numerical claims of the two reviews: truth vs reco mode in the reweighting, a bootstrap that refits the density ratio, the significance of the transport difference, fold stability of the visible readouts, and a `p_cut` holdout |
| `make_figures.py` | figures (needs matplotlib) |

## Headline

`sigma(P_tau)` per 1e5 selected Z events, shape only, signal only:

| level | sigma(P_tau) | sigma(sin^2 theta_eff) |
|---|---:|---:|
| exact `h`, Cramer-Rao | 0.00438 | 5.6e-4 |
| reco + ideal-IP oracle | 0.00531 | 6.8e-4 |
| reco + 3 IP + SV | 0.00546 | 6.9e-4 |
| reco, no geometry | 0.00552 | 7.0e-4 |
| best cross-fitted visible + MET readout | 0.00669 | 8.5e-4 |
| best cross-fitted visible-only readout | 0.00710 | 9.0e-4 |
| `E_vis/E_tau` with the truth `E_tau`, one variable | 0.00903 | 1.15e-3 |

Reconstruction costs a factor 1.22 here, against 2.71 for the CP angle and 2.73
for the entanglement witness; IP and SV buy 1.01 against 1.25 and 1.20.  The
polarisation lives in the longitudinal component, which is the one the network
already measures well and the one the impact parameter does not touch.

The limit is the acceptance, not the statistics.  The visible-pT selection gives
the unpolarised measure a first moment `E_0[h_k] = +0.049` (pi sides +0.174),
the same size as the polarisation signal itself, and `dP_hat/dE_0[h_k]` is -3.3
per side for a pure first-moment offset, -4.1 if the whole observed H/Z
difference is attributed to it.  So `E_0[h_k]` has to be known to about 1.3e-3
at 1e5 selected Z events and 1.3e-4 at 1e7 -- 1 to 3% of its own size.  That
tolerance is the transferable statement.

The H -> Z transport exercise is a demonstration of that, not a measurement: the
uncorrected bias is -0.042 +- 0.021 (2.0 sigma), matched by an H - Z difference
in `E_0[h_k]` of +0.021 in the sum (2.3 sigma), and after reweighting the control
to the Z per-side kinematics it is consistent with zero (`+0.008 +- 0.027` using
the reconstructed decay mode, `-0.005 +- 0.025` using the truth mode, from a
bootstrap that refits the density ratio in every replica).  No crossover event
count is quoted from it.

Two independent reviews (`review/`) are folded in; `q8_review.py` exists because
of them, and the retraction above is their result.  See the run note section
"レビューと訂正".

## Inputs

Same files as `analysis/cp_mixing/data/` (not tracked); see that run's README for
their provenance.  `ladder_validation.npz` supplies the visible four-vectors,
reco decay products and MET used by `q2`.

## Reproducing

```sh
python3 q0_structure.py && python3 q1_sensitivity.py && python3 q2_classical.py
python3 q3_measure.py && python3 q4_transport.py && python3 q5_translate.py
python3 q6_followups.py && python3 q7_bias_origin.py && python3 q8_review.py
python3 make_figures.py
```

Needs numpy; the figures need matplotlib.  About two minutes on a laptop.
