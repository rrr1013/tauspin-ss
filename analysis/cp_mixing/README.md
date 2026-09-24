# Can the reconstructed polarimeter vector measure the Higgs CP mixing angle?

Run `tauspin-cp-mixing-sensitivity-20260925` (Vault: `ARIADNE/Runs/`).  No
network is retrained: every number comes from artifacts of earlier runs,
evaluated with numpy.  (`p5`/`p10` do fit a small closed-form polynomial
readout of `h_pred`; that is a supervised fit, cross-fitted on validation
halves in `p10`.)  Only the H rows of the 59,390-event validation cohort are
used; the test split is never loaded.

For a CP-mixed tau Yukawa coupling `cos(phi) + i gamma5 sin(phi)` the ditau
spin correlation matrix in the (n, r, k) basis is `R(2 phi)` in the transverse
block and `-1` along k.  The sample was generated CP-even, so any `phi_tau` is
reached by the exact-h spin weight

    w(phi) = (1 + h-^T C(phi) h+) / (1 + h-^T C(0) h+)

and the score at `phi_tau = 0` is `S = (h-^T C'(0) h+) / (1 + h-^T C(0) h+)`
= `(2/beta) (h- x h+).k / f` -- the CP-odd triple product.  For any observable T, `d<T>/dphi = Cov(T, S)`, so
`sigma_N = sqrt(Var T / N) / |Cov(T, S)|` prices any method on identical events
without simulating a new sample.

| script | what it does |
|---|---|
| `cp_density.py` | `C(phi)` from the polarised Dirac trace, exact in beta; no spinor phase convention enters |
| `cp_tools.py` | score, spin reweighting, linear-response sensitivity, acoplanarity |
| `p0_align.py` | row alignment, `h` rebuild closure, side/handedness/basis audit |
| `p1_density_check.py` | `C(phi)` against the ultra-relativistic closed form |
| `p2_sensitivity.py` | ideal and reco sensitivity, mode-pair split, shuffle control, reweighting closure and ESS |
| `p3_classical.py` | classical phi*_CP (IP method, neutral-pion method) and its y-flip control |
| `p4_classical_full.py` | + truth-input version, |y1y2| weighting, binned Asimov, finite-difference cross-check |
| `p5_readout.py` | best readout of `h_pred` fitted on the train split, applied to validation |
| `p6_geometry_value.py` | what each geometry block is worth, CP vs H/Z, from the same predicted `h` |
| `p7_static_modulation.py` | the spin-independent part of the reco-`h` acoplanarity, measured on Z |
| `p8_closure_fit.py` | inject `phi_tau`, fit it back: bias and spread of the quoted sigma |
| `p9_review_followups.py` | review follow-ups: finiteness of the Fisher information, rest-frame `y`, same-seed geometry |
| `p10_review_round2.py` | paired comparisons, mode pairs on one scale, cross-fitted readout with an exact-`h` positive control, static modulation across arms |
| `p11_classical_phase.py` | phase shift of the classical phi*_CP with bootstrap errors, and its Z control |
| `p12_density_closure.py` | external closure of the spin-density model against the ideal-C(0) re-decay toy; cross-fitted classical + learned combination |
| `make_figures.py` | figures |

Two independent reviews (`review/`) are folded in; `p9`-`p11` exist because of
them, as does `p12`.  See the run note section "レビューと訂正" for what changed.

## Inputs (ICEPP lxgpu02, copied into `data/`, not tracked)

| file | source |
|---|---|
| `truth_surface_cleo.npz` | `~/azimuth-nullspace-20260918/artifacts/truth_surface.npz` |
| `gen3pi_<arm>.npz`, `gen3pi_<arm>_train.npz` | `~/gen3pi-teacher-20260924/artifacts/p11/<arm>/{validation,train}_predictions.npz` |
| `cleo_<arm>.npz` | `~/hz-beyond-ceiling-20260923/artifacts/p11/<arm>/validation_predictions.npz` |
| `ladder_validation.npz` | `~/tauspin-full-reco-information-ladder-20260917-inputs-v1/validation.npz` |
| `ip_tracks.npz` | `~/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz` |
| `hz_auc.json` | `~/hz-beyond-ceiling-20260923/artifacts/p11_readout/auc.json` |

## Reproducing

```sh
python3 p0_align.py && python3 p1_density_check.py
python3 p2_sensitivity.py && python3 p3_classical.py && python3 p4_classical_full.py
python3 p5_readout.py && python3 p6_geometry_value.py && python3 p7_static_modulation.py
python3 p8_closure_fit.py && python3 p9_review_followups.py
python3 p10_review_round2.py && python3 p11_classical_phase.py && python3 p12_density_closure.py
python3 make_figures.py
```

Needs numpy; the figures need matplotlib.  About five minutes on a laptop.
