# Can the reconstructed polarimeter vector measure the Higgs CP mixing angle?

Run `tauspin-cp-mixing-sensitivity-20260925` (Vault: `ARIADNE/Runs/`).  Nothing
is trained: every number comes from artifacts of earlier runs, evaluated with
numpy.  Only the H rows of the 59,390-event validation cohort are used; the
test split is never loaded.

For a CP-mixed tau Yukawa coupling `cos(phi) + i gamma5 sin(phi)` the ditau
spin correlation matrix in the (n, r, k) basis is `R(2 phi)` in the transverse
block and `-1` along k.  The sample was generated CP-even, so any `phi_tau` is
reached by the exact-h spin weight

    w(phi) = (1 + h-^T C(phi) h+) / (1 + h-^T C(0) h+)

and the score at `phi_tau = 0` is `S = 2 (h- x h+).k / (1 + h-^T C(0) h+)` --
the CP-odd triple product.  For any observable T, `d<T>/dphi = Cov(T, S)`, so
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
| `make_figures.py` | figures |

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
python3 make_figures.py
```

Needs numpy; the figures need matplotlib.  About five minutes on a laptop.
