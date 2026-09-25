# Is the ditau spin state entangled, and does the reconstruction keep it?

Run `tauspin-ditau-entanglement-20260926` (Vault: `ARIADNE/Runs/`).  Nothing is
retrained and no new MC is generated: every number comes from artifacts of
earlier runs (`analysis/cp_mixing/data/`), evaluated with numpy.

For this sample the two generated spin states are known analytically from the
polarised Dirac trace:

| | `C` in (n, r, k) | `B-`, `B+` | state |
|---|---|---|---|
| H | `diag(1, 1, -1)`, exact for any beta | 0 | pure Bell state `(|ud> + |du>)/sqrt2` |
| Z | `diag(0, 0, +1)` | `0.147 k` | separable mixture `0.574 |uu> + 0.426 |dd>` |

so the H/Z separation the main line measures as an AUC is, at truth level, the
separation between a maximally entangled state and a separable one.

| script | what it does |
|---|---|
| `ent_tools.py` | state parameterisation, PPT / concurrence / Horodecki `m12`, the calibrated estimator, projection onto the physical state space |
| `z_density.py` | spin density of `tau+ tau-` from an unpolarised Z, by the polarised Dirac trace |
| `ent_data.py` | shared loading of the 59,390-event validation surface and of each arm |
| `q0_naive.py` | where the textbook plug-in `C = 9<h- h+^T>` fails, and the anisotropy of `p0` that causes it |
| `q1_unfold.py` | three estimators (naive, second-moment, calibrated), the identifiability limit, injection closure against a 236,688-event independent calibration |
| `q2_reco.py` | per-arm sensitivity, events needed to certify entanglement, instrumental correlation of the two predicted polarimeters, mode pairs |
| `q4_zcontrol.py` | Z control, and the sharpest test: response calibrated on H applied to Z |
| `q5_checks.py` | the instrumental correlation without any reweighting (Z has zero generated transverse correlation), why cutting on `f` is not a control, and the mode-pair response conditioning |
| `q3_figures.py` | figures |

## The estimator, and what it can and cannot do

With `phi = (h-, h+, h- outer h+)` built from the exact polarimeter and `g` the
same features built from whatever vector is available,

    E_theta[g] = (A + R theta) / Z(theta),
    A = E0[g],  R = E0[g phi^T],  Z(theta) = 1 + E0[phi] . theta

and the measurement is `(R - <g> e^T) theta = <g> - A`.  `E0` is the unpolarised
measure, reached from a generated sample by the weight `1/f`.

The 15 moments do **not** identify `theta` on their own: calibrating on the
sample being measured returns the assumed state exactly, for any assumption.
The unpolarised measure has to come from outside.  Every reco number here is
therefore quoted Asimov-style -- the central value is the generated state by
construction and the rows differ in sigma -- and the estimator is validated by
injecting states into events the calibration never saw.

## Reproducing

```sh
export PYTHONPATH=../mode_pair_auc_origin:../cp_mixing
python3 q0_naive.py && python3 q1_unfold.py
python3 q2_reco.py && python3 q4_zcontrol.py && python3 q5_checks.py
python3 q3_figures.py
```

Needs numpy; the figures need matplotlib.  `q1` takes a few minutes because of
the joint bootstrap over the 236,688-event calibration split; the rest is under
two minutes on a laptop.
