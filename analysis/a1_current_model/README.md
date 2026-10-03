# 3pi hadronic-current model dependence of learned polarimeters

The cohort's 3pi decays were generated with the TauDecay UFO (Kuehn-Santamaria)
current.  Per-event weights move the validation cohort (59,390 events, test split
never read) to a world where tau -> 3pi nu follows the CLEO current
(Cherepanov-Veelken C++), and fixed estimators (exact h, reco point-h networks
trained with generator-h or CLEO-h teacher, frozen H/Z readouts) are re-evaluated.

    w = prod_{3pi sides} u * f(h_CLEO) / f(h_gen),   u = omega_CLEO / omega_gen * n(m3pi)

| file | role |
|---|---|
| `currents.py` | generator current with its spin-averaged rate; generator a1 form factor |
| `remote_eval.py`, `remote_toy.py` | ICEPP: both currents through the same HybridPolarimeter path / on the toy |
| `toy.py` | flat tau -> 3pi nu phase space at the cohort's m3pi values (normalisation n) |
| `reweight.py` | CLEO-world weights (primary: generator m3pi spectrum; variant `lineshape='cleo'`) |
| `validate.py` | fig 1: Dalitz marginals vs rates, normalisation, correlation closure |
| `analyze.py` | AUCs in both worlds, teacher/world contrasts, template fraction, triple product |
| `responses.py` | spin-response (calibration) ratios, AUC world effect vs m3pi |
| `dalitz_discrimination.py` | fig 4: model separation from reconstructed tracks |
| `decompose.py`, `interaction.py` | review follow-up: Dalitz/spin split, offsets, interaction CIs |
| `make_figures.py` | figs 2, 3, 5 |

Inputs copied into `data/` (untracked, ~170 MB): ICEPP `~/a1-current-model-20261004/output/`
(`eval.npz`, `toy*.npz`, `reco_pions_validation.npz`) and the p11 readout scores of
`gen3pi-teacher-20260924` (`data/gen/`) and `hz-beyond-ceiling-20260923` (`data/cleo/`).
Run note: `AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-a1-current-model-20261004/`.
