# Classical posterior polarimeter with the reco-rho pi0 marginalised

Automatic ARIADNE run `tauspin-rho-pi0-marginal-20261022` (Vault `ARIADNE/Runs/`), branch
`ariadne/auto-20261022-pi0-marginal`.  Large artifacts on lxgpu02 under `~/rho-pi0-marginal-20261022/` (`out/`, `logs/`,
one `code-<commit>` copy per deployment).  ATLAS full-reco H/Z cohort; every AUC on VALIDATION (59,390 events,
parent-overlap weights), every fitted table on TRAIN, test never loaded.

Question (from the 10/17 and 10/18 runs): the 10/17 classical posterior (IP/SV/MET/tau mass shell/|M|^2, no ML) is
0.047 below the main-line learner on rho x rho, and a truth-pi0 oracle closes all of it.  How much does a physical
treatment of the reco pi0 error (TRAIN response as likelihood, rho line shape and phase space as prior) recover?

| file | what it does |
|---|---|
| `p0_train_truth.py` | truth pi+- / pi0 / nu of every TRAIN event with a truth-rho side, re-read from ROOT |
| `response.py` | stage-1 TRAIN pi0 response (u = log E_reco/E_true, l = log angle) per raw mode and true-E bin, proposal table, tau-energy prior; holds out event_number % 5 == 0 |
| `response_cond.py` | stage-2 response conditioned on the hypothesis pi+-/pi0 separation (angle x E) and the reco PFO-mass class |
| `posterior_marg.py` | the sampler: pi0 hypotheses, derived Jacobian, then IP/SV/MET/(mass) as in `../atlas_classical_posterior/posterior.py`; per-pair polarimeter with the pair's own pi0; PIT outputs; `--split train` for the TRAIN PIT subset |
| `eval_marg.py` | PIT summaries; AUC by truth pair, paired bootstrap, cross-fit combination with the learner, rho-side h quality |
| `make_figures.py` | figures in `results/` |

Main numbers (IP+SV+MET+parent mass, invE prior): rho x rho 0.575 -> 0.597 (+0.022 [0.015, 0.029]), overall
0.590 -> 0.604; truth-pi0 oracle 0.636 / 0.626; learner 0.621 / 0.629; combination with the learner +0.0005.
Stage 2 +0.002 (n.s.).  Parent-mass width 3 -> 1 GeV: no gain.  See the Vault run note for the claim boundary.
