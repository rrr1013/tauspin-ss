# Classical posterior vs learner: where the remaining gap is (tau kinematics, IP azimuth)

Automatic ARIADNE run `tauspin-taukin-gap-20261023` (Vault `ARIADNE/Runs/`), branch `ariadne/auto-20261023-taukin-gap`.
Large artifacts on lxgpu02 under `~/taukin-gap-20261023/` (`out/`, `logs/`, `figures/`, one `code-<commit>` per deployment).
ATLAS full-reco H/Z cohort; every AUC on VALIDATION (59,390 events, parent-overlap weights), every table on TRAIN
(event_number % 5 == 0 held out), test never loaded.

| file | what it does |
|---|---|
| `kin_tables.py` | TRAIN tables: |IP| vs expected IP (decay length), SV length, MET by visible ditau pT, truth m(tau tau) |
| `posterior_kin.py` | classical posterior on top of the 10/22 primary (`../rho_pi0_marginal/posterior_marg.py`): `--vis track`, `--decay-length`, `--mass-prior train`, `--met-model ptbin`, IP response variants `--ip-fine / --ip-aniso / --ip-orient / --ip-joint`, diagnostics `--met-scale`, `--ip-vm`, `--ip-truth-psi`, `--stride` |
| `eval_kin.py` | AUC by truth pair, per-side h quality, tau energy, PIT, ESS, side swap with the learner, paired bootstrap |
| `learner_intervene.py` | input interventions on the fixed full22 s42 learner (partner / own geometry shuffle, own IP rotate / flip / |IP| fixed / offset zero, token d0 z0 zeroed) |
| `make_figures.py` | figures in `results/` |
| `probe/` | the one-off probes and launch scripts used during the run (paths are the lxgpu02 run directory) |

Main numbers (IP+SV+MET+parent mass): 10/22 primary 0.6044 -> visible system = track 0.6065 -> + decay length 0.6077 ->
+ IP response by signed measured orientation 0.6107 (cumulative +0.0063 [+0.0040, +0.0086]); learner 0.6287
(gap -0.0180 [-0.0212, -0.0151]).  pi sides: tau energy (from h_k) classical 0.950 vs learner 0.954; transverse
0.74/0.67 -> 0.77/0.70 vs learner 0.80/0.74.  The measured PV-referenced IP azimuth is pulled toward the z0 axis
(TRAIN, fig5); a response that depends only on the azimuth difference cannot represent it.  Side swap: about 2/3 of
the remaining gap is on reco-rho sides.  See the Vault run note for the claim boundary.
