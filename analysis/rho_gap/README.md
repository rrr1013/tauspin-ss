# Dissecting the Reco-Rho Transverse Spin Gap (Oracle Matrix & Learner Intervention)

Automatic ARIADNE run `tauspin-rho-transverse-gap-20261024` (Vault `10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-rho-transverse-gap-20261024/`), branch `ariadne/auto-20261024-rho-gap`.
Large artifacts on lxgpu02 under `~/rho-gap-20261024/` (`out/`, `logs/`, `figures/`).
ATLAS full-reco H/Z cohort (59,390 validation events, parent-overlap weights); test never loaded.

| file | what it does |
|---|---|
| `learner_intervene_rho.py` | input interventions on fixed learner (full22 s42): IP rotation (+90°), flip (180°), |IP| const, track d0/z0 zeroed, neutral PFO zeroed/shuffled on rho vs pi sides |
| `eval_intervene_rho.py` | evaluates (hn, hr, hk) correlation with exact h for all intervention variants across pi, rho, 3pi |
| `posterior_rho_oracle.py` | modular classical posterior supporting the 2x2 oracle matrix on reco-rho sides: `--rho-pi0 {marg, truth, truthdir, truthe, fixed}` and `--rho-psi {reco, truth}` |
| `eval_rho_matrix.py` | H/Z weighted AUC across truth pairs, per-side quality, and paired bootstrap |
| `make_figures_rho.py` | figures for Vault (`fig1_learner_rho_intervention.png`, `fig2_rho_oracle_matrix.png`, `fig3_pi0_decomposition.png`) |
| `run_stride10_probe.sh` | launch script for stride-10 probe across all oracle configurations |
| `probe_diag.py` | fast probe evaluation summary |
