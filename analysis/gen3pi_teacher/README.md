# Generator-current 3pi teacher: reco point-h readout and null-space ceiling redone

Follow-up of `../mode_pair_auc_origin/`: the canonical exact h used the CLEO a1
current for 3pi, while the sample was generated with the TauDecay UFO
Kuehn-Santamaria current (<h_CLEO . h_gen> = 0.889).  Here only the 3pi
polarimeter is changed; inputs, recipe, seeds, readout and null-space
construction are those of `../hz_beyond_ceiling/`.

| file | what it does |
|---|---|
| `generator_current.py` | drop-in replacement for the CLEO current inside `HybridPolarimeter` (same frame and sign) |
| `build_targets.py` | copies `reco-h-supervision-20260906-targets/{train,validation}.npz`, re-reads truth from ROOT, replaces h on 3pi sides only; pi/rho closure against the stored h; optional truth surface with the new `h_ref` |
| `train_point_h.py` | unchanged p11 training with `targets_dir` pointed at the new targets |
| `nullspace_gen3pi.py` | unchanged p12 with the 3pi current of the loaded polarimeter replaced |
| `compare_point_h.py` | weighted fixed-readout AUCs, new - old per arm with paired bootstrap, h quality vs generator h |
| `compare_nullspace.py` | null-space LR AUCs, generator - CLEO on identical draws |
| `make_figures.py` | three figures |

Results (validation, test never loaded): reco +IP/SV 2-seed mean 0.6264 -> 0.6287
(+0.0023 [+0.0015, +0.0031]); ceiling 3 IPs+SV+parent mass 0.6505 -> 0.6596;
exact h 0.7187 -> 0.7293 (weighted).  Outputs in
`analysis/outputs/gen3pi-teacher-20260924/`; large artifacts on lxgpu02 under
`~/gen3pi-teacher-20260924/` (`run_train.sh`, `run_nullspace.sh`).

The null space fixes the contraction mass at 1.7769 GeV, so the replaced current
reconstructs the neutrino from that mass; hypotheses lie on that shell and are
exact, while the truth-neutrino closure differs by up to 1e-3 because the truth
tau mass is 1.7770 GeV.
