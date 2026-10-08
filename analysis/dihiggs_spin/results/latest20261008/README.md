# Latest-baseline ARIADNE run — 2026-10-08

Read [analysis_note.md](analysis_note.md) for the scientific conclusions and figures; [review.md](review.md) records independent reviews and their resolution. The frozen protocol is [latest_protocol.md](../../latest_protocol.md).

- `spin/`: identical-HH-event readout experiment, paired MC intervals, Product15 and K+Modes controls, input provenance and exact-row overlap check. This compares expressions on the same TauSpin estimates, not trained TauPolaris networks.
- `lephad/`: **final** fixed-score inference from source `9f4b634`, with models trained at `e34d272`. All 2,296 profiled fits succeeded. `summary.json`, `aggregate.json`, `bootstrap.json`, `fit_diagnostics.json` and `profile_validation.json` are the final numerical artifacts. Flavor and GBDT ratios with unsupported baseline bins remain diagnostic.
- `figures/`: final PNG/PDF figures. All four spin PNGs and all ten final lep-had PNGs were opened by main and an independent validity reviewer.
- `sources/`: the official ATLAS auxiliary tables used for whole-SR context.
- `lephad/numerical-attempt/` and `figures/lephad/numerical-attempt/`: archived **computational failures**, preserved for provenance. Their sensitivity values and incomplete bootstrap intervals are superseded and must not be used as physics results.

Models, scores, prepared large arrays and raw execution logs remain on `lxgpu02:~/dihiggs-latest-20261008/`. This directory contains only the report, small numerical results, convergence records, provenance and figures. The existing `outputs` symlink is not part of this run's Git changes.
