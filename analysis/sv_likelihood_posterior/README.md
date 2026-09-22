# SV-likelihood posterior update

This analysis updates the saved full-reconstruction joint neutrino posterior
with the train-only secondary-vertex direction response from
`reco_sheet_identifiability`:

```text
q_SV(nu-,nu+ | reco,SV) proportional to
q_0(nu-,nu+ | reco) * L_SV(SV | nu-,nu+).
```

The primary likelihood is the decay-mode-conditional vMF core plus uniform
outlier mixture fitted in the prior sheet-identifiability run, tempered by the
scalar posterior calibration fitted on a disjoint training partition.  The
untempered mixture and the old global 7.90 mrad Gaussian are diagnostics, not
model-selection candidates.  A matched-geometry shuffle is the event-specific
null control.

The analysis deliberately uses `reco_visible_tau_lab4 + sampled neutrino` for
the candidate tau momentum.  The massless `reco_tau_lab4` projection is not
used.  Posterior h samples are the already saved values from the established
reco-visible `HybridPolarimeter` implementation.

Scripts:

- `resample_train.py` regenerates paired train neutrino/h draws from the frozen
  baseline flow so the 64-draw train posterior can be reweighted.  It records
  parity with the previously saved train h draws.
- `surface_mask.py` computes the operational exact reconstructed-surface mask
  with the same `metspace` implementation and sampling settings as WP-A.
- `build_representations.py` computes likelihood weights, posterior-quality
  diagnostics, weighted h means/moments, raw-distribution plots, and h metrics.
- `train_readouts.py` applies the prior fixed readout recipe to the new
  representations and produces paired-bootstrap H/Z AUC comparisons and the
  final baseline-to-SV-to-oracle figure.
- `export_results.py` reduces the full reports to compact CSV/JSON tables for
  versioned review without copying the large posterior arrays into Git.
- `sample_stability.py` checks whether the weighted h estimate is stable as the
  saved validation posterior is increased from 32 to 256 draws.

No test split is loaded.  Large arrays stay on the analysis host; curated
machine-readable summaries and figures are copied back into `results/`.
