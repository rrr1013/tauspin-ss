# Static azimuthal modulation of reconstructed h

Autonomous ARIADNE run `tauspin-h-prior-leakage-20261001`.

The first stage decomposes the complex first harmonic of the two predicted
polarimeter azimuths,

```text
m_obs  = E[exp(i(phi_minus - phi_plus))]
m_fact = E[exp(i phi_minus)] E[exp(-i phi_plus)]
m_conn = m_obs - m_fact
```

into a factorisable one-side contribution and a connected same-event
contribution.  Z rho-rho is the primary population because the v2 Z sample has
no transverse spin correlation and the rho polarimeter is independent of the
CLEO-vs-generator 3pi-current choice.

The input directory is the untracked `data/` directory from the 2026-09-25 CP
mixing run.  The test split is not read.

```sh
python moment_decomposition.py --input-dir /path/to/cp_mixing/data
python make_figures.py
```

Outputs are written below `results/` and `figures/`.

`counterfactual_remote.py` is the fixed-network Stage-2 evaluator.  It is run
on ICEPP from a clean checkout with explicit paths to the existing p11 code,
generator-current targets, geometry features, selected checkpoint, reference
predictions, and preprocessing statistics.  It writes only validation
predictions and donor identities; no network is trained.
