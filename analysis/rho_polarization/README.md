# Rho Decay Internal Polarization Analysis

This directory contains the code and artifacts for the ARIADNE run on rho decay internal polarization dynamics (`tau -> rho nu -> pi+/- pi0 nu`) and energy sharing asymmetry `y = (E_pi+/- - E_pi0) / (E_pi+/- + E_pi0)`.

## Scripts

- `analyze.py`: Evaluates per-side polarimeter quality vs `y`, calculates H/Z AUC across 2D `(y0, y1)` phase space, evaluates detector charge/neutral asymmetry, unphysical visible mass tail (`m_vis > m_tau`), and effective spin correlation matrix tomography.
- `make_figures.py`: Generates publication-ready figures in `figures/`.

## Results

- `results/summary.json`: Complete numerical metrics, bootstrap 95% confidence intervals, and category breakdowns.
- `results/arrays.npz`: Compressed arrays containing per-event quantities for plotting.
- `figures/`: High-resolution figures (Fig 1 to Fig 4).
