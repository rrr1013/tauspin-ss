# Audit of Track-Frame vs Visible-Frame Tilt in Tau Decay Impact Parameters

This directory contains the code and artifacts for the ARIADNE exploratory run
`tauspin-rho-ip-frame-tilt-20260923` (Vault: `ARIADNE/Runs/tauspin-rho-ip-frame-tilt-20260923/`).

## Context and Findings

In `tauspin-hz-beyond-ceiling-20260923`, the PV-referenced impact parameter improved
$1\mathrm{p}0\mathrm{n} \times 3\mathrm{p}0\mathrm{n}$ by $\Delta\mathrm{AUC} = +0.017$ and
$1\mathrm{p}0\mathrm{n} \times 1\mathrm{p}0\mathrm{n}$ by $+0.007$, but yielded only $+0.003$
in $\rho \times \rho$ ($1\mathrm{p}1\mathrm{n} \times 1\mathrm{p}1\mathrm{n}$), despite $\rho$
constituting 56.3% of the cohort.

This audit establishes:
1. **Intrinsically sharp IP in $\rho$**: Measured around the charged pion track axis $\hat{t}$,
   the IP azimuth resolution in $\rho$ decays is identical to $\pi$ ($\langle\cos\Delta\phi\rangle = 0.6913$ vs $0.6834$).
2. **Parallax / Frame tilt**: Because $\vec{p}_{\mathrm{vis}} = \vec{p}_{\pi^\pm} + \vec{p}_{\pi^0}$,
   the track direction $\hat{t}$ is tilted from the visible tau direction $\hat{k}_{\mathrm{vis}}$ by
   median 6.10 mrad. Projecting the IP into the visible $(n, r)$ basis without accounting for the
   track offset vector $\vec{t}_\perp$ causes a geometric parallax error, cutting the apparent
   azimuth correlation by half ($\langle\cos\Delta\phi\rangle = 0.3440$, median $|\Delta\phi| = 0.88$ rad).
3. **$\Upsilon$ dependence**: In decays where $\pi^0$ carries most energy ($\Upsilon < 0$),
   $\theta_{tk}$ expands to $10 - 16$ mrad and naive visible-frame azimuth correlation collapses to
   $\sim 0.06 - 0.10$ (virtual noise).
4. **Null space vs Learned model gap**: In the null-space solution set (P5), where the IP likelihood
   was evaluated in the track frame, $\rho \times \rho$ gained $\Delta\mathrm{AUC} = +0.0158$ and
   $\pi \times \rho$ gained $+0.0278$. In the learned point-h model (P9), where the naive visible-frame
   projection was fed to the network, $\rho \times \rho$ captured only 19% of this gain ($+0.0030$).

## Files

- `audit_frame_tilt.py`: Extracts and evaluates the opening angles, azimuth distributions, $\Upsilon$ dependencies, and parallax corrections on the validation cohort.
- `plot_frame_tilt.py`: Generates publication-quality figures:
  - `figures/fig1_frame_tilt_distributions.png`: Opening angle and azimuth resolution distributions.
  - `figures/fig2_upsilon_and_parallax_mechanism.png`: Parallax geometry diagram and $\Upsilon$ dependence.
  - `figures/fig3_nullspace_vs_learned_mode_hierarchy.png`: Mode-pair $\Delta\mathrm{AUC}$ comparison between P5 (track frame) and P9 (visible frame).
- `results/audit_summary.json`: Complete quantitative metrics in machine-readable JSON format.
