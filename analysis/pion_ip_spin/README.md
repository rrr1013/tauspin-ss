# 1-prong Pion IP Kinematics and Spin Headroom Analysis

Explores the physical origin of the large headroom between reconstructed IP (`full22`, AUC 0.6550) and ideal IP (`idealip22`, AUC 0.6947) in $\tau \to \pi\nu$ (1p0n) decays.

## Key Findings

1. **Kinematic Lever-Arm Collapse**: In $\tau \to \pi\nu$, the 3D impact parameter is $d_0 = L \sin\theta_{\tau\pi} \approx c\tau_0 \tan(\theta^*/2)$. As visible energy fraction $x = E_\pi / E_\tau \to 1$ (forward emission), the lever arm collapses from $132\,\mu\text{m}$ down to $32\,\mu\text{m}$. Below the detector resolution threshold ($d_0 \lesssim 30\,\mu\text{m}$), the reconstructed IP direction error explodes from $6.8^\circ$ up to $37.1^\circ$.
2. **Polarimeter Response Asymmetry**: Transverse polarimeter correlation $\text{corr}(h_\perp^{\rm pred}, h_\perp^{\rm exact})$ for `full22` reaches $0.822$ at low $x$ but plunges to $0.555$ at high $x$. Meanwhile, longitudinal correlation $h_k$ ($0.905$) is invariant to IP.
3. **Phase Space AUC Stratification in $\pi\times\pi$**:
   - In low-$x$ events ($x_1, x_2 < 0.45$), `full22` achieves AUC **0.6700** (a massive $+0.0740$ gain over `base` 0.5960, capturing 62% of the gap to `idealip22` 0.7183).
   - In high-$x$ events ($x_1, x_2 \ge 0.75$), the vanishing physical lever arm causes vertex noise to corrupt the prediction, pulling `full22` (0.6369) below `base` (0.6554) by $-0.0186$, while `idealip22` reaches $0.7649$ and exact $h$ reaches $0.8125$.
   - When $\min(d_{0,1}, d_{0,2}) \in [70, 100)\,\mu\text{m}$, `full22` reaches AUC **0.7295**, virtually closing the entire gap to `idealip22` (0.7341).
4. **Mixed Modes and Quality Cuts**:
   - In $\pi\times\rho$ ($N=15,943$), selecting events with $d_{0,\pi} \ge 90\,\mu\text{m}$ boosts `full22` AUC from $0.6338 \to 0.6454$ ($+0.0141$ over base).
   - In $\pi\times 3\pi$ ($N=5,609$), selecting $d_{0,\pi} \ge 90\,\mu\text{m}$ boosts `full22` AUC from $0.6158 \to 0.6280$ ($+0.0177$ over base).
   - In $\pi\times\pi$, a pure reconstructed selection $\min(d_{0,1}, d_{0,2}) \ge 60\,\mu\text{m}$ retains 27.1% of events with AUC **0.6656** (compared to inclusive 0.6475).

## Scripts

- `analyze.py`: Loads validation cohort, computes kinematic lever arms, IP angular resolutions, and evaluates model predictions.
- `make_figures.py`: Generates Figures 1–4.
