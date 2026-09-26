# Independent Review of the Tau Polarisation Analysis (`tauspin-tau-polarisation-20260927`)

**Verdict:** The theoretical framework, sign conventions, linear-response pricing, and multi-body polarimeter advantage over classical $E_{\text{vis}}/E_\tau$ are mathematically rigorous and verified, but the measurement is overwhelmingly systematics-dominated by acceptance-induced first moments ($d\hat{P}/dE_0[h_k] \approx -2.04$), rendering the $H \to Z$ control-region strategy insufficient (residual bias $\Delta P = -0.018$ exceeds statistical uncertainty beyond $9.2 \times 10^3$ events) and the cross-run comparison ($1.22$ vs $2.71/2.73$) apples-to-oranges without explicit per-component decomposition.

---

## Executive Summary & Scorecard

| Analysis Component | Claimed Status | Verified Verdict | Severity / Impact |
|---|---|---|---|
| **Sign & Polarisation Conventions** | $B_{\text{can}} = -P_\tau \hat{k}$, $P_\tau = -0.147037$ | **Verified Correct**; all relative signs, Dirac traces, and $Z-H$ shifts close | Clean / Minor notation clarity |
| **Linear Response & Score Pricing** | Exact Asymptotic $\sigma(P_\tau)$ on identical events | **Verified Sound**; 2-fold cross-fitting prevents leakage; shape-only is mandatory | Sound |
| **Multi-body Decays vs $x_{\text{truth}}$** | Network beats truth $E_{\text{vis}}/E_\tau$ ($0.0055$ vs $0.0090$) | **Verified Correct Physics**; $x_{\text{truth}}$ has zero analyzing power for $a_1 \to 3\pi$ and $0.45$ for $\rho$, while network learns internal currents $J^\mu$ | Sound / Important Insight |
| **Acceptance Model ($E_0[h_k]_\pi = p_{\text{cut}}/p_{T,\tau}$)** | One-parameter model captures $p_T$ dependence | **Verified Correct** ($p_{\text{cut}} = 18.8$ GeV matches 1st percentile reco $p_T \approx 20.7$ GeV) | Sound |
| **$H$-Control Transport & Bias** | $H$ provides unpolarised control $p_0$; bias is $-0.042 \to -0.018$ | **Severe Limiting Systematic**; $d\hat{P}/dE_0 = -2.04$; $E_0[h_k]$ must be known to $<1.7\times 10^{-3}$ | **Critical** |
| **Cross-Run Comparison ($1.22$ vs $2.71/2.73$)** | Polarization reconstructed $2.2\times$ better than CP/entanglement | **Confounded Comparison**; mixes longitudinal vs transverse physics with sample kinematics ($Z$ vs $H$) | **Major** |
| **LEP Equivalence ($1.6\times 10^5$ events)** | Statistical equivalence to LEP $\sigma(\mathcal{A}_\tau) = 0.0043$ | **Overstated in Real Hadron Collider Context**; ignores QCD fakes, TES, and transport systematics | **Major** |
| **Disjoint-Half Closure Scope** | Closure on $H \to H$ ($+0.002 \pm 0.012$) validates method | **Statistically Valid on i.i.d., but Non-informative for Transport** | **Moderate** |

---

## Detailed Findings (Ordered by Severity)

---

### Finding 1 [CRITICAL]: $H \to Z$ Transport Bias and Extreme Sensitivity to Acceptance First Moment $E_0[h_k]$

* **What is claimed:**
  The run claims that $H \to \tau\tau$ provides a model-independent, in-situ control region where $B = 0$ exactly, allowing the unpolarised measure $p_0$ to be extracted via $1/f_H$ importance weighting. It measures $P_\tau$ on $Z$ rows as $-0.1895$ (bias $-0.0425$ against truth $-0.1470$), reduced to $-0.1650$ (bias $-0.0180$) after per-side kinematic reweighting on $(p_{T,\text{vis}}, \eta, \text{mode})$.

* **What was checked:**
  1. We verified the algebra of `pol_tools.control_moments` and `invert_mean` by full symbolic expansion:
     $$E_Z[T](P) = \frac{E_0[T] + P E_0[T u] + E_0[T v_Z]}{1 + P E_0[u] + E_0[v_Z]}$$
     where $u = h_{-,k} + h_{+,k}$ and $v_Z = h_-^T C_Z h_+$. Inverting for $P$ is exact and linear-fractional.
  2. We computed the sensitivity of $\hat{P}$ to shifts in the control moment $E_0[h_k]$:
     $$\frac{d\hat{P}}{d E_0[h_k]} = -2.04 \quad (\text{per tau side}).$$
  3. We evaluated the true unpolarised first moments $E_0[h_k]$ directly on $H$ vs $Z$:
     - On $H$: $E_0[h_{-,k}] = +0.0486$, $E_0[h_{+,k}] = +0.0502$ (mean $+0.0494$).
     - On $Z$: $E_0[h_{-,k}] = +0.0373$, $E_0[h_{+,k}] = +0.0408$ (mean $+0.0390$).
     - The intrinsic transport difference is $\Delta E_0[h_k] = +0.0104$ per side ($+0.0208$ in the sum $u$).
     - Direct multiplication yields the uncorrected bias: $(-2.04) \times (+0.0208) = -0.0424$, exactly matching the observed uncorrected bias.
  4. We evaluated the reweighting performance: even after 6D binned reweighting on $(p_{T,\text{vis},1}, p_{T,\text{vis},2}, \eta_1, \eta_2, \text{mode}_1, \text{mode}_2)$, the residual bias is $\Delta P = -0.0180$.
  5. The statistical precision $\sigma(P_\tau) = 0.00546$ per $10^5$ events scales as $0.00546 / \sqrt{N / 10^5}$. The crossover point where the systematic bias equals the statistical uncertainty is:
     $$N_{\text{crossover}} = 10^5 \times \left(\frac{0.00546}{0.0180}\right)^2 \approx 9.2 \times 10^3 \text{ events!}$$

* **What was found:**
  - The measurement is **systematics-dominated already at $9,200$ events**.
  - To achieve an unbiased measurement at $10^5$ events (where $\sigma_{\text{stat}} \approx 0.0055$), the unpolarised acceptance moment $E_0[h_k]$ must be known to better than $\Delta E_0[h_k] < 0.00546 / 2.04 \approx 1.66 \times 10^{-3}$ per side.
  - At $10^7$ events (LHC Run 3 scale), $E_0[h_k]$ must be controlled to $1.66 \times 10^{-4}$ ($0.016\%$).
  - Transporting $p_0$ from $H \to \tau\tau$ to $Z \to \tau\tau$ fails because:
    1. $H$ and $Z$ have different parent masses ($125$ GeV vs $91.2$ GeV), leading to different tau boost distributions ($\gamma_\tau$). Because $E_0[h_k] \sim p_{\text{cut}} / p_{T,\tau}$, softer taus suffer greater truncation.
    2. Different production modes ($gg \to H$ vs $q\bar{q} \to Z$) yield different jet recoils and $\eta$ distributions.
    3. Reweighting on visible $p_T$ is intrinsically flawed because $p_{T,\text{vis}} \approx \frac{1+h_k}{2} p_{T,\tau}$ is directly coupled to the decay polarimeter angle $h_k$, distorting the true $p_{T,\tau}$ spectrum.

* **What should change:**
  - The manuscript / README must explicitly lead with the statement that **this measurement cannot rely on $H \to \tau\tau$ as an external data-driven control region** to reach precision EW goals.
  - The $H \to Z$ exercise should be presented as a *demonstration of the magnitude of acceptance systematics*, not as a viable experimental measurement pipeline.
  - Point out that in an actual experimental analysis, $p_0(Z)$ must be calibrated from unpolarised Monte Carlo with full detector simulation, constrained by auxiliary measurements of the $Z$ boson differential cross-section ($p_T^Z, y^Z$) and in-situ validation on $Z \to \mu\mu / ee$.

---

### Finding 2 [MAJOR]: Confounded Cross-Run Comparison ($1.22$ vs $2.71$ vs $2.73$)

* **What is claimed:**
  Reconstruction degrades polarisation sensitivity by only a factor of $1.22$ (exact $0.00451 \to$ reco $0.00552$), compared to $2.71$ for the CP-mixing angle $\phi_\tau$ and $2.73$ for the entanglement witness $\langle W \rangle$. Geometry (IP+SV) buys only $1.01$ for polarisation versus $1.25$ for CP and $1.20$ for entanglement. The stated reason is that polarisation lives in the longitudinal component $h_k$.

* **What was checked:**
  1. We audited the three runs to identify the underlying sample and observable definitions:
     - **CP Mixing (2026-09-25):** Evaluated on **$H \to \tau\tau$ validation rows** ($m_{\tau\tau} = 125$ GeV); observable is the transverse phase $\phi^* = \arctan(h_r / h_n)$ or $C_{nr} - C_{rn}$.
     - **Entanglement (2026-09-26):** Evaluated on **$H \to \tau\tau$ validation rows** ($m_{\tau\tau} = 125$ GeV); observable is the trace/magnitude of correlation matrix $C$.
     - **Polarisation (2026-09-27):** Evaluated on **$Z \to \tau\tau$ validation rows** ($m_{\tau\tau} = 91.2$ GeV); observable is the longitudinal first moment $B_k = -P_\tau \hat{k}$ ($h_{-,k} + h_{+,k}$).
  2. To separate the **observable dimension effect** ($h_k$ vs $h_T$) from the **sample kinematic effect** ($H$ vs $Z$), we evaluated the per-component reconstruction MSE and correlation $\text{Corr}(h_{\text{pred}}, h_{\text{true}})$ on both samples under identical conditions:

| Sample | Quantity | Base Arm (No Geom) | IP + SV Arm | Ideal IP Oracle | IP+SV Gain |
|---|---|---|---|---|---|
| **$H \to \tau\tau$** | $\text{MSE}(h_k)$ (longitudinal) | $0.1311$ | $0.1252$ | $0.1174$ | $+0.006$ (MSE) |
| | $\text{Corr}(h_k)$ | $0.7627$ | $0.7750$ | $0.7891$ | $\mathbf{1.016\times}$ |
| | $\text{MSE}(h_T)$ (transverse) | $0.2156$ | $0.1839$ | $0.1350$ | $+0.032$ (MSE) |
| | $\text{Corr}(h_T)$ | $0.6125$ | $0.6842$ | $0.7648$ | $\mathbf{1.117\times}$ |
| **$Z \to \tau\tau$** | $\text{MSE}(h_k)$ (longitudinal) | $0.1288$ | $0.1238$ | $0.1165$ | $+0.005$ (MSE) |
| | $\text{Corr}(h_k)$ | $0.7670$ | $0.7771$ | $0.7904$ | $\mathbf{1.013\times}$ |
| | $\text{MSE}(h_T)$ (transverse) | $0.2260$ | $0.1925$ | $0.1412$ | $+0.033$ (MSE) |
| | $\text{Corr}(h_T)$ | $0.5907$ | $0.6678$ | $0.7512$ | $\mathbf{1.131\times}$ |

* **What was found:**
  - The physical conclusion that *longitudinal reconstruction is vastly superior to transverse reconstruction* is completely true: $\text{Corr}(h_k) \approx 0.77$ without geometry, whereas $\text{Corr}(h_T) \approx 0.59$. Adding IP+SV improves $h_k$ correlation by only $0.010$ ($1.01\times$), while improving $h_T$ correlation by $0.077$ ($1.13\times$).
  - However, quoting the headline numbers $1.22$ vs $2.71$ vs $2.73$ directly compares three different mathematical functionals on two different event samples ($Z$ vs $H$).
  - Specifically, CP mixing is a 2D angle in the transverse plane, whose variance scales as $1/\text{Corr}(h_T)^2$, compounding resolution degradation, whereas polarization is a 1D linear sum along the boost axis.

* **What should change:**
  - Clarify in Section 2 and Table 3 that the ratio $1.22$ vs $2.71/2.73$ reflects the fundamental physics difference between **longitudinal 1D projection ($h_k$)** and **transverse 2D correlation ($h_T$)**, and provide the unified per-component correlation table shown above to make the comparison fully rigorous.

---

### Finding 3 [MAJOR]: Overstated Experimental Context Regarding LEP Equivalence

* **What is claimed:**
  `q5_translate.py` and `README.md` state that $\sigma(P_\tau) = 0.00546$ per $10^5$ events matches the combined LEP precision $\sigma(\mathcal{A}_\tau) = 0.0043$ at $1.61 \times 10^5$ selected $Z$ events, mapping to $\sigma(\sin^2\theta_{\text{eff}}) = 6.9 \times 10^{-4}$ ($5.5 \times 10^{-4}$ at LEP).

* **What was checked:**
  1. We verified the conversion algebra:
     $$v = -1/2 + 2\sin^2\theta_W = -0.03696, \quad a = -0.5$$
     $$\mathcal{A}_\tau = -P_\tau = \frac{2va}{v^2+a^2} = 0.147037$$
     $$\left|\frac{dP_\tau}{d\sin^2\theta_W}\right| = 2 \left|\frac{dv}{d\sin^2\theta_W}\right| \frac{a(a^2 - v^2)}{(v^2+a^2)^2} = 2 \times 3.9350 = 7.8700$$
     $$\sigma(\sin^2\theta_{\text{eff}}) = \frac{\sigma(P_\tau)}{7.8700} = \frac{0.005462}{7.870} = 6.94 \times 10^{-4}.$$
  2. The LEP baseline: LEP combined $\mathcal{A}_\tau = 0.1439 \pm 0.0043$ ($0.0043 / 7.870 = 5.46 \times 10^{-4}$). The event calculation $N_{\text{LEP}} = 10^5 \times (0.005462 / 0.0043)^2 = 1.61 \times 10^5$ is arithmetically exact for signal-only statistics.
  3. We evaluated real-world hadron collider conditions (LHC ATLAS/CMS):
     - At LEP ($e^+e^- \to Z \to \tau^+\tau^-$), events were recorded at $\sqrt{s} \approx 91.2$ GeV with known initial state, zero pileup, negligible background ($<1\%$), and known beam energy.
     - At LHC ($pp \to Z \to \tau_h\tau_h$), di-tau triggers require high $p_T$ thresholds ($p_T > 35-40$ GeV or asymmetric $p_T > 35/25$ GeV), hadronic tau selection efficiency $\times$ acceptance is only $\sim 2-5\%$, and QCD multijet background is massive ($10-30\%$ in signal regions).
     - Fake taus from quark/gluon jets have asymmetric polarimeter distributions that directly bias $\langle h_k \rangle$.
     - Tau Energy Scale (TES) uncertainty of $\pm 0.5-1.0\%$ shifts the visible energy fraction $x$, directly shifting $E_0[h_k]$ by $\sim \mathcal{O}(10^{-2})$, which is $10\times$ larger than the statistical target ($1.7 \times 10^{-3}$).

* **What was found:**
  - While the script docstring mentions "signal events only", the headline claim "LEP precision matched at $1.6 \times 10^5$ events" gives a misleading impression of feasibility.
  - The measurement at a hadron collider is completely dominated by systematic uncertainties (acceptance modeling, TES, jet fakes), and statistical equivalence at $1.6 \times 10^5$ events is practically irrelevant given that transport systematics saturate at $9.2 \times 10^3$ events.

* **What should change:**
  - Upgrade the caveat to a prominent warning: clarify that the $1.6 \times 10^5$ event figure is a *pure signal statistical benchmark*, and that experimental systematic uncertainties (TES, background subtraction, jet-to-tau fakes) will dictate the precision at the LHC.

---

### Finding 4 [MODERATE]: Scope and Interpretation of Disjoint-Half Closure Tests

* **What is claimed:**
  `q4_transport.py` tests closure by splitting $H$ into disjoint halves, using half $A$ to obtain $p_0$ and measuring half $B$, recovering $P = +0.0020 \pm 0.0118$ (truth $0.0$). On $Z$ disjoint halves, it recovers $P = -0.1508 \pm 0.0131$ (truth $-0.1470$).

* **What was checked:**
  1. We tested the bootstrap procedure and importance weighting stability:
     - Minimum weight denominator $f_H^{\text{min}} = 0.0145$, maximum importance weight $w_H^{\text{max}} = 68.78$.
     - Maximum event weight fraction is only $0.235\%$, and ESS fraction is $40.7\%$.
  2. We verified why the 2026-09-26 entanglement run found its own self-calibration circular:
     - In the entanglement run, when the *same* sample was used for calibration and measurement with $1/f(\theta_{\text{assumed}})$, it algebraically returned $\theta_{\text{assumed}}$ by identity.
     - In `q4_transport.py`, half $A$ and half $B$ are statistically disjoint sets of events.

* **What was found:**
  - The disjoint-half test successfully validates that the empirical estimator $\text{invert\_mean}(m_A, \bar{T}_B)$ is statistically unbiased and numerically stable on i.i.d. events.
  - **However**, because half $A$ and half $B$ share the exact same underlying MC process ($pp \to H+j$), it does **not** test physical transport across different processes ($H \to Z$).
  - Readers might conflate "disjoint-half closure" with "validation of the control region methodology".

* **What should change:**
  - Explicitly emphasize in `q4_transport.py` and README that disjoint-half closure is a test of *estimator consistency under i.i.d. sampling*, not a validation of $H \to Z$ transport invariance.

---

### Finding 5 [VERIFIED SOUND]: Polarisation Signs, Basis Conventions, and $Z-H$ First Moment Shift

* **What is claimed:**
  - The physical polarimeter satisfies $d\Gamma \propto 1 + \vec{h}_{\text{phys}} \cdot \vec{s}$.
  - The canonical polarimeter is $\vec{h}_{\text{can}} = -\vec{h}_{\text{phys}}$ on both sides.
  - In the $(n, r, k)$ basis where $\hat{k} = \hat{p}_{\tau^+}$ in the ditau CM frame, an unpolarised $Z$ yields $B_{\text{can}} = -P_\tau \hat{k}$ with $P_\tau = -0.147037$, giving ditau density:
    $$f_Z = 1 + P_\tau (h_{-,k} + h_{+,k}) + h_-^T C_Z h_+.$$
  - The measured difference $\langle h_k \rangle_Z - \langle h_k \rangle_H = -0.0632$ confirms the negative sign of $P_\tau$.

* **What was checked:**
  1. **Spin amplitudes & Helicity:**
     - Left-handed $\tau^-$ ($h = -1$) has momentum $-\hat{k} \implies \vec{s}_{\tau^-} = +\hat{k} \implies \langle \vec{s}_{\tau^-} \rangle = -P_\tau \hat{k} = +0.147037 \hat{k}$.
     - Right-handed $\tau^+$ ($h = +1$) has momentum $+\hat{k} \implies \vec{s}_{\tau^+} = +\hat{k} \implies \langle \vec{s}_{\tau^+} \rangle = -P_\tau \hat{k} = +0.147037 \hat{k}$.
     - Thus both spin vectors point along $+\hat{k}$ with magnitude $-P_\tau = +0.147037$.
  2. **Polarimeter sign flip:**
     - For $\tau^- \to \pi^- \nu_\tau$, $\vec{h}_{\text{phys}} = +\hat{p}_\pi$.
     - Because $\vec{h}_{\text{can}} = -\vec{h}_{\text{phys}}$, the term $\vec{B}_{\text{phys}} \cdot \vec{h}_{\text{phys}} = (-P_\tau \hat{k}) \cdot (-\vec{h}_{\text{can}}) = +P_\tau h_{\text{can}, k}$.
     - Summing both sides gives $+P_\tau (h_{-,k} + h_{+,k})$.
  3. **Observed $Z - H$ moment shift:**
     - On $H$: $\langle h_{-,k} + h_{+,k} \rangle = +0.07465$.
     - On $Z$: $\langle h_{-,k} + h_{+,k} \rangle = +0.01145$.
     - Difference: $\Delta \langle u \rangle = -0.06321$.
     - If $P_\tau$ had the opposite sign ($+0.147$), the observed sum on $Z$ would have shifted to $\approx +0.173$. The negative shift $-0.0632$ directly confirms the negative sign of $P_\tau$.
  4. Injected $P$ closure: injecting $P \in \{-0.30, -0.20, -0.147, -0.05, 0.0, +0.15\}$ in `q3_measure.py` yields exact unit linear slope ($d\hat{P}/dP_{\text{inj}} = 1.0000$).

* **What was found:**
  - All signs, conventions, Dirac traces, and empirical moment shifts are **100% physically and mathematically correct**.
  - Note on nomenclature: `README.md` line 20 states $B_{\text{can}} = -P_\tau \hat{k}$, but if written as $f = 1 + \vec{B}_{\text{can}} \cdot \vec{h}_{\text{can}}$, then $\vec{B}_{\text{can}} = +P_\tau \hat{k} = -0.147 \hat{k}$. The actual functional form $f = 1 + P(h_{-,k} + h_{+,k}) + \dots$ used in code is completely unambiguous.

---

### Finding 6 [VERIFIED SOUND]: Why the Neural Network Beats $E_{\text{vis}}/E_\tau$ ($x_{\text{truth}}$)

* **What is claimed:**
  The neural network $h_{\text{pred}, k}$ achieves $\sigma(P_\tau) = 0.00552$ (no geometry) and $0.00546$ (with IP+SV), significantly outperforming the classical LEP observable $x_{\text{truth}} = E_{\text{vis}} / E_\tau$ evaluated with the *exact truth tau energy* ($\sigma = 0.00903$).

* **What was checked:**
  1. We verified the four-vector structure of `ladder_validation.npz`:
     - Arrays are structured as $(p_x, p_y, p_z, E)$ with index $3$ being Energy $E$.
     - Neutrino invariant mass $E^2 - |\vec{p}|^2 = 0.0$ and visible tau mass $E^2 - |\vec{p}|^2 = m_{\text{vis}}^2 \ge m_\pi^2$.
     - $x_{\text{truth}} = E_{\text{vis}} / (E_{\text{vis}} + E_\nu)$ is properly formed.
  2. We audited the mode-by-mode sensitivity in `q2_classical.json`:

| Truth Mode | Fraction | $\sigma(P_\tau)$ from $x_{\text{truth}}$ | $\sigma(P_\tau)$ from Network (`full22`) | $\sigma(P_\tau)$ from Exact $h_k$ |
|---|---|---|---|---|
| **$\pi \nu$** | $23.4\%$ | $\mathbf{0.00906}$ (Response $+0.124$) | $\mathbf{0.00962}$ (Response $+0.114$) | $\mathbf{0.00905}$ (Response $+0.125$) |
| **$\rho \nu \to \pi^\pm\pi^0\nu$** | $56.7\%$ | $\mathbf{0.01474}$ (Response $+0.110$) | $\mathbf{0.00706}$ (Response $+0.224$) | $\mathbf{0.00559}$ (Response $+0.373$) |
| **$a_1 \nu \to 3\pi\nu$** | $19.9\%$ | $\mathbf{0.21395}$ (Response $-0.005$) | $\mathbf{0.01227}$ (Response $+0.074$) | $\mathbf{0.00891}$ (Response $+0.133$) |
| **Combined** | $100\%$ | $\mathbf{0.00903}$ | $\mathbf{0.00546}$ | $\mathbf{0.00451}$ |

* **What was found:**
  - This is **not a bug**; it is a fundamental consequence of hadronic weak decays:
    1. For $\tau \to \pi\nu$ (spin-0 pion), $x$ captures the full spin analyzing power ($\alpha_\pi = 1.0$), and $x_{\text{truth}}$ matches exact $h_k$ perfectly ($0.00906$ vs $0.00905$).
    2. For $\tau \to \rho\nu$ (spin-1 vector meson), the energy fraction $x_\rho$ alone has analyzing power $\alpha_\rho(x) = \frac{m_\tau^2 - 2m_\rho^2}{m_\tau^2 + 2m_\rho^2} \approx 0.45$. The full polarimeter incorporates the internal decay angle $\Upsilon = (p_{T,\pi^\pm} - p_{T,\pi^0}) / p_{T,\rho}$, raising the analyzing power to $\sim 0.65$. The network learns this internal decay asymmetry.
    3. For $\tau \to a_1\nu \to 3\pi\nu$, $m_\tau^2 - 2m_{a_1}^2 \approx 1.777^2 - 2(1.26)^2 \approx 0.0$. The energy fraction $x_{a_1}$ has **zero analyzing power** ($\sigma = 0.214$, response $\approx 0$). All polarization sensitivity resides in the 3-pion Dalitz/spatial current $J^\mu$. The neural network reconstructs this internal structure from the visible pions, achieving $\sigma = 0.01227$.
  - Combining all modes, the network leverages the $76.6\%$ branching fraction of $\rho$ and $a_1$ modes that $x_{\text{truth}}$ squanders, easily beating $x_{\text{truth}}$ overall.

---

### Finding 7 [VERIFIED SOUND]: One-Parameter Acceptance Mechanism ($E_0[h_k]_\pi = p_{\text{cut}} / p_{T,\tau}$)

* **What is claimed:**
  For $\tau \to \pi\nu$, the visible transverse momentum cut $p_{T,\text{vis}} > p_{\text{cut}}$ induces an unpolarised expectation $E_0[h_k] = p_{\text{cut}} / p_{T,\tau}$. A one-parameter fit yields $p_{\text{cut}} = 18.8$ GeV with maximum residual $0.035$.

* **What was checked:**
  1. **Analytical derivation:**
     In the collinear limit, $p_{T,\pi} = x p_{T,\tau}$ with $x = (1 + h_k)/2$.
     Under $p_0$, $h_k \sim \text{Uniform}[-1, 1] \implies x \sim \text{Uniform}[0, 1]$.
     The selection $p_{T,\pi} > p_{\text{cut}}$ imposes $x > x_{\text{min}} = p_{\text{cut}} / p_{T,\tau}$.
     The conditional expectation is:
     $$E[x \mid x > x_{\text{min}}] = \frac{1 + x_{\text{min}}}{2} \implies E_0[h_k] = 2 E[x] - 1 = x_{\text{min}} = \frac{p_{\text{cut}}}{p_{T,\tau}}.$$
  2. We verified the numerical fit across truth $p_{T,\tau}$ bins:
     - At $p_{T,\tau} = 26.3$ GeV: measured $+0.715$, model $+0.725$ (residual $-0.010$).
     - At $p_{T,\tau} = 45.5$ GeV: measured $+0.411$, model $+0.415$ (residual $-0.005$).
     - At $p_{T,\tau} = 109.9$ GeV: measured $+0.177$, model $+0.172$ (residual $+0.005$).
     - At $p_{T,\tau} = 255.9$ GeV: measured $+0.077$, model $+0.076$ (residual $+0.001$).
     - Fitted $p_{\text{cut}} = 18.83$ GeV closely matches the ntuple selection threshold (1st percentile reco $p_T = 20.71$ GeV).

* **What was found:**
  - The derivation is elegant, transparent, and confirmed by data. It rigorously explains why the unpolarised first moment is positive ($+0.174$ for $\pi$) and why softer taus exhibit larger acceptance bias.

---

## Summary of Actionable Recommendations

1. **Refocus the Narrative on Systematics:**
   Clearly state in `README.md` and paper conclusions that measuring tau polarisation at hadron colliders is an **acceptance-systematics limited measurement** ($d\hat{P}/dE_0[h_k] = -2.04$), not a statistics-limited one.
2. **De-emphasize $H \to Z$ as a Solution:**
   Present the $H \to \tau\tau$ control region as an illustration of the severe challenge of kinematic transport, rather than an unbiased experimental calibration method.
3. **Refine Cross-Run Benchmark ($1.22$ vs $2.71$):**
   Explicitly decompose the reconstruction gains into **longitudinal ($h_k$) vs transverse ($h_T$)** components to avoid an apples-to-oranges comparison across different physics observables.
4. **Clarify Hadron Collider Caveats:**
   Explicitly document that matching the LEP statistical threshold ($1.6 \times 10^5$ events) does not translate to matching LEP electroweak precision due to QCD fake backgrounds, trigger thresholds, and Tau Energy Scale systematics.
