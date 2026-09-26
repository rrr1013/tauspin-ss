**Verdict: revise — the shape-only sensitivity ladder is mostly reproducible, but the written sign convention is internally inconsistent and the H-to-Z transport study does not establish the quoted 9.2k-event systematic limit.**

## 1. High — the corrected transport bias and “systematics limited at 9.2k events” conclusion are not established

**What is claimed.** `control_moments` is said to give an unbiased estimator, H-to-Z kinematic reweighting is said to reduce the bias from `-0.0425` to `-0.0180`, and that residual is converted into a crossover at $9.2\times10^3$ selected Z events.

**What I checked.** I rederived the joint-moment formula, inspected the $1/f_H$ weights, reproduced all central values, measured histogram support, and bootstrapped both samples while refitting the kinematic density ratio in every replica.

**What I found.** The algebra in [`control_moments` / `invert_mean`](../pol_tools.py) is correct. For a common joint spin-independent measure $p_0(h,T,\ldots)$,

\[
E_P[T]=\frac{E_0[T]+P E_0[Tu]+E_0[Tv]}{1+P E_0[u]+E_0[v]}
\]

is exact and assumes no factorisation between the two tau sides. Self-normalised $1/f$ importance weighting is consistent if the control has support everywhere needed. That is not the weak point.

The numerical transport conclusion is much weaker than stated:

- The uncorrected result is $\hat P=-0.18948$, bias `-0.04245`, with the run's own joint bootstrap SD `0.02069`: only about $2.05\sigma$ from zero transport bias.
- On validation H, the $1/f_H$ ESS fraction is `0.407`, but the lowest 1% of events contribute `52.6%` of the sum of squared weights. On the independent train artifact the ESS is `0.270`, $f_{\min}=0.00233$, the maximum weight is `430`, and one event contributes `21.2%` of the squared-weight sum. ESS `0.41` is therefore not by itself a stability certificate.
- The advertised `pT_vis + eta + mode` correction is a sparse six-dimensional histogram: $8^2\times3^2\times6^2=20{,}736$ possible cells. Of 7,624 cells occupied by Z, only 6,108 are also occupied by H; `8.38%` of the Z target count is outside H support. The calculation silently assigns zero ratio weight there.
- The code uses **truth** decay modes (`modes`), not reconstructed decay modes, for this correction. Replacing truth mode by the available reconstructed mode changes the same-data central bias from `-0.0180` to `-0.0041`. This large analysis-choice dependence is not propagated.
- In a 250-replica bootstrap that refits the histogram ratio each time, the six-dimensional truth-mode result has mean $\hat P=-0.1518$ and SD `0.0280`; the published single-fit value is `-0.1650`. The residual `-0.018` is smaller than that uncertainty. A disjoint fit/apply split is still less stable because sparse-bin support collapses. The plot in `fig5_systematic.png` has no uncertainty bars for these reweighted points.

The H→H split is a useful implementation and finite-sample check, but it is not a transport validation: both halves have the same $p_0$. The Z→Z version additionally unfolds with the true $P_0$, so it is partly tautological. Forty overlapping random half-splits do not establish unbiased H→Z transport.

**What should change.** Keep the estimator algebra, but retract the corrected `-0.018`, the `9.2e3` crossover, and the factor-4.2 “systematics limited” statement as quantitative conclusions. Call the observed H/Z difference a roughly $2\sigma$, sample-specific transport warning. A defensible correction needs an independent or cross-fitted smooth density-ratio model using data-available variables, explicit support/overlap diagnostics, and a joint bootstrap that refits the ratio and recalculates every moment. The calibration-sample uncertainty and the $1/f$ tail must be included. Only then should a crossover event count be quoted.

## 2. High — $B_{\rm canonical}=-P_\tau\hat k$ is the wrong label, although the estimator code uses the right sign

**What is claimed.** The README calls the trace result $B^-_{\rm can}=B^+_{\rm can}=-P_\tau\hat k=+0.147\hat k$, while also writing the stored-$h$ density as $f=1+P_\tau(h^-_k+h^+_k)+\cdots$.

**What I checked.** I reran the explicit Dirac trace, followed the basis definition $\hat k=\hat p_{\tau^+}$, and applied the documented relation $h_{\rm can}=-h_{\rm phys}$ on both sides.

**What I found.** The clean convention chain is

\[
P_\tau\equiv -A_\tau=-\frac{2va}{v^2+a^2}=-0.1470366,
\]
\[
B^-_{\rm phys}=B^+_{\rm phys}=+0.1471465\,\hat k\simeq -P_\tau\hat k,
\qquad
B^-_{\rm can}=B^+_{\rm can}=-B_{\rm phys}\simeq P_\tau\hat k.
\]

Both physical-spin coefficients have the same sign in this **common** basis. The tau-minus momentum is $-\hat k$, so a negative tau-minus helicity polarization gives a spin component along $+\hat k$; the CP-conjugate tau-plus also has spin along $+\hat k$. The comment `B- = -B+` in [`z_density.py`](../../entanglement/z_density.py) is therefore inconsistent with its own trace output.

The actual pricing code in [`pol_tools.py`](../pol_tools.py) is correct: it uses $f=1+P_\tau(h^-_{k,\rm can}+h^+_{k,\rm can})+\cdots$. Thus the reported negative-$P_\tau$ sensitivities do not need a numerical sign flip. The finite-mass difference between `0.1471465` from the trace and `0.1470366` from the massless asymmetry formula is $1.1\times10^{-4}$, negligible here but worth labelling.

The raw Z–H moment difference `-0.0632` is consistent with the negative coefficient. Using the H-derived moments, the target-Z mean is predicted as `0.0378` for $P=-0.147$, `0.1274` for $P=0$, and `0.2144` for $P=+0.147$, versus the observed `0.0114`. But the raw difference alone is not a convention proof, because the H and Z acceptance measures demonstrably differ; an offset could partly mask it.

**What should change.** Rename the trace output as $B_{\rm phys}$, set $B_{\rm can}=P_\tau\hat k$, correct the `z_density.py` docstring, and remove the misleading JSON key `B_Z_canonical_sign`. State separately that $P_\tau$ is the tau-minus helicity polarization, while both spin coefficients are equal in the common $(n,r,k)$ basis.

## 3. Medium — the 1.22 vs 2.71 vs 2.73 arithmetic is right, but it is not an apples-to-apples performance ratio

**What is claimed.** Reconstruction costs `1.22` for polarization versus `2.71` for the CP angle and `2.73` for the entanglement witness; geometry buys `1.01`, `1.25`, and `1.20`, respectively, because polarization is longitudinal.

**What I checked.** I traced every number to its source estimator and cohort. The arithmetic is correct: `0.0055209/0.0045149=1.2228`, `1.604/0.591=2.714`, and $\sqrt{344/46}=2.735$.

**What I found.** These are three conditional degradation factors, not a controlled comparison of information loss. CP and entanglement use H rows, polarization uses Z rows; the scores and observables are different (CP triple product, a calibrated witness/event-count reach, and a first moment); the accepted $p_0$, mode composition, and spin state differ. The entanglement event-count number also has the finite-$N$ and fixed-calibration qualifications documented in its independent review. Therefore the exact numerical contrast cannot by itself be attributed to “longitudinal versus transverse.”

The qualitative mechanism is nevertheless supported. Separating samples, the no-geometry component correlations are $h_k:0.763/0.767$ for H/Z, compared with $h_n:0.632/0.609$ and $h_r:0.594/0.572$. IP/SV moves $h_k$ only to `0.775/0.777`, while it moves the transverse components to roughly `0.69/0.68` (H) and `0.68/0.66` (Z). Conditional on the fixed trained artifacts, the same-seed base/full polarization gain is `1.0098 ± 0.0012` in a paired event bootstrap; training-seed/model-selection uncertainty is not included.

**What should change.** Present `2.71/2.73/1.22` as a descriptive cross-run comparison, not a causal measurement. To make it safe, build all three parameter scores on one common event/p0 population, use the same rows, mode mix, exact/current definition, train/test protocol and readout class, and evaluate matched reco/exact and geometry/no-geometry ratios with paired uncertainty. The component-correlation result can remain as the main evidence for the longitudinal interpretation.

## 4. Medium — linear-response pricing is legitimate for shape only; exact-score cross-fitting is not leakage, but its uncertainty is understated

**What is claimed.** $\sigma(P)=\sqrt{\operatorname{Var}T/N}/|\operatorname{Cov}(T,S)|$ fairly compares all observables, including cross-fitted visible baselines trained against the exact-$h$ score. Rate information is intentionally discarded.

**What I checked.** I differentiated the normalized selected distribution, compared covariance responses to finite-difference reweighting, and repeated the two-fold readout with many fold assignments.

**What I found.** For fixed selected event count,

\[
\frac{dE[T]}{dP}=\operatorname{Cov}(T,S),\qquad
I_{\rm shape}=\operatorname{Var}(S)
\]

is correct even though the raw score has nonzero mean. The missing normalization derivative is a constant and disappears from covariance/variance. For $T=h^-_k+h^+_k$, the covariance response is `0.630272`; finite differences from $10^{-4}$ through `0.05` give `0.630251–0.630261`. The quoted `0.00438` is therefore a valid **shape-only, exact-$h$** Cramér–Rao value, not the full selected-rate likelihood and not a reco-level CR bound. Here the dropped rate term is small: $E[S]^2=0.00628$ versus shape information `0.52199`.

Training a readout against the exact score is supervised simulation calibration, but held-out cross-fitting prevents same-event target leakage. It does not flatter the network: it directly optimizes the handcrafted baseline for this parameter while the network was trained generically to predict $h$, so the network comparison is, if anything, conservative. “Best visible” still means best in the listed low-order feature class, not an information-theoretic best visible observable.

The quoted readout uncertainty omits fitting/fold randomness. Across 40 two-fold partitions, visible-only gives mean `0.00747`, SD `0.00051`, range `0.00704–0.00907`; visible+MET gives mean `0.00694`, SD `0.00031`, range `0.00665–0.00778`. The published seed-0 values (`0.00741`, `0.00666`) are within those ranges, but the displayed bootstrap errors (`5.3e-5`, `3.9e-5`) resample frozen cross-fitted predictions and are much too small for the full procedure. Five- to ten-fold fits are more stable (`~0.00709` and `~0.00668`). The qualitative ordering behind the network advantage remains.

**What should change.** Keep the linear-response formula and explicit “shape only” label. Refit the readout inside the bootstrap or quote repeated-fold variability, use more stable folds/regularisation fixed in advance, and replace “best visible” by “best cross-fitted linear readout of this feature set.”

## 5. Medium — `x_truth` has the correct four-vector layout, but $h_k=2x-1$ is not exact in this implementation

**What is claimed.** For $\tau\to\pi\nu$, exact $h_k$ equals $2E_{\rm vis}/E_\tau-1$ exactly; the equality of sensitivities is called an implementation closure. The network beating truth-$E_\tau$ energy fraction is presented as surprising.

**What I checked.** I inspected [`pol_visible.py`](../pol_visible.py) and [`q2_classical.py`](../q2_classical.py), then recomputed tau masses, energy fractions and eventwise residuals from `ladder_validation.npz`.

**What I found.** There is no `(E,px,py,pz)` bug. The ntuple is `(px,py,pz,E)`, `V.E=3`, reconstructed truth tau masses are `1.77681–1.77704 GeV`, and all $x$ lie in `[0.04899, 0.99997]`.

However, on the 28,091 pi sides,

\[
h_k-(2x_{\rm truth}-1)
\]

has mean `-0.00593`, RMS `0.02964`, and maximum absolute deviation `0.1719`. The relation is exact only in the appropriate ultra-relativistic/collinear frame convention; the code uses a **lab-energy** fraction while $h_k$ is defined with the tau-pair-rest-frame axis. Finite tau/pion masses and the boosted pair prevent eventwise equality. The two sensitivities (`0.009047` and `0.009059`) agree to `0.14%`, so the numerical near-closure is real, but “exact” is false.

It is not suspicious that the network beats this one-dimensional truth-energy scalar over all modes. For rho and a1, the full polarimeter contains internal angular and invariant-mass information that $E_{\rm vis}/E_\tau$ discards; the measured a1 response `-0.00485` is indeed consistent with nearly zero. The comparison is not against the optimal LEP likelihood using all mode-specific decay variables.

**What should change.** Call this a high-boost numerical closure, give the exact finite-boost relation or quote the residual above, and relabel the last ladder row as a truth-$E_\tau$ **single-variable baseline**, not a universal “LEP-style” ceiling. Then the network result is physically expected rather than a bug.

## 6. Medium — the LEP number and local derivative are correct, but the event-count comparison is not an LHC reach estimate

**What is claimed.** $\sigma(\sin^2\theta_{\rm eff})=\sigma(P)/7.87$; the LEP $A_\tau=0.1439\pm0.0043$ precision is matched statistically at $1.6\times10^5$ selected Z events.

**What I checked.** I differentiated the exact asymmetry formula and checked the LEP combination and a modern hadron-collider polarization treatment.

**What I found.** The numerical inputs are right. At `sin²θ=0.23152`, $dP/d\sin^2\theta=7.87005$, so `0.0054586/7.87005=6.94e-4`. The LEP combination is $A_\tau=0.1439\pm0.0043$, and CMS explicitly uses $P_\tau(Z)=-A_\tau$ at the pole ([LEP Electroweak Working Group report](https://cds.cern.ch/record/892831/files/phep-2005-041.pdf), [CMS tau-polarization paper](https://arxiv.org/abs/2309.12408)). The arithmetic $10^5(0.0054586/0.0043)^2=1.61\times10^5$ is correct.

But `0.0043` is the **total** LEP uncertainty; the LEP table separates about `0.0035` statistical and `0.0026` systematic. Matching LEP's statistical component would require about $2.43\times10^5$ events under this idealized scaling. Moreover, this run uses selected signal-only hadronic events, fixed event count, a development validation sample, and the v2 unpolarized-Z construction. It omits backgrounds, calibration statistics, trigger/ID, migrations and energy scales, the physical Drell–Yan mass/quark mixture, photon exchange/interference, and the pole correction. CMS's treatment explicitly averages over mass and quark type before translating to the pole.

The README and figure axis do say “signal only, shape only,” so the caveat is present. It is not strong enough to support wording such as “reaches LEP at 1.6e5” or the downstream factor-4.2 systematics conclusion.

**What should change.** Say “the idealized signal-only shape uncertainty equals the numerical **total LEP uncertainty** at $1.61\times10^5$ selected events.” Explicitly state that this is not an event-yield or luminosity projection. Keep `6.9e-4` as a conditional per-selected-signal-event result; defer a reach or systematics-limited claim to full-ME Z/γ*, backgrounds, nuisance parameters, and independent calibration.

## 7. Low — the $p_{\rm cut}/p_T^\tau$ model is a useful description, not yet a validated acceptance law

**What is claimed.** A one-parameter model $E_0[h_k]_{\pi}=p_{\rm cut}/p_T^\tau$, with $p_{\rm cut}=18.8$ GeV, reproduces the measured dependence with worst residual `0.035`; $E_0[h_k]$ must be known to `1.7e-3` at $10^5$ events.

**What I checked.** I reproduced the fit and its residuals and inspected the perturbation used to obtain the requirement.

**What I found.** The stated fitted value and maximum binned residual are correct. The $1/p_T^\tau$ trend is persuasive evidence that soft-tau selection drives much of the pi offset. But `18.8 GeV` is an effective parameter fitted and evaluated on the same truth-$p_T^\tau$ sample, not a measured detector threshold; no fit uncertainty, goodness-of-fit or holdout test is supplied. The derivation also inherits the approximate, not exact, $h_k=2x-1$ relation and neglects eta, the opposite side and the rest of the event selection.

The `1.7e-3` arithmetic is correct only for the local perturbation implemented in [`q4_transport.py`](../q4_transport.py): shift $E_0[u]$ while holding $E_0[u^2]$, $E_0[uv]$, $E_0[v]$, and the reco response exact. It is a requirement on one nuisance component, not “the acceptance” in general. Attributing the observed full transport difference to the first moment gives a different coefficient (about `-4.09` instead of `-3.29`) and a requirement near `1.34e-3`, illustrating the model dependence.

**What should change.** Retain the model as a descriptive one-parameter diagnostic, add uncertainty/holdout validation, and call `1.7e-3` the conditional tolerance on a first-moment offset with all other response moments fixed. Do not call it the unique limiting systematic.

## Bottom line

The following survive review: the stored-array layout; $P_\tau=-A_\tau$ and the derivative `7.87`; the shape-only covariance pricing; the central exact/reco sensitivity numbers; the near-zero a1 energy-fraction response; and the qualitative observation that current reconstruction retains longitudinal information much better than transverse information.

The following do not survive in their present form: the label $B_{\rm canonical}=-P_\tau\hat k$; eventwise “exact” pi $h_k=2x-1$; a causal apples-to-apples reading of `1.22 vs 2.71 vs 2.73`; the corrected transport bias `-0.018`; the `9.2e3` crossover and factor-4.2 systematics limit; and any reading of $1.6\times10^5$ events as a realistic LHC reach.
