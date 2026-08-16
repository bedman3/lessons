# Chapter 6 — Non-Stationarity, Regimes & Robustness

Financial relationships change because participants adapt, regulations shift, technology evolves, risk premia vary, and data pipelines change. A model trained on history faces uncertainty not only about parameters but about whether the same relationship still exists.

Robustness is evidence across plausible environments, not immunity to change.

## 6.1 Forms of change

- **Covariate shift:** $P_t(X)$ changes.
- **Label shift:** $P_t(Y)$ changes.
- **Concept drift:** $P_t(Y\mid X)$ changes.
- **Structural break:** parameters or mechanisms change abruptly.
- **Gradual decay:** predictive relation weakens over time.
- **Measurement shift:** vendor, definition, or collection process changes.

Each suggests different diagnostics. Reweighting covariates cannot fix a changed conditional relationship.

## 6.2 Stationarity as a local approximation

Weak stationarity requires constant mean and covariance depending only on lag. Returns may be closer to stationary than prices, while volatility and correlations remain time-varying.

Differencing, demeaning, scaling, or residualization can stabilize moments, but transformations do not guarantee stable predictive structure. A stationary-looking feature can have a changing relation to returns.

## 6.3 Rolling and exponentially weighted estimation

A rolling window assumes recent $L$ observations are relevant and older ones are not. Exponential weights

$$
w_k\propto\lambda^k,\qquad0<\lambda<1,
$$

create gradual decay with effective sample size smaller than the raw count.

Short memory adapts quickly but raises estimation variance. Long memory stabilizes estimates but reacts slowly. Select decay under nested walk-forward validation and examine stability across nearby values.

## 6.4 Regime models

Regimes may be observable definitions—high volatility, recession, illiquidity—or latent states such as a hidden Markov model. Conditional models can learn different relationships by state:

$$
E[Y\mid X,Z=z]=f_z(X).
$$

Risks include small samples per state, hindsight-defined regimes, unstable state labels, and false certainty about current state. Soft probabilities are often safer than hard switches.

A regime label built from future full-period information leaks even if the predictive model itself does not.

## 6.5 Shrinkage and partial pooling

When sector- or regime-specific estimates are noisy, shrink them toward a common estimate:

$$
\widehat\theta_g^{shrunk}
=\alpha_g\widehat\theta_g+(1-\alpha_g)\widehat\theta_{global}.
$$

Groups with more evidence receive larger $\alpha_g$. Hierarchical models formalize this partial pooling and quantify uncertainty.

Shrinkage sacrifices local flexibility to reduce extreme sampling error—a useful trade in sparse regimes.

## 6.6 Ensembles

Ensembles help when component errors are imperfectly correlated. Diversify across:

- model families;
- feature views;
- training windows;
- horizons;
- economic mechanisms;
- retraining schedules.

Combining many variants of one dataset and objective may provide little true diversity. Weight selection itself can overfit; equal or strongly shrunk weights are robust baselines.

## 6.7 Sources of uncertainty

- **Aleatoric:** irreducible outcome noise.
- **Parameter:** finite-sample uncertainty within a model.
- **Model-form:** uncertainty over functional assumptions.
- **Distributional:** uncertainty about future environment.

Prediction intervals estimated under one stationary model usually omit the last two. Scenario and ensemble dispersion can reveal some model uncertainty but are not automatically calibrated probabilities.

## 6.8 Robust objectives

Instead of maximizing average historical performance, consider:

- worst-regime or lower-quantile performance;
- penalties for unstable coefficients or turnover;
- distributionally robust loss over nearby distributions;
- Bayesian posterior or shrinkage-aware sizing;
- forecast scaling by estimated uncertainty;
- portfolio constraints on stress exposures.

Robustness is not free: overly pessimistic objectives can discard real opportunity. Specify the uncertainty set from plausible mechanisms rather than arbitrary severity.

## 6.9 Sensitivity surfaces

Plot results over reasonable ranges of:

- training window and decay;
- feature lags and prediction horizon;
- regularization and model capacity;
- neutralization strength;
- turnover penalty and execution delay;
- cost and impact multipliers;
- universe filters.

A broad plateau is more credible than a sharp isolated optimum. Sensitivity analysis also identifies which assumptions require monitoring.

## 6.10 Stress tests

Historical scenarios cover known joint moves but only one realized path. Synthetic stresses perturb volatility, correlation, liquidity, spreads, borrow, delays, missing data, and factor relationships.

Stress the entire strategy pipeline: signals may become stale, covariance estimates unstable, optimization concentrated, and costs nonlinear simultaneously.

Reverse stress asks what combination of changes causes unacceptable loss or invalidates the research thesis.

## 6.11 Detecting degradation

Track leading and lagging indicators:

- feature availability and distribution;
- forecast dispersion and calibration;
- IC/rank IC by date;
- turnover and optimizer constraint binding;
- gross-to-net decay;
- factor exposure and crowding proxies;
- realized P&L and drawdown;
- model disagreement.

Use control limits carefully: repeated monitoring creates sequential-testing issues. Predefine escalation, reduced-risk, retraining, and shutdown actions.

## 6.12 Failure modes

- Calling one favourable historical crisis a robust stress test.
- Defining regimes with future information.
- Selecting windows from one validation period.
- Treating ensemble size as diversity.
- Reporting within-model intervals as full uncertainty.
- Optimizing worst-case outcomes over arbitrary impossible scenarios.
- Retraining automatically when the data pipeline itself has broken.

## 6.13 Knowledge checks

1. Distinguish covariate shift from concept drift.
2. What is the bias–variance trade-off in window length?
3. Why can hard regime switching be unstable?
4. What makes an ensemble genuinely diverse?
5. Why is a sensitivity plateau stronger evidence than one optimum?

### Solution outlines

1. Input distribution changes versus the conditional target relationship changing.
2. Short windows adapt with high variance; long windows stabilize but retain stale structure.
3. State estimates are uncertain and small changes can flip the entire model or portfolio.
4. Distinct economic mechanisms, data views, horizons, or inductive biases with low error correlation.
5. It shows performance is not dependent on precise retrospective tuning.

## 6.14 What to retain

- Financial models operate under distributional uncertainty, not merely parameter noise.
- Recency weighting and regimes trade adaptation for statistical stability.
- Shrinkage and simple ensembles protect against fragile local estimates.
- Robustness needs sensitivity, stress, and regime evidence.
- Monitoring should connect detected degradation to predefined actions.

Next: [Chapter 7 — From Research to Production](ch7-research-production-viewer.html).
