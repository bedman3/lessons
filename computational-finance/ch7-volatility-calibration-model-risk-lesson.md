# Chapter 7 — Volatility Surfaces, Calibration & Model Risk

Black–Scholes assumes one constant volatility. Markets quote a different implied volatility across strike and maturity. The resulting surface is not merely a chart of beliefs: it is a compact representation of option prices constrained by no-arbitrage.

More flexible models can fit the surface, but fit alone does not identify realistic dynamics or reliable hedges.

## 7.1 Implied volatility is a price coordinate

For an observed option price $C_{mkt}$, implied volatility $\sigma_{imp}$ solves

$$
C_{BS}(S_0,K,T,r,d,\sigma_{imp})=C_{mkt}.
$$

It is the constant Black–Scholes volatility that reproduces one price. It is not directly the expected future realized volatility.

The inversion is ill-conditioned when vega is small, such as deep in/out-of-the-money or near-expiry options. Price noise can then create large implied-volatility noise.

Useful coordinates include forward log-moneyness

$$
k=\log(K/F_T)
$$

and total implied variance

$$
w(k,T)=\sigma_{imp}(k,T)^2T.
$$

Total variance often behaves more smoothly across maturity than volatility itself.

## 7.2 Static no-arbitrage across strikes

For fixed maturity, a call price must be decreasing and convex in strike:

$$
\frac{\partial C}{\partial K}\le0,
\qquad
\frac{\partial^2C}{\partial K^2}\ge0.
$$

The second derivative is related to the risk-neutral terminal density:

$$
\frac{\partial^2C}{\partial K^2}
=e^{-rT}f_Q(K).
$$

Negative curvature implies a negative density and a butterfly arbitrage. Put–call parity and intrinsic/forward bounds must also hold.

## 7.3 Calendar arbitrage

For comparable forward-normalized strikes, option value should not violate monotonicity in maturity. Surface parameterizations often impose nondecreasing total variance in $T$ under suitable coordinates.

Checking raw implied volatilities is insufficient: a lower volatility at a longer maturity can still correspond to higher total variance and no arbitrage.

Interpolation must preserve strike convexity and calendar consistency. Smooth-looking splines can create arbitrage between quoted nodes.

## 7.4 Local volatility

A local-volatility model specifies

$$
dS_t=(r-d)S_tdt+\sigma_{loc}(S_t,t)S_tdW_t^Q.
$$

Under regularity and a complete arbitrage-free option surface, Dupire's relation recovers local variance schematically as

$$
\sigma_{loc}^2(K,T)
=\frac{\text{time derivative and carry terms of }C}
{\tfrac12K^2C_{KK}}.
$$

Local volatility can reproduce all European marginal distributions by construction. But matching every marginal $S_T$ does not determine the joint path dynamics. Barrier prices and smile evolution can still be unrealistic.

The formula differentiates market data twice in strike, amplifying noise. Stable surface smoothing and arbitrage control are part of the model, not cosmetic preprocessing.

## 7.5 Stochastic volatility

The Heston model uses

$$
dS_t=(r-d)S_tdt+\sqrt{v_t}S_tdW_t^S,
$$

$$
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,
$$

with

$$
d\langle W^S,W^v\rangle_t=\rho dt.
$$

- $\theta$: long-run variance;
- $\kappa$: mean-reversion speed;
- $\xi$: volatility of variance;
- $\rho$: spot–variance correlation, strongly influencing skew;
- $v_0$: initial variance.

The Feller condition $2\kappa\theta\ge\xi^2$ helps keep variance strictly positive, though numerical schemes and calibration may operate outside it. Naive Euler steps can make variance negative; full truncation or specialized schemes are safer.

## 7.6 Local versus stochastic volatility

| Question | Local volatility | Stochastic volatility |
|---|---|---|
| Fits today's European surface | Exactly in idealized theory | Approximately through calibration |
| Extra randomness beyond spot | No | Yes |
| Market completeness | Complete in one-factor diffusion idealization | Generally incomplete with stock alone |
| Smile dynamics | Deterministic function of spot/time | Random variance factor |
| Path-dependent pricing | Can be fragile if dynamics are unrealistic | Often richer, still model-dependent |

Hybrid local-stochastic volatility aims to combine exact surface fit with richer dynamics, at higher calibration and implementation complexity.

## 7.7 Calibration is an inverse problem

Given parameters $\theta$, a weighted least-squares objective might be

$$
\min_\theta\sum_iw_i\left(C_{model,i}(\theta)-C_{market,i}\right)^2
+\lambda R(\theta).
$$

Weights may use bid–ask spreads, vegas, liquidity, or risk relevance. Price-space errors align with currency P&L; volatility-space errors equalize quoted units but overweight low-vega options unless adjusted.

Regularization $R(\theta)$ discourages rough surfaces, implausible parameters, or unstable day-to-day changes. Constraints enforce positivity and no-arbitrage structure.

## 7.8 Identifiability and parameter uncertainty

Different parameter combinations can fit vanilla prices almost equally well. Flat objective directions mean parameters are weakly identified. A precise optimizer result is not the same as precise economic knowledge.

Diagnose with:

- multi-start optimization;
- profile objectives and local curvature;
- parameter stability across dates;
- out-of-sample instruments or maturities;
- sensitivity of exotic prices and hedges across near-optimal calibrations.

Calibration error, quote noise, and model-form error should not be collapsed into one residual statistic.

## 7.9 Pricing fit versus hedging dynamics

A model calibrated perfectly at time zero can hedge poorly because hedging depends on how the surface moves with spot and time. Sticky-strike, sticky-delta, local-vol, and stochastic-vol models imply different smile dynamics and Greeks.

Assess:

- repricing error inside bid–ask spreads;
- delta/vega hedge P&L through historical moves;
- scenario response to spot, skew, vol level, rates, and correlation;
- stability of Greeks under numerical and calibration choices;
- path-dependent benchmark prices across plausible models.

The relevant model is often the one whose residual risks are understandable and manageable, not the one with the smallest in-sample error.

## 7.10 Physical versus risk-neutral volatility dynamics

Option calibration primarily identifies risk-neutral dynamics. Forecasting realized variance requires a physical model and volatility risk premium. The drift of the variance process generally differs between $P$ and $Q$.

Using risk-neutral parameters as real-world forecasts without an explicit measure transformation confuses pricing with prediction, just as in Chapter 1.

## 7.11 Model-risk governance

A defensible model process records:

1. product scope and assumptions;
2. data sources, cleaning, and stale-quote rules;
3. arbitrage checks and interpolation method;
4. calibration objective, weights, bounds, and optimizer diagnostics;
5. independent benchmarks and limiting cases;
6. Greek and hedge validation;
7. parameter and P&L monitoring;
8. fallback procedures and known limitations.

Model reserves or valuation adjustments may reflect uncertainty that cannot be diversified or hedged reliably.

## 7.12 Failure modes

- Reading implied volatility as a direct physical forecast.
- Fitting each strike independently and creating butterfly arbitrage.
- Differentiating a noisy surface to obtain unstable local volatility.
- Reporting optimizer precision while ignoring parameter non-identifiability.
- Validating only vanilla price fit for an exotic-pricing model.
- Using naive Euler for a square-root variance process without boundary analysis.
- Assuming a perfect snapshot fit guarantees realistic smile dynamics.

## 7.13 Knowledge checks

1. Why can implied volatility be unstable when vega is small?
2. What no-arbitrage property connects call convexity to a density?
3. Why does exact European calibration not determine barrier prices?
4. How does $\rho$ influence Heston skew intuitively?
5. Why can two near-identical calibration errors imply different exotic risks?

### Solution outlines

1. A small price change requires a large volatility change when $\partial C/\partial\sigma$ is small.
2. $e^{rT}C_{KK}$ acts as the nonnegative risk-neutral terminal density.
3. Europeans determine one-time marginals, while barriers depend on joint path behaviour.
4. Negative spot–variance correlation raises variance during spot declines, producing downside skew.
5. Weakly identified parameters can control unquoted path or volatility exposures differently.

## 7.14 Course synthesis

Computational derivatives pricing is a chain:

1. no-arbitrage selects a pricing measure;
2. expectation or PDE represents the claim value;
3. simulation or grids approximate that representation;
4. calibration connects the model to market prices;
5. hedging tests the model's dynamics;
6. error decomposition separates numerical uncertainty from model risk.

Return to the [Monte Carlo, PDEs & Advanced Derivatives contents](index.html).
