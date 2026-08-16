# Chapter 5 — From Signal Evaluation to Realistic Portfolios

A forecast is not a portfolio. Between the two lie ranking, calibration, neutralization, sizing, constraints, trading, and costs. A model can predict returns out of sample yet lose money after this translation; a weak point forecast can still support a useful ranking.

Evaluate each layer separately so success or failure can be attributed.

## 5.1 Prediction metrics

For cross-sectional forecasts $\widehat y_{i,t}$, the information coefficient is the correlation with realized outcomes on date $t$:

$$
IC_t=\operatorname{Corr}_i(\widehat y_{i,t},y_{i,t}).
$$

Rank IC uses Spearman correlation and is robust to monotone transformations and outliers. Report the time-series mean, dispersion, autocorrelation, hit rate, and regime stability of $IC_t$.

Pooling all asset-dates into one correlation lets large cross-sectional dates or time trends dominate and understates common-shock dependence.

## 5.2 Calibration and monotonicity

Sort assets into forecast buckets and compare average predicted and realized outcomes. A useful score should generally produce monotone realized returns, though sampling error is large in tails.

Calibration asks whether a predicted 20 basis points corresponds to about 20 basis points under the stated horizon and universe. Many portfolio rules require ranking only, but sizing from forecast magnitude requires stable calibration.

Use out-of-sample predictions and form buckets independently within each date.

## 5.3 Quantile portfolios

A basic diagnostic forms long top-quantile and short bottom-quantile portfolios:

$$
r_{LS,t}=r_{top,t}-r_{bottom,t}.
$$

Vary the number of buckets and weighting rule. Smooth degradation away from extremes is stronger evidence than one isolated winning cutoff.

Equal weighting, value weighting, volatility scaling, and liquidity weighting test different economic implementations. Each changes exposure and capacity.

## 5.4 Neutralization

Forecasts may load on market beta, sector, size, value, momentum, country, or other known factors. Cross-sectional residualization projects signal $s$ away from exposure matrix $X$:

$$
s_{\perp}=s-X(X^TWX)^{-1}X^TWs.
$$

Alternatively impose exposure constraints in portfolio optimization. Neutralization clarifies what incremental information remains, but excessive controls can remove genuine signal or create unstable trades.

Use point-in-time exposures and inspect both raw and neutralized performance.

## 5.5 From forecasts to weights

A mean–variance-style optimizer chooses

$$
\max_w\left\{w^T\widehat\mu
-\frac\lambda2w^T\Sigma w
-C(w-w_{prev})\right\}
$$

subject to leverage, position, factor, liquidity, and mandate constraints.

$\widehat\mu$ comes from the model; $\Sigma$ and cost function are also estimated models. Portfolio output can be more sensitive to covariance and cost assumptions than to small forecast improvements.

Simple rank or proportional rules make good diagnostic baselines before optimization.

## 5.6 Turnover and holding overlap

One-way turnover is commonly

$$
TO_t=\frac12\sum_i|w_{i,t}-w_{i,t^-}|.
$$

Conventions differ for cash, flows, and leverage, so state the definition. With staggered holding cohorts, portfolio turnover is not the same as signal rank churn.

Inspect signal autocorrelation, desired versus executed weights, and turnover by source: entry/exit, resizing, universe changes, risk rebalancing, and corporate actions.

## 5.7 Transaction costs and impact

A stylized cost model may include

$$
C(q)=\text{fees}+\frac{\text{spread}}2|q|
+\eta\sigma\left(\frac{|q|}{ADV}\right)^\alpha |q|.
$$

Spread and fees are roughly linear; market impact is nonlinear in participation. Costs vary through time, across assets, and with urgency.

Calibrate only from information available to the historical strategy. Stress cost multipliers and delays. If profitability disappears under a small plausible change, the signal has little margin of safety.

## 5.8 Capacity

Scaling capital increases trade size relative to liquidity, worsening impact and fill quality. Capacity analysis recomputes net return across asset under management, not simply multiplying a fixed Sharpe.

Crowding, borrow availability, short fees, market participation limits, and alpha decay can create hard constraints. Capacity is strategy- and implementation-specific.

## 5.9 Backtest accounting

A self-financing return must reconcile:

$$
\text{ending equity}
=\text{starting equity}
+\text{holding P\&L}
+\text{trading P\&L}
+\text{income}
-\text{costs}
-\text{financing}.
$$

Handle dividends, splits, delistings, borrow, futures rolls, FX conversion, cash returns, and stale prices consistently. Compare positions before and after trades and preserve an audit trail from forecasts to orders to P&L.

## 5.10 Performance and risk evidence

Report more than annualized Sharpe:

- mean, volatility, drawdown, downside/tail loss;
- serial correlation and effective observations;
- gross and net performance;
- factor and sector exposures;
- turnover, costs, participation, and capacity;
- long and short legs separately;
- performance by time, universe, and market regime;
- forecast IC and portfolio attribution.

Sharpe annualization $\sqrt{K}\bar r/s$ assumes a sampling frequency and can mislead under autocorrelation or overlapping holdings.

## 5.11 Attribution

Decompose performance into forecast selection, factor exposure, sizing, constraints, trading delay, and costs. Compare:

1. raw signal portfolio;
2. neutralized portfolio;
3. risk-scaled portfolio;
4. constrained desired portfolio;
5. executed net-cost portfolio.

The incremental changes reveal where value is created or lost.

## 5.12 Worked evaluation ladder

For a daily equity score:

1. store walk-forward predictions;
2. compute daily rank IC with date-clustered uncertainty;
3. inspect decile monotonicity and sector-conditioned results;
4. compare raw and factor-neutralized spreads;
5. build simple equal-risk weights;
6. add realistic one-day execution delay;
7. add spread, impact, borrow, and financing;
8. scale AUM and stress costs;
9. attribute net P&L and exposures.

Only then decide whether a more complex optimizer is justified.

## 5.13 Failure modes

- Pooling panel observations and overstating statistical precision.
- Selecting a quantile cutoff on the final backtest.
- Reporting gross performance while ignoring signal churn.
- Neutralizing with future or unstable factor exposures.
- Using future average liquidity in historical costs.
- Treating covariance and impact estimates as known.
- Annualizing overlapping returns with an iid formula.

## 5.14 Knowledge checks

1. Why can rank IC be useful when forecast calibration is poor?
2. What does neutralization remove, and what can it accidentally remove?
3. Why is capacity nonlinear in capital?
4. How does a prediction backtest differ from an executed portfolio backtest?
5. What comparisons isolate the cost of constraints and trading?

### Solution outlines

1. A monotone score can order opportunities correctly without accurate magnitudes.
2. Modelled factor exposure; it may also remove economically genuine signal correlated with those factors.
3. Participation and impact grow with trade size and available liquidity is finite.
4. The latter includes weights, risk, turnover, delays, fills, financing, and costs.
5. Evaluate raw, neutralized, desired constrained, delayed, and net-cost stages on identical predictions.

## 5.15 What to retain

- Forecast skill, portfolio construction, and execution are separate evidence layers.
- Cross-sectional metrics should be formed by date and summarized through time.
- Neutralization clarifies incremental signal but adds model assumptions.
- Turnover, nonlinear impact, and capacity determine realizability.
- P&L attribution must reconcile forecasts, positions, trades, and costs.

Next: [Chapter 6 — Non-Stationarity & Robustness](ch6-nonstationarity-robustness-viewer.html).
