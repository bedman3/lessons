# Chapter 1 — From Economic Hypothesis to Prediction Target

Quant research begins before the model. A useful project states why information might predict an outcome, when that information is available, what action follows, and why the effect could survive costs and competition.

Without those choices, an algorithm can optimize a mathematically precise target that no trading decision can use.

## 1.1 Hypothesis, forecast, and decision

Keep three layers separate:

1. **Economic hypothesis:** a proposed mechanism, such as slow information diffusion or compensation for risk.
2. **Forecasting problem:** estimate a quantity such as next-month residual return or future volatility.
3. **Decision rule:** turn forecasts into positions subject to risk, liquidity, and cost constraints.

A statistically predictable target need not create positive net P&L. The signal may be too small, crowded, expensive, or already captured by another exposure.

## 1.2 Define the prediction clock

Every sample needs an information time $t$, prediction horizon $h$, and label whose construction begins after information is available.

For a simple forward return,

$$
y_{i,t}^{(h)}=\frac{P_{i,t+h}}{P_{i,t}}-1.
$$

But “$P_{i,t}$” must mean an executable or consistently marked price at a specified time. A close-to-close label is unavailable for a decision executed at the same close unless the execution protocol accounts for the auction.

Write a timeline for:

- last feature timestamp;
- model evaluation timestamp;
- order submission and fill assumptions;
- label start and end;
- when outcome becomes observable for retraining.

## 1.3 Choose the estimand

Possible targets include:

- raw return;
- excess return over cash or benchmark;
- residual return after factor neutralization;
- sign or rank of return;
- realized volatility, correlation, spread, volume, or default;
- quantile or tail-event probability.

The target determines the loss. Squared loss estimates conditional means; quantile loss estimates conditional quantiles; cross-entropy estimates event probabilities. A ranking objective may align better with top-versus-bottom portfolio construction than pointwise MSE.

Residualizing labels changes the question. It can focus the learner on idiosyncratic structure, but estimated factors and residualization timing must be point-in-time correct.

## 1.4 Time-series versus cross-sectional learning

**Time-series prediction** asks how one asset or aggregate evolves across time:

$$
E[Y_{t+h}\mid X_t].
$$

**Cross-sectional prediction** asks which assets outperform others at a common future horizon:

$$
E[Y_{i,t+h}\mid X_{i,t},\mathcal I_t].
$$

Panel models combine both. The distinction affects normalization, loss weighting, validation, and dependence. A cross-sectional z-score uses contemporaneous peers and should not be computed using assets outside the actual point-in-time universe.

## 1.5 Overlapping labels

If monthly forward returns are sampled daily, adjacent labels share most of their price path. For horizon $h$, labels $y_t^{(h)}$ and $y_{t+1}^{(h)}$ overlap.

This creates serial dependence, inflates apparent sample size, and lets training outcomes overlap validation information near boundaries. Remedies include non-overlapping sampling, dependence-aware standard errors, and purging/embargo described later.

More rows do not necessarily mean more independent decisions.

## 1.6 Universe definition is part of the target

A universe rule must specify:

- eligible security types and venues;
- minimum price, liquidity, seasoning, and data history;
- reconstitution frequency;
- treatment of IPOs, mergers, delistings, and suspended names;
- when eligibility information becomes known.

Defining today's survivors and applying them to history creates survivorship bias. Liquidity filters based on future averages do the same.

The prediction target is conditional on this universe and should not be generalized beyond it without evidence.

## 1.7 Baselines

Useful baselines include:

- zero or historical mean return;
- benchmark/factor prediction;
- previous volatility or exponentially weighted estimate;
- equal-weight or current production signal;
- regularized linear model.

A model should beat baselines under identical timing, cost, and risk rules. Comparing a complex net-cost backtest with a gross or differently filtered baseline is not evidence.

## 1.8 Research degrees of freedom

Choices over universe, horizon, label, winsorization, features, neutralization, model, loss, retraining frequency, and evaluation period form a large implicit search.

Document these choices before or during experimentation. A research log should distinguish:

- mechanism-motivated choices;
- data-quality corrections;
- validation-driven tuning;
- post-hoc discoveries needing confirmation.

One final backtest hides the number of paths tried to reach it.

## 1.9 Worked target specification

Suppose the hypothesis is that analyst revisions diffuse slowly among mid-cap equities.

- **Information time:** 30 minutes after a revision first appears in the licensed feed.
- **Universe:** point-in-time mid-cap common shares passing prior-month liquidity rules.
- **Feature:** standardized revision magnitude relative to contemporaneous sector peers.
- **Label:** five-day return residualized against market and sector using only prior estimates.
- **Action:** trade next available liquid interval; hold five days with staggered cohorts.
- **Evaluation:** rank IC, net long–short return, turnover, factor exposure, and capacity.

This specification makes leakage and execution assumptions inspectable before model choice.

## 1.10 Failure modes

- Choosing a label because it gives the strongest historical score.
- Using a price or fundamental value unavailable at decision time.
- Treating overlapping labels as independent.
- Defining the historical universe from current constituents.
- Optimizing prediction error without specifying a portfolio action.
- Residualizing with future factor estimates.
- Presenting an explored configuration as a preregistered hypothesis.

## 1.11 Knowledge checks

1. Why are hypothesis, forecast, and decision distinct layers?
2. Give an example of a label that leaks through timing.
3. How do cross-sectional and time-series targets differ?
4. Why do overlapping labels reduce effective sample size?
5. What must a universe definition include to avoid survivor conditioning?

### Solution outlines

1. Mechanism motivates signal, prediction estimates an outcome, and the decision applies constraints/costs; success at one layer need not imply success at the next.
2. Using the same closing price for features and an assumed pre-close execution without auction timing.
3. One predicts evolution through time; the other relative outcomes across entities at a common time.
4. Adjacent outcomes share future returns and are strongly dependent.
5. Point-in-time membership, eligibility timing, entry/exit, delistings, and liquidity history.

## 1.12 What to retain

- A quant target is defined by information time, horizon, universe, and action.
- Labels and losses encode the economic question.
- Cross-sectional, time-series, and panel problems need different validation logic.
- Overlap and universe selection change the true information content.
- Research flexibility must be recorded and later penalized or confirmed.

Next: [Chapter 2 — Point-in-Time Financial Data](ch2-point-in-time-data-viewer.html).
