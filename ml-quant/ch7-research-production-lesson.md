# Chapter 7 — From Research to Production

A research notebook proves that one historical experiment ran. A production strategy must reproduce the same information logic every decision time, generate auditable orders, survive failures, and reveal when its evidence no longer applies.

Research-to-production is therefore a continuity problem: preserve semantics from raw data to realized P&L.

## 7.1 The research contract

Before implementation, record:

- hypothesis and intended market mechanism;
- universe, prediction time, horizon, and label;
- exact feature definitions and availability rules;
- training window, validation, and retraining procedure;
- model, loss, and hyperparameter-selection policy;
- forecast-to-position mapping and constraints;
- execution, cost, borrow, and financing assumptions;
- expected capacity, failure modes, and kill criteria.

This contract separates an intentional strategy from a collection of code paths that happen to backtest well.

## 7.2 Artefact lineage

A production prediction should be traceable through

$$
\text{raw data version}
\to\text{feature version}
\to\text{training set}
\to\text{model artefact}
\to\text{prediction}
\to\text{target weight}
\to\text{order}
\to\text{fill}
\to\text{P\&L}.
$$

Store immutable identifiers, timestamps, configuration, code revision, environment, and checksums where appropriate. Reconstructing a decision should not depend on today's database contents.

## 7.3 Offline–online feature parity

The same feature must mean the same thing in research and live trading:

- time zones and market calendars;
- corporate-action treatment;
- as-of joins and data latency;
- missing-value defaults;
- cross-sectional universe and normalization;
- rolling-window boundaries;
- category or identifier mapping;
- numerical precision and library behaviour.

Shared feature definitions, point-in-time replay, and golden test cases reduce divergence. Equality of code helps but is not sufficient if upstream data or clocks differ.

## 7.4 Decision-time validation

Before producing orders, validate:

- required feeds are fresh and complete;
- universe count and exposure totals are plausible;
- feature values lie within monitored ranges;
- model version is approved and compatible with schema;
- prediction distribution and missingness are plausible;
- optimizer constraints and turnover are within limits;
- reference prices, borrow, and venue status are current.

Fail closed, reduce risk, or use a documented fallback depending on severity. Silent imputation after a feed outage can transform operational failure into market exposure.

## 7.5 Shadow and staged deployment

Replay production data without trading to compare live features and predictions with research expectations. Then use staged capital, instruments, or venues while monitoring slippage and operational behaviour.

Acceptance evidence includes:

- exact or tolerance-based feature parity;
- predicted-versus-realized latency;
- order and fill reconciliation;
- realized versus modelled costs;
- exposure and constraint agreement;
- P&L attribution closure.

A successful shadow period cannot reveal market impact caused only by real orders, so capital scaling remains a separate experiment.

## 7.6 Monitoring layers

### Data

Freshness, coverage, revisions, missingness, schema, extreme values, and cross-vendor discrepancies.

### Model

Prediction distribution, rank stability, calibration, feature contributions, model disagreement, and inference failures.

### Portfolio

Gross/net exposure, factor/sector risk, concentration, turnover, constraint binding, liquidity, borrow, and capacity.

### Execution

Rejects, participation, fill rate, spread capture, delay, slippage, impact, and venue mix.

### Outcomes

IC, gross/net return, factor attribution, costs, drawdown, tail loss, and regime performance.

Monitoring must have owners, thresholds, and actions. A dashboard without response logic is observation, not control.

## 7.7 Attribution and expectation gaps

Compare realized results with staged expectations:

$$
\text{research forecast}
\to\text{paper portfolio}
\to\text{live target}
\to\text{executed holdings}
\to\text{realized P\&L}.
$$

Differences arise from model decay, constraint changes, execution delay, partial fills, spread/impact, borrow, financing, data revisions, and operational incidents.

Attribution should explain both daily P&L and cumulative divergence from backtest expectations.

## 7.8 Retraining policy

Retraining can be:

- scheduled at a fixed cadence;
- triggered by enough new labelled data;
- triggered by monitored degradation;
- blocked during data or market anomalies.

The policy itself must be backtested. Retraining changes feature distributions, selected parameters, calibration, positions, and turnover. A new model should pass temporal validation, parity tests, exposure/cost checks, and staged comparison before promotion.

Keep rollback artefacts and avoid training automatically on suspected corrupted labels.

## 7.9 Champion–challenger evaluation

The production champion and a challenger should receive identical point-in-time inputs. Compare predictions and paper decisions prospectively, not by repeatedly retuning the challenger on the same live period.

Promotion criteria should combine forecast evidence, net portfolio value, risk, capacity, stability, and operational cost. Statistical superiority on one metric is insufficient.

## 7.10 Feedback and adaptive markets

Trading changes prices, liquidity, and observed fills. Scaling a strategy can reduce its own alpha. Competitors learn similar signals. Execution choices determine which costs and opportunities are observed.

Monitor performance versus capital and participation. Distinguish signal decay from self-impact by comparing low-participation segments, delayed paper portfolios, and counterfactual cost models.

The production distribution depends partly on the strategy's policy.

## 7.11 Incident response

A useful playbook classifies:

- data incident;
- model/inference incident;
- optimizer/risk incident;
- execution/venue incident;
- unexplained P&L incident;
- market-regime emergency.

For each, define detection, immediate risk action, escalation owner, evidence capture, recovery checks, and post-incident review. Preserve the state needed to replay decisions before changing systems.

## 7.12 Kill and reduction criteria

Predefine conditions such as:

- critical data parity failure;
- exposure or loss limit breach;
- unexplained attribution gap;
- realized costs outside validated range;
- sustained IC or calibration degradation;
- liquidity/borrow loss that invalidates capacity;
- evidence that the economic mechanism no longer holds.

Use graded responses—block new positions, reduce risk, switch to fallback, or stop—rather than improvising under loss.

Statistical thresholds should account for repeated monitoring and natural strategy variance.

## 7.13 Governance and independent review

Independent review should reproduce data timing, validation, accounting, and limiting cases—not merely read model code. Document material changes, approvals, model scope, exceptions, and residual risks.

Complexity needs a benefit. A simpler model with clearer controls, lower turnover, and stable behaviour may have higher expected production value than a marginally stronger backtest.

## 7.14 Failure modes

- Reimplementing live features without parity replay.
- Promoting on one recent favourable period.
- Monitoring predictions while ignoring orders and fills.
- Retraining during a corrupted-data incident.
- Comparing live P&L directly with an uncosted research backtest.
- Setting kill rules only after losses occur.
- Losing artefacts required to reconstruct a historical decision.

## 7.15 Knowledge checks

1. What lineage is needed to reconstruct one trading decision?
2. Why can shared feature code still fail parity?
3. What does shadow deployment test, and what can it not test?
4. How do signal decay and self-impact differ?
5. Why must a retraining policy itself be validated?

### Solution outlines

1. Data, feature, model, prediction, portfolio, order, fill, and P&L versions/timestamps.
2. Live clocks, upstream data, calendars, schemas, and latency can differ.
3. It tests data, feature, prediction, and order logic without capital; it cannot fully reveal own market impact.
4. One is loss of predictive relation; the other is alpha consumed by the strategy's own trading.
5. Cadence and triggers affect adaptation, turnover, selection, and exposure just like model hyperparameters.

## 7.16 Course synthesis

ML for quant research is an evidence chain:

1. define a timed economic question;
2. reconstruct the point-in-time information set;
3. validate the full adaptive pipeline through time;
4. control leakage and research search;
5. translate forecasts into net, capacity-aware portfolios;
6. stress non-stationarity and uncertainty;
7. preserve semantics and attribution in production.

Return to the [Machine Learning for Quant Research contents](index.html).
