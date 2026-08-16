# Chapter 2 — Point-in-Time Financial Data

A backtest is a historical simulation of what could have been known and traded. A database containing correct final values is not automatically a correct historical information set.

Point-in-time data engineering is therefore part of research methodology, not a preprocessing detail.

## 2.1 Four clocks

For each observation distinguish:

- **event time:** when the underlying economic event occurred;
- **publication time:** when the source released it;
- **ingestion time:** when the research system received it;
- **effective time:** when it may first be used by the strategy after latency and processing.

For revised data, also record the period the value describes and the version timestamp. A fiscal-quarter revenue value published weeks later cannot be attached to quarter end as though it were known then.

## 2.2 Bitemporal thinking

A point-in-time table often needs two axes:

1. **valid time:** when the fact applies economically;
2. **knowledge time:** when the system knew that version.

A query “as of 2019-06-01” should return only versions available then, not today's restated history. This protects research from silent look-ahead when vendors update old rows in place.

## 2.3 Prices and corporate actions

Raw prices are affected by splits, dividends, rights, spin-offs, and symbol changes. Adjusted prices simplify return calculations but can be dangerous if the entire historical adjustment factor embeds a future corporate action.

For each use case distinguish:

- raw executable prices;
- total-return series;
- split-adjusted features;
- cash dividends known on declaration, ex-date, or payment date;
- adjustment factors and when they become available.

Do not mix adjusted labels with unadjusted execution prices without a consistent accounting identity.

## 2.4 Delistings and survivorship

Failed and delisted securities often have poor final returns. Omitting them biases strategies upward. Historical universes should include securities that later disappear and incorporate delisting returns, cash settlements, or documented conservative assumptions.

Index constituent histories require entry and deletion dates known at each rebalance. Today's constituents are not a historical universe.

## 2.5 Fundamentals and revisions

Financial statements have period end, filing/publication timestamp, vendor ingestion timestamp, and possible restatement timestamp. Common safe constructions use the latest version known as of the feature cutoff, perhaps with an additional reporting lag.

Restatements can be useful prediction targets or diagnostics, but only if version history is preserved. Replacing original values with restated values manufactures foresight.

## 2.6 Alternative data

Web traffic, satellite, transactions, news, and text data add unique failure modes:

- vendor coverage changes over time;
- entities are mapped using future identifiers;
- historical archives are backfilled after onboarding;
- timestamps reflect crawl rather than publication;
- deduplication uses future knowledge;
- compliance and licensing restrict use.

An apparently long history may be reconstructed using a collection process that did not exist historically.

## 2.7 Joins are information operations

Joining tables can leak through keys and timing. An as-of join should select the latest record whose availability time is no later than the prediction cutoff:

$$
t_{record}\le t_{decision}.
$$

Entity mapping must also be historical. Mergers, ticker reuse, share classes, and identifier changes make naive symbol joins unreliable.

Validate join coverage by time and subgroup. Sudden coverage improvement can become a hidden time feature.

## 2.8 Missingness and selection

Missing data may mean “not yet reported,” “not covered,” “not applicable,” “system failure,” or “trading suspended.” These mechanisms have different economic meanings.

Imputation can use only contemporaneously available information. Cross-sectional median imputation should use the point-in-time universe at that date. A missingness indicator may be predictive, but if missingness reflects vendor operations rather than economics, the relation may not survive production.

## 2.9 Quality checks

For each data source, monitor:

- duplicate keys and conflicting versions;
- monotonic publication/ingestion timestamps;
- values arriving before event or publication time;
- extreme returns around corporate actions;
- universe counts, coverage, and missingness through time;
- stale values and impossible prices/volumes;
- cross-vendor reconciliation;
- revision frequency and magnitude.

Plot data availability through time before modelling. Structural jumps often reveal backfills or methodology changes.

## 2.10 Reproducible snapshots

A research result should identify:

- immutable source snapshot or version;
- query and as-of logic;
- corporate-action policy;
- universe construction;
- entity mapping version;
- feature code revision;
- row counts and quality-report artefact.

If historical queries change when the vendor refreshes data, the research is not reproducible.

## 2.11 Worked example: earnings feature

To compute earnings surprise at decision time:

1. identify the release and exact publication timestamp;
2. use consensus estimates frozen before release;
3. preserve original reported value rather than later restatement;
4. map company/security using historical identifiers;
5. delay availability for feed and processing latency;
6. execute no earlier than the strategy's feasible next trading point;
7. retain firms later delisted.

Attaching earnings to fiscal-quarter end or using final consensus history would leak.

## 2.12 Failure modes

- Treating event time as availability time.
- Using final revised fundamentals in historical features.
- Applying future split adjustments to purportedly executable prices without care.
- Dropping delisted securities or missing delisting returns.
- Joining by today's ticker mapping.
- Treating vendor backfill as a historically live feed.
- Imputing from future or out-of-universe observations.

## 2.13 Knowledge checks

1. Distinguish valid time from knowledge time.
2. Why can a fully adjusted price series introduce subtle look-ahead?
3. How does survivorship bias affect return research?
4. What is the correct predicate for a point-in-time as-of join?
5. Why should data coverage be plotted through time?

### Solution outlines

1. One says when the fact applies; the other when a particular version became known.
2. Adjustment factors can be based on corporate actions announced after the historical observation.
3. It disproportionately removes failures and poor outcomes, overstating historical performance.
4. Select the latest eligible version with availability timestamp no later than decision time.
5. Jumps reveal backfills, vendor changes, or selection shifts that models may exploit.

## 2.14 What to retain

- Correct final data are not necessarily point-in-time correct data.
- Publication, ingestion, and effective-use clocks determine feature availability.
- Corporate actions, delistings, revisions, and entity histories belong in the research design.
- Joins and imputations can leak just as models can.
- Immutable snapshots and data-quality artefacts are required evidence.

Next: [Chapter 3 — Validation for Financial Data](ch3-financial-validation-viewer.html).
