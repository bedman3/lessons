# Chapter 5 — Validation, Leakage & Feature Pipelines

Most real model failures are not caused by choosing the wrong algorithm. They come from evaluating the right algorithm on the wrong information set.

## 5.1 Three data roles

- **Training data** fit parameters.
- **Validation data** choose features, hyperparameters, thresholds, and stopping rules.
- **Test data** estimate performance after all choices are fixed.

This is a separation of information, not merely files. If test results influence another attempt, the test has become validation data.

## 5.2 Cross-validation

In $K$-fold cross-validation, each fold is held out once and validation losses are aggregated:

$$
\widehat R_{CV}=\frac1K\sum_{k=1}^K\widehat R_k.
$$

Larger $K$ trains on more data but increases computation and correlation among fitted models. Fold choice must respect dependence:

- stratified folds preserve class mix;
- grouped folds keep related entities together;
- time splits preserve causality and ordering;
- spatial blocks address geographic correlation.

## 5.3 Nested model selection

Hyperparameter tuning overfits validation scores. Nested cross-validation uses an inner loop for selection and an outer loop for evaluation. It is expensive but clean when data are limited and an unbiased comparison matters.

For deployment, a final untouched temporal or group holdout can be more informative than elaborate iid resampling.

## 5.4 Leakage taxonomy

Leakage occurs when training uses information unavailable at prediction time or information from the evaluation target.

- **Target leakage:** a feature is downstream of the label.
- **Temporal leakage:** future data influence past predictions.
- **Preprocessing leakage:** transformations learn from held-out rows.
- **Entity leakage:** the same person, asset, document, or near-duplicate spans splits.
- **Selection leakage:** the test result directs repeated model choices.

Ask for every feature: when exactly was this value known, to whom, and with what revision history?

## 5.5 Pipelines

A pipeline treats imputation, scaling, encoding, feature selection, and modelling as one fitted object. Within each fold:

1. fit preprocessing on training rows;
2. transform training and validation rows using those fitted values;
3. fit the model on transformed training rows;
4. evaluate on transformed validation rows.

Global normalization can leak distributional information. Target encoding is especially dangerous and requires out-of-fold construction.

## 5.6 Missing data

Missingness mechanisms include:

- MCAR: unrelated to observed or missing values;
- MAR: explained by observed variables;
- MNAR: related to the missing value itself after conditioning.

Simple imputation changes distributions and uncertainty. Missingness indicators can be predictive when absence itself carries process information. At deployment, missingness may shift because data pipelines or user behaviour change.

## 5.7 Feature engineering

Useful features express domain invariances or reduce the burden on the learner:

- ratios that remove scale;
- log transforms for multiplicative processes;
- lags and rolling summaries using only past data;
- cyclic encodings for periodic variables;
- interactions motivated by mechanism;
- embeddings or aggregations that respect entity boundaries.

Feature engineering is hypothesis construction. Every candidate added after inspecting results expands the effective search space.

## 5.8 Feature selection

Filter methods rank features independently of the final model. Wrapper methods evaluate subsets through model performance. Embedded methods such as lasso or tree splitting select during fitting.

Selection must occur inside validation folds. Stability across resamples and time is often more informative than a single selected set. Prediction importance does not imply causal importance.

## 5.9 Distribution shift diagnostics

Compare training and deployment-like data using:

- univariate distributions and missing rates;
- multivariate classifiers that distinguish periods;
- label prevalence and calibration;
- subgroup and temporal performance;
- feature ranges and category novelty.

A powerful discriminator between train and test periods shows shift, but not automatically which shift harms the target relationship.

## 5.10 Worked split design

For customer transactions with repeated users and monthly deployment:

1. choose a chronological outer test period;
2. create rolling training/validation windows before it;
3. keep each transaction's features point-in-time correct;
4. decide whether users can legitimately recur across time;
5. fit all preprocessing within each window;
6. select thresholds without touching the outer test.

Random row splitting would mix future conditions and user-specific information into both sides.

## 5.11 Failure modes

- Scaling or selecting features before splitting.
- Using revised historical data unavailable at the original timestamp.
- Letting duplicates cross folds.
- Filling missing values with full-dataset statistics.
- Tuning against one lucky validation period.
- Treating an iid split as evidence for future-regime performance.

## 5.12 Knowledge checks

1. Why must preprocessing be fitted inside each fold?
2. Give examples of target, temporal, and entity leakage.
3. When is grouped validation preferable to stratified random folds?
4. Why does target encoding require out-of-fold construction?
5. What does a train-versus-test classifier diagnose?

### Solution outlines

1. Otherwise held-out distribution information affects learned transformations.
2. Post-outcome feature; future rolling value; same customer or duplicate document across splits.
3. When related rows share information and deployment requires generalization to unseen groups.
4. A row's label must not help construct its own encoded feature.
5. Detectability of distribution shift, not necessarily target-performance damage.

## 5.13 What to retain

- Validation is an information-boundary design.
- The entire pipeline, including preprocessing and selection, must be evaluated out of sample.
- Splits should imitate deployment dependence and timing.
- Leakage can produce plausible but fictitious performance.
- Feature work expands the search process and must be accounted for.

Next: [Chapter 6 — Debugging, Interpretation & Production](ch6-debugging-production-viewer.html).
