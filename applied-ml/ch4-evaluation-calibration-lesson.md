# Chapter 4 — Evaluation, Calibration & Decisions

A model does not have one universal performance number. Evaluation asks whether its outputs support a particular decision under a particular data distribution and cost structure.

## 4.1 Start from the decision

For binary classification, predictions create true positives, false positives, true negatives, and false negatives. From these:

$$
\text{precision}=\frac{TP}{TP+FP},\qquad
\text{recall}=\frac{TP}{TP+FN},
$$

$$
\text{specificity}=\frac{TN}{TN+FP}.
$$

Accuracy weights every error equally. That is rarely appropriate when prevalence is low or error costs differ.

## 4.2 Base rates change meaning

Suppose default prevalence is 1%. A classifier with 90% sensitivity and 90% specificity applied to 10,000 cases produces roughly:

- 90 true positives and 10 false negatives;
- 990 false positives and 8,910 true negatives.

Precision is only $90/(90+990)=8.3\%$. Good sensitivity and specificity do not guarantee a high posterior probability when the event is rare.

## 4.3 Thresholds are decisions

If a model estimates $p=P(Y=1\mid X)$ and false-positive and false-negative costs are $C_{FP}$ and $C_{FN}$, predict positive when

$$
p>\frac{C_{FP}}{C_{FP}+C_{FN}}
$$

under the simplest two-action setup. A 0.5 threshold is optimal only for a special symmetric cost structure and well-calibrated probabilities.

Operational constraints can instead fix capacity: review only the top 1,000 cases, trade only the top decile, or target a minimum recall.

## 4.4 ROC and precision–recall curves

An ROC curve plots true-positive rate against false-positive rate as the threshold varies. ROC AUC equals the probability that a randomly chosen positive receives a higher score than a randomly chosen negative, with ties handled appropriately.

Precision–recall curves emphasize performance on the positive class and make prevalence visible. Under rare events, a modest false-positive rate can still overwhelm true positives, so PR curves are often more informative.

AUC measures ranking, not probability accuracy or business value. Two models with identical AUC can differ greatly at the operating threshold.

## 4.5 Proper scoring rules

Log loss

$$
-[y\log p+(1-y)\log(1-p)]
$$

and Brier score

$$
(y-p)^2
$$

are proper scoring rules: in expectation, honest probabilities minimize them. Log loss punishes confident errors especially strongly.

Hard-label metrics discard probability information. Use them for the final action, but use proper scores and calibration diagnostics when probability quality matters.

## 4.6 Calibration

A model is calibrated if cases assigned probability $p$ experience the event about proportion $p$:

$$
P(Y=1\mid \widehat p=p)\approx p.
$$

Calibration plots group predictions and compare average prediction with observed frequency. Binning introduces variance and can hide local problems, so sample size and uncertainty bands matter.

Platt scaling fits a logistic mapping; isotonic regression fits a monotone nonparametric mapping. Calibration must be trained on data separate from the base model fit and evaluated on another held-out set or nested scheme.

## 4.7 Imbalanced data

Class weighting and resampling change the effective training objective. They can improve ranking or minority recall, but predicted probabilities may no longer reflect deployment prevalence without correction.

Often the cleanest approach is:

1. train an appropriate probabilistic or ranking model;
2. evaluate with prevalence-aware metrics;
3. calibrate under the deployment distribution;
4. choose a threshold from costs or constraints.

Synthetic oversampling is not a substitute for understanding how minority examples arise.

## 4.8 Regression metrics

Mean squared error emphasizes large deviations and estimates conditional means. Mean absolute error is more robust and targets medians. $R^2$ compares squared error with a mean baseline:

$$
R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

Out-of-sample $R^2$ can be negative, meaning the model underperforms the reference mean. Percentage errors behave badly near zero. Quantile loss is appropriate when asymmetric intervals or tail decisions matter.

## 4.9 Uncertainty in metrics

An evaluation score is an estimate. Its uncertainty depends on sample size, prevalence, clustering, and repeated model selection.

- paired bootstrap can compare models on the same cases;
- cluster or block bootstrap preserves dependence;
- repeated splits reveal selection instability;
- subgroup intervals prevent overreading tiny slices.

Do not treat cross-validation folds as independent replicates without examining overlap and dependence.

## 4.10 Failure modes

- Reporting accuracy for a rare-event problem.
- Selecting thresholds on the final test set.
- Calling AUC a measure of calibration.
- Resampling classes and trusting raw predicted probabilities.
- Comparing metrics computed on different populations.
- Reporting tiny subgroup differences without uncertainty.
- Optimizing a proxy metric disconnected from downstream utility.

## 4.11 Knowledge checks

1. Why can high specificity still produce low precision for rare events?
2. What does ROC AUC mean probabilistically?
3. How do discrimination and calibration differ?
4. Why is log loss called proper?
5. When can out-of-sample $R^2$ be negative?

### Solution outlines

1. The large negative class can generate many false positives even at a low false-positive rate.
2. It is the probability a random positive outranks a random negative.
3. Discrimination concerns ordering classes; calibration concerns numerical probability accuracy.
4. Reporting the true conditional probability minimizes expected loss.
5. When squared prediction error exceeds that of the chosen mean baseline.

## 4.12 What to retain

- Metrics are meaningful only relative to a target distribution and decision.
- Ranking, probability estimation, and thresholded actions are distinct tasks.
- Base rates and costs determine operating value.
- Calibration requires held-out data and can drift.
- Metric uncertainty is part of evaluation.

Next: [Chapter 5 — Validation, Leakage & Features](ch5-validation-features-viewer.html).
