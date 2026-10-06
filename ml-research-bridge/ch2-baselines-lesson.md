# Chapter 2 — Baselines Before Complexity

A baseline is not a ceremonial first line in a notebook. It defines what “better” means. A useful model study compares assumptions: no signal, a stable linear effect, and a nonlinear interaction model. If an added model cannot beat a simple reference on the same dates and under the same metric, complexity has not yet earned its cost.

## 2.1 Start with decision-matched baselines

Common regression references include:

- **Zero predictor:** $\hat y=0$. Useful when zero is a meaningful no-change or no-position estimate.
- **Training mean:** $\hat y=\bar y_{train}$. This is the least-squares constant fitted to the training observations. The usual reported $R^2$ denominator instead uses the mean of the evaluated targets; that scoring convention does not make the future target mean a valid live predictor.
- **Date-specific cross-sectional reference:** useful for scoring relative predictions within each date only when the reference prediction can be formed from information available at decision time (for example, a point-in-time neutral forecast). The realized cross-sectional mean or median of future validation targets is an evaluation statistic, not a live prediction baseline.
- **Existing score or policy:** often the most relevant practical comparator when the model would replace or augment an existing process.

Never compute a baseline from evaluation outcomes unless the real decision would know those outcomes. For future observations, a train-period mean is fixed before testing; any date-level reference must be computable using information available at the live decision time. Subtracting the realized future date median from predictions leaks the answer and cannot be reported as a valid predictive baseline.

For a non-return target, a zero predictor may be nonsensical. For a probability target, a training-prevalence predictor or calibrated existing model may be more informative. Baseline choice is part of the experiment contract.

## 2.2 Linear models expose structure

Ordinary least squares selects coefficients to minimize squared residuals:

$$
\hat\beta=\arg\min_\beta\sum_i (y_i-x_i^\top\beta)^2.
$$

It is fast, interpretable at a basic level, and useful for detecting sign, scale, and data issues. Correlated inputs can make individual coefficients unstable even when predictions are similar. A coefficient is conditional on other features in the fitted model; it is not automatically causal or a standalone feature value.

Ridge adds an $L_2$ penalty:

$$
\hat\beta_\lambda=\arg\min_\beta\sum_i(y_i-x_i^\top\beta)^2+\lambda\sum_j\beta_j^2.
$$

Increasing $\lambda$ shrinks coefficients and usually trades bias for lower variance. The intercept is conventionally not penalized. Standardization matters because otherwise a one-unit change in a feature with a large numeric scale is treated differently from a one-unit change in a small-scale feature. Fit the scaler using training data only; then apply its stored mean and scale to validation and test values.

### Worked example: shrinkage by hand

Take centered feature values $x=[-1,0,1]$ and centered targets $y=[-2,0,2]$. The unregularized slope is

$$
\hat\beta=\frac{\sum x_i y_i}{\sum x_i^2}=\frac{4}{2}=2.
$$

With ridge penalty $\lambda=2$, the one-feature solution is $4/(2+2)=1$. If the uncentered target was $[ -1,1,3 ]$, its mean is 1, so the intercept remains 1. Ridge predictions are $[0,1,2]$, with absolute errors $[1,0,1]$ and MAE $2/3$. The training-mean reference predicts $[1,1,1]$, with MAE $4/3$ on these same three points. This tiny example illustrates the arithmetic only; it is far too small to establish generalization. On real data, choose $\lambda$ using earlier validation blocks, never the final holdout.

## 2.3 A tree baseline asks a different question

A shallow decision tree partitions feature space into regions and predicts a constant per leaf. It can represent thresholds and interactions without pre-specifying them. That flexibility can capture genuine nonlinearities, but it can also partition noise. Depth, minimum leaf size, number of leaves, and boosting rounds control the fit-versus-stability tradeoff.

For cross-sectional equity data, a tree split on one feature might be useful only in one volatility or liquidity regime. Inspect whether a split recurs across time, not only whether it improves one aggregate score. A tree’s feature importance is a property of that fitted representation, not a causal explanation. Correlated features can exchange importance, and impurity-based importance can favor variables with more candidate split points.

A benchmark set might therefore be: zero or training mean, standardized ridge, and one constrained tree/boosting model. This is enough to ask whether signal is absent, approximately linear, or benefits from a modest nonlinear partition. Do not increase depth or grid size until the evaluation design and labels survive audit.

## 2.4 Preprocessing is fitted model state

Imputation values, means and scales, winsorization thresholds, PCA loadings, feature selectors, and encoders are learned from data. They belong inside the fit procedure. At each temporal split:

1. fit transformations on the training window;
2. transform that training window;
3. fit the predictor on transformed training rows;
4. transform validation rows with the already-fitted transformations;
5. score predictions without refitting on validation outcomes.

A pipeline helps enforce this separation, but it cannot decide whether a feature is point-in-time correct, whether label windows overlap, or whether the split unit should be date or asset. Those are research-design responsibilities.

## 2.5 Keep metrics tied to the use

Use the same dates, assets, target units, missing-data policy, and metric for every comparison. MAE and RMSE evaluate pointwise errors; correlation evaluates association; rank correlation evaluates within-sample ordering. None alone measures execution-adjusted P&L. If ranking is the intended use, compute the rank statistic per date and summarize it across dates. Pooling every asset-date row can let dates with more instruments dominate.

For each metric, compare model minus baseline on the same date blocks. A small positive average with wide block-to-block variation is weak evidence. A large score on a few dates may be a regime artifact or data problem. Chapter 3 covers uncertainty and selection controls.

## 2.6 Practice and answer key

**Exercise A.** Your features have ranges from $10^{-5}$ to $10^4$. You standardize the full dataset, then split it chronologically. What information crossed the boundary, and what is the corrected order?

**Exercise B.** A deep tree beats ridge on random folds but loses on the last three chronological blocks. What does this say, and what should you inspect before tuning?

**Answer key.** A: Full-sample means and variances used the later feature distribution. Split first; fit the scaler per training window; transform that window and its later validation block with the fitted scaler. B: The random folds do not represent future deployment, and the flexible tree may be unstable or regime-dependent. Audit date boundaries, label availability and overlap, compare per-date losses with uncertainty, examine tree complexity and drift, and verify the same dates/universe/cost assumptions. Do not choose the model by the best random-fold score.

**Checkpoint rubric (0–2 each):** baseline matches question; ridge penalty and scale understood; tree limits recognized; all learned preprocessing stays inside each train fold; metrics and comparison units are explicit. Score 8/10 with no zero on preprocessing or baseline before expanding model capacity.

## 2.7 Failure diagnosis

If ridge changes dramatically with feature units, check scaling and penalty conventions. If a tree dominates training but fails later, reduce its degrees of freedom and inspect date-block behavior before proposing more features. If all models tie the baseline, verify target construction, feature availability, and metric calculation. A null result can be a successful diagnostic: it prevents unsupported complexity from becoming a research conclusion.

For feature/validation leakage, see [Applied ML Chapter 5 — Validation, Leakage & Feature Pipelines](../applied-ml/ch5-validation-features-lesson.md). For signal versus portfolio evaluation, see [ML Quant Chapter 5 — Signal & Portfolio Evaluation](../ml-quant/ch5-signal-portfolio-evaluation-lesson.md). Next: [Chapter 3 — Honest Validation for Time-Ordered Data](ch3-honest-validation-lesson.md).

## References

- scikit-learn, [Linear models and Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html).
- scikit-learn, [Pipelines and leakage prevention](https://scikit-learn.org/stable/getting_started).
- scikit-learn, [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
