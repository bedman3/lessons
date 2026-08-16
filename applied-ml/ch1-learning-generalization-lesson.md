# Chapter 1 — Learning, Risk & Generalization

Machine learning is not the art of fitting data. It is the problem of using finite past observations to make useful decisions on future observations. A model that memorizes its training set has solved an optimization problem but may have learned nothing that generalizes.

## 1.1 The learning problem

Let $(X,Y)\sim P$ be an input–target pair from an unknown data-generating distribution. A model $f\in\mathcal F$ maps inputs to predictions. A loss $\ell(Y,f(X))$ says how costly a prediction is.

The population risk is

$$
R(f)=E_{(X,Y)\sim P}[\ell(Y,f(X))].
$$

Because $P$ is unknown, training minimizes empirical risk on observations $(x_i,y_i)$:

$$
\widehat R_n(f)=\frac1n\sum_{i=1}^n\ell(y_i,f(x_i)).
$$

The gap $R(f)-\widehat R_n(f)$ is the **generalization gap**. Training loss is observable; future risk is the target.

## 1.2 Loss functions encode decisions

Squared loss penalizes large errors heavily and makes the conditional mean optimal:

$$
f^*(x)=E[Y\mid X=x].
$$

Absolute loss makes a conditional median optimal. Bernoulli cross-entropy is minimized by the true conditional probability $P(Y=1\mid X=x)$. Ranking losses, asymmetric losses, and quantile loss target different decisions.

Choosing a loss is therefore a modelling decision, not a software default. A fraud model with asymmetric false-negative cost may need probability estimation plus a decision threshold, not accuracy maximization.

## 1.3 Three sources of learning error

A useful decomposition separates:

1. **Approximation error:** the best model in $\mathcal F$ cannot represent the true relationship.
2. **Estimation error:** finite data do not identify the best member of $\mathcal F$ exactly.
3. **Optimization error:** training fails to reach the best empirical solution.

A richer model class can reduce approximation error while increasing estimation and optimization difficulty. Better optimization cannot repair a poor target, leakage, or insufficient information.

## 1.4 Bias and variance

For squared error at a fixed $x$, prediction error can be written conceptually as

$$
E[(Y-\widehat f(x))^2]
=\sigma_\varepsilon^2
+\operatorname{Bias}(\widehat f(x))^2
+\operatorname{Var}(\widehat f(x)).
$$

- Irreducible noise $\sigma_\varepsilon^2$ remains even with the true conditional mean.
- Bias measures systematic model mismatch.
- Variance measures sensitivity to the training sample.

Regularization, bagging, more representative data, and smaller model classes usually reduce variance. Richer features and models can reduce bias. Modern models can complicate the classical U-shaped curve, but the underlying question—how sensitive is the learned rule to finite data?—remains essential.

## 1.5 Capacity and inductive bias

A hypothesis class has high capacity if it can express many labelings or functions. Capacity is useful only with an **inductive bias**: assumptions that prefer some solutions over others.

- Linear models prefer additive linear relationships.
- k-nearest neighbours assumes nearby inputs have similar targets.
- Trees prefer piecewise-constant axis-aligned rules.
- CNNs encode locality and translation structure.
- Regularization prefers smaller or smoother solutions.

There is no assumption-free learner. “Flexible” means the assumptions are weaker or encoded differently, not absent.

## 1.6 Why empirical risk can mislead

Suppose a class can memorize any training labels. It can achieve zero empirical error even when labels are random. Generalization requires restricting effective capacity relative to information in the sample.

Learning theory formalizes this with quantities such as VC dimension, Rademacher complexity, stability, and margin. A schematic bound has the form

$$
R(f)\lesssim \widehat R_n(f)
+\text{complexity penalty}
+\sqrt{\frac{\log(1/\delta)}{n}}
$$

with probability at least $1-\delta$. The constants and assumptions matter, so bounds are usually conceptual guides rather than model-selection calculators.

## 1.7 Validation estimates future risk

Training data choose parameters. Validation data choose models and hyperparameters. Test data estimate performance after choices are frozen.

Every time results influence another choice, those results become part of training. Repeatedly checking the test set causes adaptive overfitting even if its rows never enter gradient descent.

Cross-validation reuses limited data by rotating held-out folds. It estimates performance for the entire pipeline, not just the final estimator. Preprocessing performed before the split leaks validation information.

## 1.8 Distribution shift

Classical generalization assumes future examples follow the training distribution. Real systems face:

- **covariate shift:** $P(X)$ changes;
- **label shift:** $P(Y)$ changes;
- **concept drift:** $P(Y\mid X)$ changes;
- **selection shift:** observed data arise through a changed sampling mechanism.

A low iid validation error cannot prove robustness to a future regime absent from the sample. Time-based, group-based, and stress validation should match deployment.

## 1.9 Baselines and learning curves

A baseline defines what “learning” means. Useful baselines include a constant predictor, a simple linear model, the previous value, or an established business rule.

Learning curves plot training and validation error against sample size:

- both high and close: likely bias, weak features, or noisy target;
- training low, validation much higher: likely variance or leakage;
- validation still improving with data: more representative data may help;
- both implausibly strong: inspect leakage and duplicated entities.

## 1.10 Worked example: polynomial regression

Fit noisy observations from a smooth curve with polynomial degrees 1, 4, and 20.

- Degree 1 underfits: training and validation errors are high.
- Degree 4 captures the shape: both errors are moderate and close.
- Degree 20 can interpolate training points: training error is tiny, validation error unstable.

The lesson is not “degree 4 is best.” It is that model complexity must be selected using unseen data generated like the intended future.

## 1.11 Failure modes

- Optimizing a convenient metric that does not match the decision.
- Calling training accuracy “model performance.”
- Treating random rows as independent when entities or time periods repeat.
- Adding complexity to solve noisy labels or target ambiguity.
- Assuming a generalization theorem applies after adaptive data reuse.
- Comparing models without a strong simple baseline.

## 1.12 Knowledge checks

1. Distinguish approximation, estimation, and optimization error.
2. Why does squared loss target a conditional mean while absolute loss targets a median?
3. A model has zero training loss and poor validation loss. Name three different possible causes.
4. When does a test set stop being a test set?
5. Why is “the model makes no assumptions” impossible?

### Solution outlines

1. Respectively: class mismatch, finite-sample uncertainty, and failure to find the empirical optimum.
2. The mean minimizes expected squared deviations; any conditional median minimizes expected absolute deviations.
3. Excess capacity, leakage-free but small data, distribution mismatch, or unstable optimization can all contribute.
4. Once its results influence model, feature, threshold, or stopping choices.
5. Generalizing beyond observed cases requires preferring some unseen behaviours over others—an inductive bias.

## 1.13 What to retain

- Population risk is the target; empirical risk is a sample proxy.
- Loss functions specify what prediction quality means.
- Generalization depends on data, capacity, inductive bias, and deployment shift.
- Validation must cover the full adaptive pipeline.
- A simple baseline is evidence, not an embarrassment.

Next: [Chapter 2 — Linear Models & Regularization](ch2-linear-models-regularization-viewer.html).
