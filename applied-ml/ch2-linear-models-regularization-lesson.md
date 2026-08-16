# Chapter 2 — Linear Models, Likelihood & Regularization

Linear models are the best laboratory for understanding machine learning. They make objectives, geometry, probability assumptions, regularization, and uncertainty visible. Deep networks reuse the same building blocks at larger scale.

## 2.1 Linear regression as projection

For feature vector $x\in\mathbb R^p$,

$$
\widehat y=\beta_0+x^T\beta.
$$

Ordinary least squares minimizes

$$
\sum_i(y_i-\beta_0-x_i^T\beta)^2.
$$

Geometrically it projects the target vector onto the column space of the design matrix. Statistically it estimates a conditional mean when $E[\varepsilon\mid X]=0$. Predictively it supplies a low-variance baseline whose extrapolation is easy to inspect.

Interactions and nonlinear basis functions can remain linear in parameters:

$$
\widehat y=\beta_0+\beta_1x+\beta_2x^2+\beta_3xz.
$$

“Linear model” refers to coefficients, not necessarily raw inputs.

## 2.2 Logistic regression

Binary probabilities must lie in $[0,1]$. Logistic regression makes log-odds linear:

$$
\log\frac{p(x)}{1-p(x)}=\beta_0+x^T\beta,
$$

so

$$
p(x)=\frac{1}{1+e^{-(\beta_0+x^T\beta)}}.
$$

A coefficient $\beta_j$ multiplies odds by $e^{\beta_j}$ for a one-unit feature increase, holding other features fixed. That is not automatically a causal effect.

Bernoulli negative log-likelihood gives cross-entropy:

$$
-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

The gradient for one example is $(p_i-y_i)x_i$: prediction error times input, the same pattern that propagates through neural networks.

## 2.3 Generalized linear models

A generalized linear model combines:

1. a response distribution from the exponential family;
2. a linear predictor $\eta=x^T\beta$;
3. a link relating $E[Y\mid X]$ to $\eta$.

Examples include logistic regression for Bernoulli outcomes and log-link Poisson regression for counts. The response distribution determines the variance structure and loss; the link makes constraints such as positivity natural.

## 2.4 Why maximum likelihood matters

Maximum likelihood chooses parameters that make observed data most probable under the model. Negative log-likelihood turns products into sums and produces familiar losses:

- Gaussian noise → squared error;
- Laplace noise → absolute error;
- Bernoulli outcome → cross-entropy;
- Poisson counts → Poisson deviance.

A loss therefore implies a probabilistic story even when used only for prediction. If its tail or variance assumptions are implausible, estimates and uncertainty may be fragile.

## 2.5 Ridge regularization

Ridge solves

$$
\min_\beta\left\{\frac1n\|y-X\beta\|_2^2+\lambda\|\beta\|_2^2\right\}.
$$

Its solution is

$$
\widehat\beta_{\text{ridge}}
=(X^TX+n\lambda I)^{-1}X^Ty.
$$

Adding $\lambda I$ stabilizes poorly determined directions. In the SVD basis, components with small singular values shrink most. Ridge shares weight among correlated predictors rather than choosing one arbitrarily.

## 2.6 Lasso and sparsity

Lasso uses an $L_1$ penalty:

$$
\min_\beta\left\{\frac1n\|y-X\beta\|_2^2+\lambda\|\beta\|_1\right\}.
$$

The diamond-shaped constraint has corners on coordinate axes, so optima often set coefficients exactly to zero. Sparsity can aid compression and interpretation, but correlated features make the selected member unstable.

Elastic net combines $L_1$ and $L_2$, retaining sparsity while stabilizing correlated groups.

## 2.7 Bayesian view of penalties

MAP estimation with Gaussian coefficient priors yields ridge; Laplace priors yield lasso. The penalty encodes a prior preference for smaller coefficients.

This equivalence clarifies why feature scale matters. A common penalty assumes coefficients are comparable a priori, so continuous features should usually be standardized within the training pipeline. The intercept is normally left unpenalized.

## 2.8 Bias–variance trade-off in one dimension

If noisy data weakly identify a coefficient, OLS may vary wildly across samples. Ridge pulls it toward zero:

- bias increases because estimates are systematically shrunk;
- variance decreases because the unstable direction is constrained;
- test MSE can fall if the variance reduction dominates.

The best $\lambda$ is a predictive choice selected on validation data, not the value that makes coefficients look appealing.

## 2.9 Optimization

For differentiable empirical loss $L(\beta)$, gradient descent updates

$$
\beta_{t+1}=\beta_t-\eta\nabla L(\beta_t).
$$

Convex linear and logistic objectives have no bad local minima, though conditioning affects speed. Standardization improves both optimization and regularization geometry. Newton and quasi-Newton methods use curvature to converge faster on moderate problems.

Perfect separation in logistic regression can send unregularized coefficients toward infinity even while classification looks perfect. Regularization creates a finite solution.

## 2.10 Coefficients versus predictions

A coefficient answers a conditional model question. A prediction combines all coefficients at a particular feature vector. With collinearity, many coefficient vectors can make nearly identical predictions.

For interpretation:

- state units and transformations;
- inspect uncertainty and stability;
- avoid comparing raw magnitudes across different scales;
- distinguish association from intervention;
- remember that regularized coefficients are deliberately biased.

For prediction, judge out-of-sample loss and calibration, not coefficient significance alone.

## 2.11 Worked model-selection example

Suppose 100 correlated signals predict a noisy return with only 300 observations.

- OLS can fit unstable offsetting coefficients.
- Lasso may choose one arbitrary signal from each correlated group.
- Ridge retains groups and stabilizes predictions.
- Elastic net offers a compromise if a sparse output is operationally valuable.

The appropriate choice follows validation under the real time structure and turnover cost—not a universal preference for sparsity.

## 2.12 Failure modes

- Fitting polynomial features without scaling, creating severe conditioning problems.
- Interpreting regularized coefficients as unbiased effect estimates.
- Performing standardization before cross-validation.
- Treating an odds ratio as a probability difference.
- Using lasso selection stability as evidence of truth.
- Extrapolating far outside the training range because the formula permits it.

## 2.13 Knowledge checks

1. Derive the logistic cross-entropy gradient with respect to the linear score.
2. Why does ridge especially shrink directions associated with small singular values?
3. Why can lasso choose unstable features among correlated predictors?
4. What probabilistic assumptions correspond to squared and absolute loss?
5. Why must scaling occur inside each training fold?

### Solution outlines

1. For score $z$, $d\ell/dz=\sigma(z)-y$.
2. Those directions are weakly identified by data, so the penalty is large relative to their curvature.
3. Several features provide nearly interchangeable fits, while the $L_1$ geometry favours a corner solution.
4. Gaussian and Laplace conditional noise respectively.
5. Full-data scaling leaks validation-distribution information into training.

## 2.14 What to retain

- Linear models expose the relationship among probability, loss, geometry, and optimization.
- Regularization trades bias for stability by encoding preferences over coefficients.
- Ridge, lasso, and elastic net behave differently under correlation.
- Predictive validation and coefficient inference answer different questions.
- Preprocessing is part of the learned pipeline.

Next: [Chapter 3 — Model Families & Inductive Bias](ch3-model-families-viewer.html).
