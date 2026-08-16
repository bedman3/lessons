# Chapter 6 — Regression & the Modelling Bridge

Regression is the bridge from probability and inference to machine learning and quantitative modelling. At its core it asks a conditional-expectation question:

$$
m(x)=E[Y\mid X=x].
$$

Linear regression approximates that function with a line or hyperplane. The same fitted equation can support explanation, uncertainty estimation, prediction, factor neutralization, or forecasting—but those goals require different assumptions and validation.

## 6.1 Population relationship versus fitted model

A population linear model is

$$
Y=X^T\beta+\varepsilon,
$$

where $E[\varepsilon\mid X]=0$ implies

$$
E[Y\mid X]=X^T\beta.
$$

Given observations collected in a design matrix $\mathbf X$ and response vector $\mathbf y$, ordinary least squares chooses

$$
\widehat\beta
=\arg\min_b\|\mathbf y-\mathbf Xb\|_2^2.
$$

$\beta$ is a population target under assumptions. $\widehat\beta$ is a random estimate computed from a sample. Fitted values are $\widehat{\mathbf y}=\mathbf X\widehat\beta$ and residuals are $\mathbf e=\mathbf y-\widehat{\mathbf y}$.

## 6.2 Geometry of least squares

The columns of $\mathbf X$ span the set of predictions the linear model can produce. Least squares projects $\mathbf y$ onto that column space. At the optimum, the residual is orthogonal to every design column:

$$
\mathbf X^T(\mathbf y-\mathbf X\widehat\beta)=0.
$$

These are the normal equations:

$$
\mathbf X^T\mathbf X\widehat\beta=\mathbf X^T\mathbf y.
$$

If $\mathbf X$ has full column rank,

$$
\widehat\beta=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y.
$$

In computation, do not explicitly form the inverse. QR or SVD solves are more stable and expose rank problems.

## 6.3 Worked simple regression

Suppose we fit $Y_i=\beta_0+\beta_1X_i+\varepsilon_i$. The slope is

$$
\widehat\beta_1
=\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}{\sum_i(X_i-\bar X)^2}
=\frac{\widehat{\operatorname{Cov}}(X,Y)}{\widehat{\operatorname{Var}}(X)}.
$$

The intercept is $\widehat\beta_0=\bar Y-\widehat\beta_1\bar X$.

For points $(1,2),(2,3),(3,5)$, $\bar X=2$ and $\bar Y=10/3$. The numerator is 3 and denominator is 2, so $\widehat\beta_1=1.5$ and $\widehat\beta_0=1/3$.

The slope is an association per unit $X$, conditional on included regressors. It is causal only under a credible identification argument.

## 6.4 What OLS guarantees—and what it does not

Under linearity in parameters, full rank, and $E[\varepsilon\mid X]=0$, OLS is unbiased conditional on $X$.

Under homoskedastic uncorrelated errors,

$$
\operatorname{Var}(\widehat\beta\mid X)
=\sigma^2(\mathbf X^T\mathbf X)^{-1}.
$$

The Gauss-Markov theorem says OLS has the smallest variance among linear unbiased estimators under its assumptions. It does not say:

- the relationship is causal;
- the model predicts well out of sample;
- errors are Gaussian;
- coefficients are stable over time;
- the chosen variables are appropriate.

Gaussian errors enable exact small-sample t and F inference. Large-sample inference can be approximate under weaker conditions.

## 6.5 Coefficient uncertainty

With $p$ fitted parameters, estimate residual variance by

$$
\widehat\sigma^2=\frac{\mathbf e^T\mathbf e}{n-p}.
$$

The usual estimated coefficient covariance is

$$
\widehat{\operatorname{Var}}(\widehat\beta)
=\widehat\sigma^2(\mathbf X^T\mathbf X)^{-1}.
$$

A coefficient t-statistic is

$$
t_j=\frac{\widehat\beta_j-\beta_{j,0}}{\operatorname{SE}(\widehat\beta_j)}.
$$

Small standard errors can come from plentiful independent information and good design variation. They can also be falsely small because of serial dependence, clusters, leakage, or specification search.

## 6.6 Residual diagnostics

Residuals are observed leftovers, not the unobservable true errors. Useful checks include:

- residual versus fitted plots for nonlinearity and changing variance;
- residuals versus time for drift and serial correlation;
- tail and quantile plots for outliers or heavy tails;
- leverage and influence diagnostics for observations dominating the fit;
- stability checks across periods, groups, and specifications.

A residual pattern is evidence about what the model has not captured. A clean-looking residual plot is not proof that every assumption holds.

## 6.7 Heteroskedasticity and autocorrelation

If $\operatorname{Var}(\varepsilon_i\mid X_i)$ changes across observations, OLS coefficients can remain unbiased under exogeneity, but the usual standard errors are wrong. Heteroskedasticity-consistent sandwich estimators modify the covariance calculation.

If errors are serially correlated, time-series robust estimators such as Newey-West incorporate lagged residual covariance. Cluster-robust estimators allow arbitrary dependence within predefined groups.

The correction must match the dependence mechanism. “Robust” is not a universal switch, and no covariance correction repairs endogenous regressors.

## 6.8 Multicollinearity and regularization

When design columns are nearly dependent, $\mathbf X^T\mathbf X$ is ill-conditioned. Predictions may remain reasonable while individual coefficients become unstable.

Ridge regression solves

$$
\widehat\beta_{\text{ridge}}
=\arg\min_b\left\{\|\mathbf y-\mathbf Xb\|_2^2+\lambda\|b\|_2^2\right\}.
$$

The penalty adds bias but can greatly reduce variance. Lasso uses $\lambda\|b\|_1$ and can set coefficients to zero. These are explicit examples of the bias-variance trade-off.

Standardization matters because penalties act on coefficient scale. The intercept is usually not penalized.

## 6.9 Maximum-likelihood connection

If

$$
Y_i\mid X_i\sim N(X_i^T\beta,\sigma^2)
$$

independently, the negative log-likelihood differs from squared error only by constants and scaling:

$$
-\log L(\beta)\propto\sum_i(Y_i-X_i^T\beta)^2.
$$

Thus OLS is maximum likelihood under Gaussian conditional noise. Different outcome distributions produce different losses.

For binary $Y$, logistic regression models

$$
P(Y=1\mid X=x)=\sigma(x^T\beta),\qquad
\sigma(z)=\frac{1}{1+e^{-z}},
$$

and Bernoulli likelihood produces cross-entropy loss. This is the direct bridge from statistical modelling to classification and neural-network training.

## 6.10 Prediction, explanation, and decision are different

- **Explanation:** what conditional association does a coefficient describe?
- **Prediction:** how well does the model generalize to new data?
- **Decision:** does acting on the prediction improve an objective after costs and constraints?

A stable causal coefficient can have weak predictive power. A strong prediction can rely on noncausal proxies. A predictive financial signal can be untradable after turnover and impact.

Train/test separation estimates generalization. It does not by itself establish causality or economic value.

## 6.11 Finance and ML connections

- Factor regression decomposes returns into common exposure and residual components.
- Beta hedging chooses positions so modelled factor exposure cancels.
- Cross-sectional regression estimates conditional relationships across assets at a time.
- Time-series regression faces autocorrelation, changing distributions, and temporal validation.
- Neural networks replace $X^T\beta$ with a flexible function but still optimize empirical loss.
- Regularization, validation, and uncertainty remain central even when the model is nonlinear.

## 6.12 Computational view

The normal equations explain the mathematics, but stable solvers should be used:

```python
# Conceptual pseudocode: QR-based least squares
Q, R = qr_decomposition(X)
beta_hat = solve_upper_triangular(R, transpose(Q) @ y)
residuals = y - X @ beta_hat
```

For an honest predictive estimate, preprocessing and model fitting must occur inside each training fold, never on the full dataset before splitting.

## 6.13 Failure modes

- **Causal language without identification:** conditioning on variables does not automatically remove confounding.
- **Inverting $X^TX$ directly:** numerical instability can dominate the calculation.
- **Reading high $R^2$ as useful prediction:** in-sample fit can come from leakage or overfitting.
- **Using iid errors for dependent data:** t-statistics become overstated.
- **Treating a residual as pure alpha:** it contains noise and any omitted structure.
- **Selecting features using the test set:** the test set becomes part of training.
- **Ignoring coefficient scale under regularization:** penalties become arbitrary.

## 6.14 Knowledge checks

1. Derive the normal equations by differentiating squared error.
2. Why can multicollinearity make coefficients unstable while predictions remain usable?
3. What assumption makes OLS coefficient estimates unbiased conditional on $X$?
4. Why does Gaussian conditional noise lead to squared-error loss?
5. Give one example of a model that predicts well but does not identify a causal effect.

### Solution outlines

1. The gradient is $-2X^T(y-Xb)$; setting it to zero gives $X^TXb=X^Ty$.
2. Several coefficient combinations can produce similar fitted values when columns are nearly dependent, so individual allocations are weakly identified.
3. Zero conditional mean: $E[\varepsilon\mid X]=0$.
4. The Gaussian log density is a constant minus squared residual divided by $2\sigma^2$.
5. A credit model may use a proxy correlated with repayment without changing that proxy being an intervention that changes repayment.

## 6.15 Course synthesis

The conceptual chain is now complete:

1. Probability defines uncertainty over events and random variables.
2. Distributions describe outcomes, moments, and tails.
3. Joint distributions describe dependence and conditional information.
4. Sampling distributions quantify estimator uncertainty.
5. Inference compares effects with that uncertainty.
6. Regression models conditional structure and connects statistics to prediction.

These tools are prerequisites, not isolated formulas. Later courses will repeatedly ask: What is random? What is conditioned on? What is estimated? Which sampling distribution justifies uncertainty? Which assumptions connect the fitted model to the intended decision?

Return to the [Probability & Statistics Foundations contents](index.html), or continue to the Applied Machine Learning course once available.
