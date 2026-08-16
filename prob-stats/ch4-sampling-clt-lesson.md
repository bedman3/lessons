# Chapter 4 — Samples, Estimators, the LLN & the CLT

We rarely observe an entire population. We observe a sample and use it to estimate something about the process that generated it. Statistics is therefore probability run in reverse: probability starts with a model and predicts data; inference starts with data and reasons about the model.

The central challenge is not calculating an estimate. It is understanding how much that estimate would change under repeated sampling.

## 4.1 Population, sample, statistic, and estimator

- A **population** is the process or collection we want to understand.
- A **sample** is the observed data $X_1,\ldots,X_n$.
- A **parameter** is a fixed but unknown population quantity, such as $\mu$.
- A **statistic** is any function of the sample that contains no unknown parameter.
- An **estimator** is a statistic used to estimate a parameter.
- An **estimate** is the numerical value produced for one realized sample.

The sample mean

$$
\bar X=\frac{1}{n}\sum_{i=1}^nX_i
$$

is an estimator before observing data and an estimate after values are inserted. Across hypothetical repeated samples, $\bar X$ is random even though the population mean $\mu$ is fixed.

This distinction separates the **population distribution** of individual $X_i$ from the **sampling distribution** of $\bar X$.

## 4.2 The empirical distribution

The empirical CDF places mass $1/n$ on each observation:

$$
\widehat F_n(x)=\frac{1}{n}\sum_{i=1}^n\mathbf 1\{X_i\le x\}.
$$

It estimates the full distribution rather than one parameter. Sample means, quantiles, tail rates, and other plug-in estimates are summaries of this empirical distribution.

Observed data are not automatically representative. Sampling bias, timestamp mistakes, survivorship filters, and dependence can matter more than the estimator formula.

## 4.3 Properties of estimators

For an estimator $\widehat\theta$ of parameter $\theta$:

- **Bias:** $E[\widehat\theta]-\theta$.
- **Variance:** how much $\widehat\theta$ changes across samples.
- **Mean-squared error:**

$$
E[(\widehat\theta-\theta)^2]
=\operatorname{Var}(\widehat\theta)+\operatorname{Bias}(\widehat\theta)^2.
$$

- **Consistency:** $\widehat\theta$ converges in probability to $\theta$ as $n\to\infty$.
- **Efficiency:** among comparable estimators, lower variance or MSE.
- **Robustness:** limited sensitivity to outliers or modest model violations.

Unbiased is not always best. A slightly biased shrinkage estimator can have much lower variance and therefore lower prediction error.

## 4.4 Why sample variance uses $n-1$

The natural-looking variance estimate

$$
\frac{1}{n}\sum_{i=1}^n(X_i-\bar X)^2
$$

is biased downward because the same data choose $\bar X$, making residuals artificially small. The unbiased estimator is

$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2.
$$

Only $n-1$ residuals are free: once $n-1$ deviations and their required zero sum are known, the last is determined. This is the first appearance of **degrees of freedom**.

## 4.5 Sampling distribution and standard error

Assume $X_1,\ldots,X_n$ are independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$. Then

$$
E[\bar X]=\mu
$$

and

$$
\operatorname{Var}(\bar X)
=\operatorname{Var}\left(\frac1n\sum_iX_i\right)
=\frac{1}{n^2}n\sigma^2
=\frac{\sigma^2}{n}.
$$

The standard deviation of an estimator's sampling distribution is its **standard error**:

$$
\operatorname{SE}(\bar X)=\frac{\sigma}{\sqrt n}.
$$

Because $\sigma$ is usually unknown, we estimate it with $S/\sqrt n$.

Standard deviation describes variation among observations. Standard error describes variation among estimates. Doubling $n$ does not halve standard error; reducing it by half requires four times as much independent data.

## 4.6 Law of large numbers versus central limit theorem

The law of large numbers (LLN) says, under suitable conditions,

$$
\bar X_n\xrightarrow{p}\mu.
$$

It answers: **Where does the sample average go?**

The central limit theorem (CLT) says, under stronger regularity conditions such as finite variance in the classical iid form,

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\xrightarrow{d}N(0,1).
$$

It answers: **What is the shape and scale of the remaining error?**

The LLN gives convergence. The CLT gives an approximate distribution for normalized error. Neither says the raw data become Gaussian.

## 4.7 Why the square root appears

Independent variances add. A sum of $n$ observations therefore has variance $n\sigma^2$ and standard deviation $\sqrt n\sigma$. Dividing the sum by $n$ to form an average gives scale

$$
\frac{\sqrt n\sigma}{n}=\frac{\sigma}{\sqrt n}.
$$

This square-root law appears in sampling error, Monte Carlo convergence, diffusion scaling, and signal averaging.

## 4.8 Sample proportions are sample means

For Bernoulli indicators $X_i$, the sample proportion is

$$
\widehat p=\frac1n\sum_iX_i.
$$

Therefore

$$
E[\widehat p]=p,\qquad
\operatorname{SE}(\widehat p)=\sqrt{\frac{p(1-p)}{n}}.
$$

The Gaussian approximation works poorly when expected successes or failures are too small. Exact binomial or carefully chosen interval methods are then preferable.

## 4.9 Dependence changes effective sample size

For dependent observations,

$$
\operatorname{Var}(\bar X)
=\frac{1}{n^2}\left(\sum_i\operatorname{Var}(X_i)+2\sum_{i<j}\operatorname{Cov}(X_i,X_j)\right).
$$

Positive serial correlation makes the mean noisier than $\sigma^2/n$ suggests. A time series with 1,000 highly correlated daily observations may contain far less than 1,000 independent observations' worth of information.

For a weakly stationary series with autocorrelations $\rho_k$, a rough large-sample effective size is

$$
n_{\mathrm{eff}}\approx
\frac{n}{1+2\sum_{k\ge1}\rho_k}.
$$

This is why random train/test splitting and iid standard errors can be dangerously optimistic for financial data.

## 4.10 Delta-method intuition

Suppose $\widehat\theta$ is approximately normal around $\theta$, and we want uncertainty for $g(\widehat\theta)$. A first-order Taylor expansion gives

$$
g(\widehat\theta)\approx g(\theta)+g'(\theta)(\widehat\theta-\theta).
$$

Thus

$$
\operatorname{Var}(g(\widehat\theta))
\approx [g'(\theta)]^2\operatorname{Var}(\widehat\theta).
$$

The delta method propagates uncertainty through a smooth transformation. It breaks down near non-smooth points or when first-order approximation is poor.

## 4.11 Bootstrap intuition

The nonparametric bootstrap approximates repeated sampling by resampling the observed data with replacement:

1. draw $n$ observations from the sample with replacement;
2. recompute the statistic;
3. repeat many times;
4. examine the distribution of bootstrap statistics.

```python
import random

def bootstrap_statistic(sample, statistic, repetitions=2000):
    n = len(sample)
    return [
        statistic([random.choice(sample) for _ in range(n)])
        for _ in range(repetitions)
    ]
```

The ordinary bootstrap assumes observations are exchangeable. Time series require block or model-based resampling to preserve dependence.

## 4.12 Monte Carlo error

To estimate $\theta=E[g(X)]$, simulate $X_1,\ldots,X_n$ and use

$$
\widehat\theta_n=\frac1n\sum_{i=1}^ng(X_i).
$$

Under the classical CLT, its error is approximately

$$
\widehat\theta_n-\theta\approx N\left(0,\frac{\operatorname{Var}(g(X))}{n}\right).
$$

Monte Carlo therefore converges at $n^{-1/2}$ regardless of dimension, but high payoff variance can make the constant painfully large. Variance reduction improves that constant rather than the basic rate.

## 4.13 Failure modes

- **LLN means a finite sample is accurate:** convergence is asymptotic, not a guarantee at a chosen $n$.
- **CLT means observations are Gaussian:** it concerns normalized sums or averages.
- **Large $n$ fixes sampling bias:** more biased data estimate the wrong target more precisely.
- **Rows equal independent observations:** time, clusters, and repeated entities reduce effective size.
- **Standard error equals standard deviation:** one concerns estimator uncertainty, the other data dispersion.
- **Bootstrap always works:** dependence, extreme quantiles, and non-smooth estimators require care.

## 4.14 Knowledge checks

1. A population has standard deviation 12. What is the standard error of the mean for $n=36$ independent observations?
2. Explain why 10,000 biased observations can be worse than 1,000 representative observations.
3. State separately what the LLN and CLT say.
4. If doubling sample size reduces standard error by what factor? How much data halves it?
5. Why does positive serial correlation increase uncertainty in the sample mean?

### Solution outlines

1. $12/\sqrt{36}=2$.
2. Sample size reduces random sampling variation, not systematic mismatch between sample and target population.
3. LLN: the average converges to its expectation. CLT: its normalized error approaches a Gaussian law under suitable conditions.
4. The factor is $1/\sqrt2$; four times the data halves it.
5. Positive covariance terms add to the variance of the average, reducing effective independent information.

## 4.15 What to retain

- Parameters are fixed unknowns; estimators vary across hypothetical samples.
- Standard error describes the sampling variation of an estimator.
- LLN explains convergence; CLT explains the approximate distribution of remaining error.
- Independence assumptions determine how quickly information accumulates.
- More data reduce variance but do not repair bias, leakage, or a wrong target.

Next: [Chapter 5 — Classical Inference](ch5-classical-inference-viewer.html) turns sampling distributions into intervals and hypothesis tests.
