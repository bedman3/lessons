# Chapter 5 — Classical Inference: z, t, $\chi^2$, F, and Binomial Reasoning

Inference asks whether an observed estimate is large relative to the variation expected from sampling. Most classical test statistics have the same skeleton:

$$
\text{statistic}
=\frac{\text{estimate}-\text{reference value}}{\text{standard error under stated assumptions}}.
$$

The named distributions—normal, t, chi-square, and F—arise from different ways of estimating and comparing uncertainty.

## 5.1 z-score versus z-statistic

A **z-score** standardizes one observation using a population mean and standard deviation:

$$
z=\frac{x-\mu}{\sigma}.
$$

It asks where an observation lies within a distribution.

A **z-statistic** standardizes an estimator. For a sample mean with known population standard deviation,

$$
Z=\frac{\bar X-\mu_0}{\sigma/\sqrt n}.
$$

It asks how far an estimated mean lies from a reference value, measured in sampling standard errors. The numerator can look similar, but the denominator answers a different question.

## 5.2 Why Student's t exists

Usually $\sigma$ is unknown, so we substitute sample standard deviation $S$:

$$
T=\frac{\bar X-\mu_0}{S/\sqrt n}.
$$

If observations are iid Gaussian, then

$$
\frac{(n-1)S^2}{\sigma^2}\sim\chi^2_{n-1}
$$

and it is independent of the standardized sample mean. Dividing a standard normal by the square root of an independent chi-square variable per degree of freedom produces

$$
T\sim t_{n-1}.
$$

The heavier t tails account for uncertainty in the estimated denominator. With large $n$, $S$ becomes stable and the t distribution approaches normal.

## 5.3 Degrees of freedom

Degrees of freedom count independent pieces of variation after fitted constraints. For one-sample variance, residuals obey

$$
\sum_i(X_i-\bar X)=0,
$$

leaving $n-1$ independent residual directions. In regression with $p$ fitted coefficients and full-rank design, residual degrees of freedom are $n-p$.

Degrees of freedom are not a magical sample-size correction; they describe dimension lost to estimation.

## 5.4 Chi-square and F distributions

If $Z_1,\ldots,Z_\nu$ are independent standard normals,

$$
\sum_{j=1}^{\nu}Z_j^2\sim\chi^2_\nu.
$$

This makes chi-square natural for variances and squared discrepancies. Pearson's statistic for categorical counts is

$$
\chi^2=\sum_j\frac{(O_j-E_j)^2}{E_j}.
$$

An F variable is a ratio of independent scaled chi-square variables:

$$
F=\frac{U_1/\nu_1}{U_2/\nu_2}.
$$

F tests compare explained to unexplained variation or compare nested regression models. They are sensitive to assumptions when used as literal variance-ratio tests.

## 5.5 Confidence intervals

A 95% confidence procedure is constructed to cover the true parameter in 95% of repeated samples under the model. After observing one sample, the frequentist parameter is fixed; the interval either covers it or does not.

For a mean with unknown variance and Gaussian observations,

$$
\bar X\pm t_{0.975,n-1}\frac{S}{\sqrt n}.
$$

Interval width reflects sampling error under assumptions. It does not include data bias, model misspecification, measurement error, or future regime change unless the procedure explicitly models them.

## 5.6 Hypothesis tests and p-values

A test begins with a null hypothesis $H_0$, an alternative $H_1$, a test statistic, and a rejection rule chosen before looking at the result.

The **p-value** is

> the probability, assuming the null and test assumptions are true, of obtaining a statistic at least as incompatible with the null as the observed one.

It is not the probability that the null is true. It is also not an effect size or the probability that a result will replicate.

Type I error rejects a true null; Type II error fails to reject a false null. The test level $\alpha$ controls Type I error under the model. Power is

$$
1-\beta=P(\text{reject }H_0\mid\text{specified alternative is true}).
$$

## 5.7 Effect size, uncertainty, and practical significance

A large sample can make a tiny effect statistically significant. A small sample can leave an economically important effect uncertain. Report together:

- the estimated effect;
- an uncertainty interval;
- the decision-relevant scale;
- robustness to reasonable specifications.

For a strategy, a mean return different from zero may still be irrelevant after costs, unstable across regimes, or too uncertain in tail risk.

## 5.8 One-sample z example

Suppose a process has known standard deviation $\sigma=10$. From $n=100$ observations, $\bar X=52$, and the null mean is 50.

$$
Z=\frac{52-50}{10/\sqrt{100}}=2.
$$

The two-sided p-value is about 0.0455. The corresponding 95% interval is

$$
52\pm1.96(1)=[50.04,53.96].
$$

This calculation is valid only if the standard-error model is credible.

## 5.9 Welch's two-sample t example

For independent groups with means $\bar X_1,\bar X_2$, sample variances $S_1^2,S_2^2$, and sizes $n_1,n_2$, Welch's statistic is

$$
T=\frac{\bar X_1-\bar X_2}{\sqrt{S_1^2/n_1+S_2^2/n_2}}.
$$

If $\bar X_1=8.0$, $S_1=3$, $n_1=40$ and $\bar X_2=6.5$, $S_2=5$, $n_2=50$, then

$$
\operatorname{SE}=\sqrt{9/40+25/50}\approx0.851,
$$

so $T\approx1.76$. Welch's approximate degrees of freedom come from matching denominator uncertainty and are preferable to automatically assuming equal variances.

## 5.10 Binomial inference

If $X\sim\operatorname{Binomial}(n,p)$ and $x$ successes are observed, $\widehat p=x/n$. A simple Wald interval

$$
\widehat p\pm z_{0.975}\sqrt{\frac{\widehat p(1-\widehat p)}{n}}
$$

can behave poorly near 0 or 1 and at small $n$. Score/Wilson or exact methods are safer.

For $x=8$ successes among $n=20$, $\widehat p=0.4$. Testing $p_0=0.2$ exactly uses the binomial tail under $p_0$, not a normal approximation whose expected counts are marginal.

## 5.11 Multiple testing and selection

If 100 independent null hypotheses are tested at 5%, roughly five false positives are expected. Common responses include:

- Bonferroni control of family-wise error;
- Benjamini-Hochberg control of false discovery rate;
- held-out confirmation;
- hierarchical or shrinkage models;
- reporting the full search process rather than only winners.

Trying many features, horizons, universes, and hyperparameters creates the same problem even if only one final p-value is shown.

## 5.12 Decision map

| Question | Typical statistic | Central assumption to inspect |
|---|---|---|
| One mean, known population scale | z | Independence and known $\sigma$ |
| One mean, estimated scale | one-sample t | iid Gaussian exactly, or adequate asymptotics |
| Two independent means | Welch t | Independent groups and reliable within-group variance |
| Paired before/after mean | paired t on differences | Independent pairs; model the differences |
| One proportion | binomial/score test | Bernoulli trials with stable $p$ |
| Categorical count fit | chi-square | Adequate expected counts and independent cases |
| Nested regression comparison | F or likelihood ratio | Correct nesting and error model |
| Many simultaneous questions | multiplicity procedure | Family definition and selection process |

The test name comes after the data structure and estimand, not before.

## 5.13 Failure modes

- **Treating a p-value as $P(H_0\mid\text{data})$:** it conditions in the opposite direction.
- **Choosing one-sided tests after seeing the sign:** this invalidates the stated error rate.
- **Using paired data as independent:** the standard error is wrong.
- **Ignoring selection:** the reported test no longer represents the full experiment.
- **Equating non-significance with no effect:** uncertainty may simply be wide.
- **Treating robust standard errors as a cure-all:** they do not repair biased sampling, leakage, or wrong targets.

## 5.14 Knowledge checks

1. Explain the difference between a z-score and a z-statistic.
2. Why is a t reference distribution wider than a normal distribution for small samples?
3. What does a 95% confidence level mean operationally?
4. If 200 null signals are independently tested at 1%, how many false positives are expected on average?
5. A result has $p=0.001$ but a negligible effect size. What can and cannot be concluded?

### Solution outlines

1. A z-score locates one observation using population spread; a z-statistic locates an estimator using its sampling standard error.
2. The scale is estimated, so denominator uncertainty produces heavier tails.
3. Across repeated samples, 95% of intervals constructed by the procedure cover the fixed parameter under the assumptions.
4. $200(0.01)=2$.
5. The data are difficult to reconcile with the null model, but the practical value can still be negligible and assumptions or selection can still fail.

## 5.15 What to retain

- Classical statistics standardize an estimate by its sampling uncertainty.
- z, t, chi-square, and F arise from different uncertainty constructions.
- A p-value is conditional on the null; it is not a posterior probability.
- Effect size, interval width, power, and multiplicity matter alongside significance.
- The design and dependence structure determine the valid test.

Next: [Chapter 6 — Regression & the Modelling Bridge](ch6-regression-bridge-viewer.html) connects inference to prediction, ML losses, and quantitative models.
