# Statistics & Statistical Inference

> From data to decisions: estimators and their properties, maximum likelihood, the central limit theorem in action, confidence intervals, and hypothesis testing — the machinery that turns samples into claims.

## 1. Estimators and their properties

An **estimator** $\hat\theta$ is a rule for guessing a parameter $\theta$ from data. Three properties decide whether it is any good:

| Property | Definition | Intuition |
|---|---|---|
| **Unbiased** | $\mathbb{E}[\hat\theta] = \theta$ | On average, right. |
| **Low variance** | $\operatorname{Var}(\hat\theta)$ small | Stable across samples. |
| **Consistent** | $\hat\theta \to \theta$ as $n \to \infty$ | With enough data, exact. |
| **Efficient** | Achieves the smallest possible variance | Uses the data optimally. |

The **mean squared error** combines bias and variance into one score:

$$
\operatorname{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta)^2] = \operatorname{Bias}^2(\hat\theta) + \operatorname{Var}(\hat\theta)
$$

A biased estimator can beat an unbiased one if its variance is much smaller (this is the bias–variance trade-off, and the reason ridge/LASSO exist).

**Worked example — the sample variance.** For i.i.d. data with mean $\mu$:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar x)^2 \quad \text{is unbiased;}\qquad
\sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2 \quad \text{has bias } -\sigma^2/n
$$

The $n-1$ is not magic: the MLE uses the estimated mean, which absorbs some of the spread, so the division by $n$ understates variance by the factor $(n-1)/n$.

## 2. Maximum likelihood estimation

The **likelihood** is the probability (or density) of the observed data as a function of the parameter:

$$
L(\theta) = \prod_{i=1}^n f(x_i; \theta)
$$

The **MLE** maximises it — the parameter value under which the data was most likely:

$$
\hat\theta_{\text{MLE}} = \arg\max_\theta L(\theta) = \arg\max_\theta \sum_{i=1}^n \ln f(x_i; \theta)
$$

(The log is monotone and turns products into sums.)

**Worked example — normal.** $x_i \sim \mathcal N(\mu, \sigma^2)$ i.i.d.:

$$
\ln L = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2
$$

Differentiate: $\partial \ln L / \partial \mu = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \Rightarrow \hat\mu = \bar x$. Similarly $\hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar x)^2$ (the slightly biased one).

**Worked example — Bernoulli.** $x_i \in \{0,1\}$, $k$ successes in $n$ trials:

$$
\ln L = k\ln p + (n-k)\ln(1-p), \qquad \hat p = \frac{k}{n}
$$

**Invariance property:** the MLE of a function is the function of the MLE: $\widehat{g(\theta)} = g(\hat\theta)$. So the MLE of the odds $p/(1-p)$ is $\frac{k/n}{1 - k/n}$.

**Asymptotic normality:** with enough data, $\hat\theta_{\text{MLE}} \approx \mathcal N\big(\theta,\ \frac{1}{\mathcal I(\theta)}\big)$, where $\mathcal I(\theta)$ is the Fisher information — the MLE is the most efficient estimator in the limit.

## 3. The central limit theorem in inference

For i.i.d. data with mean $\mu$ and variance $\sigma^2$:

$$
\frac{\bar x - \mu}{\sigma/\sqrt{n}} \approx \mathcal N(0, 1)
$$

The standard error $\mathrm{se}(\bar x) = \sigma/\sqrt n$ is the key number: *uncertainty about the mean shrinks like $1/\sqrt n$.* To halve the uncertainty, quadruple the sample.

**Worked example.** Daily returns with $\sigma = 1.5\%$. The standard error of a 1-month (21-day) mean return is $1.5\%/\sqrt{21} \approx 0.33\%$; for a 1-year (252-day) mean it is $1.5\%/\sqrt{252} \approx 0.094\%$.

## 4. Confidence intervals

A 95% confidence interval for the mean (known $\sigma$):

$$
\bar x \pm z_{0.975}\frac{\sigma}{\sqrt n}, \qquad z_{0.975} \approx 1.96
$$

**The correct interpretation (a common point of confusion):** if you repeated the experiment many times, 95% of the constructed intervals would contain the true $\mu$. It is *not* "the probability that $\mu$ is in this interval is 95%" — $\mu$ is fixed, the interval is random.

**Worked example.** Sample of $n = 36$, $\bar x = 100$, known $\sigma = 15$:

$$
100 \pm 1.96 \cdot \frac{15}{6} = 100 \pm 4.9 = [95.1,\ 104.9]
$$

When $\sigma$ is unknown, replace it by $s$ and $z$ by the $t$-distribution with $n-1$ degrees of freedom — the $t$ is wider (fatter tails) to reflect the extra uncertainty of estimating $\sigma$. As $n$ grows, $t$ converges to the normal.

## 5. Hypothesis testing

**The setup:**

- **Null** $H_0$: the claim to be tested (e.g., $\mu = 0$).
- **Alternative** $H_1$: what you hope to show (e.g., $\mu \neq 0$).
- **Test statistic:** a number computed from data whose distribution under $H_0$ is known.
- **p-value:** the probability, *under $H_0$*, of seeing a statistic at least as extreme as observed.
- Decision: reject $H_0$ if the p-value is below the significance level $\alpha$ (typically 0.05).

**Worked example.** Test whether the mean return of a strategy is zero. $n = 100$ months, $\bar x = 0.8\%$, $s = 4\%$:

$$
t = \frac{\bar x}{s/\sqrt n} = \frac{0.008}{0.04/10} = 2.0
$$

Under $H_0$, $t \approx \mathcal N(0,1)$ (n large). The two-sided p-value is $P(|Z| \ge 2) = 2(1 - \Phi(2)) \approx 0.0456$. Since $0.0456 < 0.05$, we reject $H_0$: the strategy's mean is statistically distinguishable from zero at the 5% level. Note the phrasing — "statistically distinguishable", not "profitable".

**Errors:**

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Do not reject | ✓ | **Type II error** (miss) |
| Reject | **Type I error** (false alarm) | ✓ |

The significance level $\alpha$ is the Type I error rate; **power** $= 1 - P(\text{Type II})$ is the probability of detecting a true effect. Power grows with sample size and effect size and shrinks with noise:

$$
n \approx \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{(\mu_1 - \mu_0)^2}
$$

**Worked example.** To detect a mean return shift of $\mu_1 - \mu_0 = 0.5\%$ with $\sigma = 4\%$, 5% significance and 80% power: $n \approx (1.96 + 0.84)^2 (0.04)^2 / (0.005)^2 \approx 502$ observations. Designing the sample *before* running the study — rather than collecting data until significance appears — is the difference between inference and fishing.

## 6. Bayesian estimation in one breath

The Bayesian takes the prior seriously and updates:

$$
\underbrace{p(\theta \mid \text{data})}_{\text{posterior}} \propto \underbrace{p(\text{data} \mid \theta)}_{\text{likelihood}} \times \underbrace{p(\theta)}_{\text{prior}}
$$

The practical workhorse is the **conjugate pair**: pick a prior in the same family as the posterior and updating is just parameter arithmetic.

| Likelihood | Conjugate prior | Posterior |
|---|---|---|
| Bernoulli/Binomial | Beta(α, β) | Beta(α + k, β + n − k) |
| Poisson | Gamma(α, β) | Gamma(α + Σx, β + n) |
| Normal (known variance) | Normal | Normal (precision-weighted mean) |

**Worked example.** Prior $\text{Beta}(2, 8)$ for a coin's $p$ (2 heads, 8 tails of prior experience). Observe 30 heads in 100 flips. Posterior: $\text{Beta}(32, 78)$, posterior mean $\frac{32}{110} \approx 0.291$ — pulled from the sample mean 0.30 toward the prior mean 0.20. The posterior mean is a *shrinkage* estimator: a weighted average of prior and data.

## 7. Common traps

| Trap | The truth |
|---|---|
| "p < 0.05 means the effect is real" | p-value is about the data given the null, not the null given the data. Multiple testing, data snooping, and p-hacking inflate false discoveries. |
| "95% CI contains μ with 95% probability" | The interval is random; μ is fixed. Interpretation is about the procedure, not the draw. |
| "Uncorrelated means independent" | Uncorrelated is weaker. Dependent variables can be uncorrelated. |
| "More data fixes a biased estimator" | Bias does not vanish with n (only variance does). Unbiasedness is a property of the estimator, not the sample. |
| "The MLE is always unbiased" | No — the MLE of variance is biased (divides by n); consistency and unbiasedness are different. |
| "Significant means large" | With enough data, any non-zero effect becomes significant. Statistical significance ≠ economic importance. |

## 8. The toolkit in one page

| Task | Tool | Key formula |
|---|---|---|
| Estimate a parameter | MLE | maximise $\sum \ln f(x_i; \theta)$ |
| Judge an estimator | MSE | $\text{Bias}^2 + \text{Variance}$ |
| Measure uncertainty of a mean | Standard error | $\sigma/\sqrt n$ |
| Interval for a mean | CI | $\bar x \pm z \cdot \sigma/\sqrt n$ |
| Test a claim | p-value / t-stat | $t = \hat\theta / \mathrm{se}(\hat\theta)$ |
| Size a study | Power | $n \approx (z_{\alpha/2} + z_\beta)^2 \sigma^2 / \delta^2$ |
| Incorporate prior beliefs | Bayes / conjugate priors | posterior ∝ likelihood × prior |

## 9. Practice

1. For i.i.d. uniform $(0, \theta)$ data, show $\hat\theta = 2\bar x$ is unbiased. *(Answer: $E[\bar x] = \theta/2$, so $E[2\bar x] = \theta$.)*
2. A poll of 400 people finds 220 in favour. Construct a 95% CI for the true proportion. *(Answer: $\hat p = 0.55$, $\mathrm{se} = \sqrt{0.55 \cdot 0.45/400} = 0.0249$, so $0.55 \pm 1.96 \cdot 0.0249 = [0.501, 0.599]$.)*
3. A fund has 12 months of returns, mean $1\%$, std $3\%$. Test $H_0: \mu = 0$ at 5%. *(Answer: $t = 0.01/(0.03/\sqrt{12}) = 1.155$; $t_{0.975, 11} \approx 2.20$; fail to reject — 12 months is too little data to conclude.)*
4. Which is the tighter interval: $n = 100$ or $n = 400$, same data quality? *(Answer: $n = 400$ — the standard error halves when the sample quadruples.)*
5. An estimator is unbiased with variance 4; another is biased by 1 with variance 1. Which has lower MSE? *(Answer: biased one: MSE = 1 + 1 = 2 < 4.)*
