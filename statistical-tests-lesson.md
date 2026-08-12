# Statistical Tests: z, t, χ², F & The Binomial in Depth

> The test statistics that power quantitative work — what each measures, when to use it, and the distribution facts (including the binomial's mean and variance) that make the choice obvious.

## 1. Summary statistics: the vocabulary

Every test is built from a few descriptive numbers:

| Statistic | Definition | What it tells you |
|---|---|---|
| Mean | $\bar x = \frac{1}{n}\sum x_i$ | Centre of the data |
| Median | Middle value when sorted | Robust centre (survives outliers) |
| Variance | $s^2 = \frac{1}{n-1}\sum (x_i - \bar x)^2$ | Spread |
| Standard deviation | $s = \sqrt{s^2}$ | Spread in the data's own units |
| Skewness | $\frac{m_3}{m_2^{3/2}}$ | Asymmetry (negative = long left tail) |
| Excess kurtosis | $\frac{m_4}{m_2^2} - 3$ | Tail heaviness (0 for normal) |
| Covariance | $\frac{1}{n-1}\sum(x_i - \bar x)(y_i - \bar y)$ | Co-movement |
| Correlation | $r = \frac{\operatorname{Cov}(x,y)}{s_x s_y}$ | Co-movement, scaled to $[-1, 1]$ |

**The recurring pattern:** every test statistic below is a *signal-to-noise ratio* — the effect you care about divided by its standard error.

## 2. The binomial distribution, in depth

$X \sim \text{Binomial}(n, p)$ counts successes in $n$ independent trials, each with success probability $p$:

$$
\Pr(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

**The three facts to know cold:**

| Fact | Formula | Intuition |
|---|---|---|
| Mean | $\mathbb{E}[X] = np$ | Expected number of successes |
| Variance | $\operatorname{Var}(X) = np(1-p)$ | Largest at $p = 1/2$, zero at the edges |
| Standard deviation | $\sqrt{np(1-p)}$ | Fluctuation scale |

**Worked example.** $X \sim \text{Binomial}(100, 0.4)$: mean $= 40$, variance $= 100 \cdot 0.4 \cdot 0.6 = 24$, standard deviation $= \sqrt{24} \approx 4.899$. The mode is $\lfloor (n+1)p \rfloor = \lfloor 40.4 \rfloor = 40$.

**Why the variance peaks at $p = 1/2$:** when $p$ is near 0 or 1, outcomes are nearly certain; at $p = 1/2$ the process is maximally unpredictable.

### 2.1 The two approximations (know when each applies)

**Normal approximation** — good when $np \ge 5$ and $n(1-p) \ge 5$:

$$
X \approx \mathcal N\big(np,\ np(1-p)\big)
$$

**Worked example.** $P(X \le 45)$ for $\text{Binomial}(100, 0.4)$, using the continuity correction (extend the boundary by 0.5):

$$
P(X \le 45) \approx P\!\left(Z \le \frac{45.5 - 40}{4.899}\right) = P(Z \le 1.1227) \approx 0.869
$$

Exact: $P(X \le 45) = 0.8686$ — the approximation is off by 0.0004. The continuity correction is the difference between a good and a mediocre approximation.

**Poisson approximation** — good when $n$ is large and $p$ small (rare events), with $\lambda = np$:

$$
\text{Binomial}(n, p) \approx \text{Poisson}(np)
$$

**Worked example.** $X \sim \text{Binomial}(1000, 0.001)$: $P(X = 0) = 0.999^{1000} \approx 0.3677$; the Poisson approximation gives $e^{-1} \approx 0.3679$. Two large numbers, nearly identical answers.

## 3. The z-statistic: measuring in standard deviations

A **z-score** answers: *how many standard deviations is this value from the mean?*

**For a single observation:**

$$
z = \frac{x - \mu}{\sigma}
$$

**For a sample mean (the z-statistic):**

$$
z = \frac{\bar x - \mu_0}{\sigma / \sqrt n}
$$

The z-statistic is a standard normal under the null *when $\sigma$ is known* — it is the exact tool when you know the population spread (or have a large sample so $s \approx \sigma$).

**Worked example.** A daily return of $+2.5\%$ when the mean is $0.1\%$ and volatility is $1.5\%$:

$$
z = \frac{2.5 - 0.1}{1.5} = 1.6
$$

A 1.6-sigma event — mildly unusual (about 5.5% chance of being this extreme in either direction), not alarming. The z-score is the universal language of "how extreme is this?"

**Rule of thumb for the normal:** 68% within $\pm1\sigma$, 95% within $\pm1.96\sigma$, 99.7% within $\pm3\sigma$. These numbers come from the z-table and appear everywhere in risk reporting.

## 4. The t-statistic: same idea, estimated spread

The t-statistic replaces the known $\sigma$ with the estimated $s$:

$$
t = \frac{\bar x - \mu_0}{s / \sqrt n}, \qquad \text{degrees of freedom } \nu = n - 1
$$

**Why a different distribution?** Estimating $\sigma$ with $s$ adds uncertainty. The $t$-distribution has heavier tails than the normal, and the penalty is biggest for tiny samples:

| Degrees of freedom | $t_{0.975}$ | vs normal $z_{0.975} = 1.96$ |
|---|---|---|
| 1 | 12.71 | far wider |
| 5 | 2.571 | noticeably wider |
| 10 | 2.228 | wider |
| 30 | 2.042 | close |
| ∞ | 1.96 | identical |

As $\nu \to \infty$, $t \to z$. The practical rule: **use $t$ whenever $\sigma$ is estimated** — which is almost always in real work. Many production pipelines use $z$ only for very large samples where the distinction is negligible.

**Worked example (one-sample).** 12 monthly returns, mean $1\%$, $s = 3\%$; test whether the mean is zero:

$$
t = \frac{0.01}{0.03/\sqrt{12}} = 1.155, \qquad \text{critical } t_{0.975, 11} = 2.20
$$

$1.155 < 2.20$ — fail to reject. Twelve months cannot distinguish a 1% monthly mean from zero.

**Worked example (two-sample, Welch).** Fund A: $\bar x_1 = 1.2\%$, $s_1 = 3.0\%$, $n_1 = 60$; Fund B: $\bar x_2 = 0.8\%$, $s_2 = 2.5\%$, $n_2 = 60$:

$$
t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}} = \frac{0.4}{\sqrt{0.150 + 0.104}} = \frac{0.4}{0.504} = 0.793
$$

The Welch degrees of freedom:

$$
\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1 - 1} + \frac{(s_2^2/n_2)^2}{n_2 - 1}} \approx 114
$$

p-value $\approx 0.43$ — the 0.4% gap between the funds is well within sampling noise. Two funds differing by 0.4% with these volatilities cannot be distinguished with 60 observations each.

**Paired test** (same assets, two treatments): $t = \bar d / (s_d / \sqrt n)$ on the *differences* — pairing removes the between-observation variance and is far more powerful than an unpaired comparison when the pairs are correlated.

## 5. z vs t: the decision in one table

| | z-statistic | t-statistic |
|---|---|---|
| Uses | Known $\sigma$ (or very large $n$) | Estimated $s$ |
| Null distribution | $\mathcal N(0, 1)$ | $t_\nu$, $\nu = n - 1$ (or Welch $\nu$) |
| Tails | Fixed | Heavier for small $\nu$ |
| Example | "How many sigmas is this return?" | "Is this fund's mean different from zero?" |
| When to prefer | Risk reporting, large samples | Everything with estimated variance |

The z is for *calibrating* single observations against a known scale; the t is for *inference* about parameters you estimated. When in doubt, t.

## 6. The chi-square statistic: deviations from expected counts

The χ² statistic measures how far observed counts sit from expected counts:

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

**Goodness-of-fit example.** A die rolled 60 times gives counts $(8, 9, 12, 11, 10, 10)$. Expected: 10 each.

$$
\chi^2 = \frac{4 + 1 + 4 + 1 + 0 + 0}{10} = 1.0, \qquad \text{df} = 6 - 1 = 5
$$

$p$-value $\approx 0.96$ — the die looks fair; deviations this size are entirely typical. (A suspicious test would be χ² above roughly 11 with 5 df.)

**Independence (contingency) test:** same formula on a table of counts, with df $= (r-1)(c-1)$. If default rates differ across sectors, the χ² test formalizes "how unlikely is this table under independence?" The χ² distribution has mean $\nu$ and variance $2\nu$ — a quick sanity anchor.

## 7. The F-statistic: comparing variances

The F-statistic is a ratio of variances:

$$
F = \frac{s_1^2}{s_2^2}, \qquad \text{df} = (n_1 - 1,\ n_2 - 1)
$$

**Worked example.** Compare two volatility estimates: $s_1 = 4.5\%$ (31 obs), $s_2 = 3.5\%$ (41 obs):

$$
F = \frac{4.5^2}{3.5^2} = \frac{20.25}{12.25} = 1.653, \qquad F_{0.975}(30, 40) \approx 1.94
$$

$1.653 < 1.94$ — no significant difference. Volatility estimates from 31 vs 41 observations are too noisy to separate at 4.5% vs 3.5%.

F also powers ANOVA (comparing several group means via between-group vs within-group variance) and the overall significance of a regression (the model's explained variance vs residual variance). The one-way ANOVA F is:

$$
F = \frac{\text{between-group variance} / (k-1)}{\text{within-group variance} / (n-k)}
$$

## 8. Correlation and its test

The Pearson correlation measures linear association; Spearman's rank correlation measures monotonic association (robust to outliers and non-linearity).

**Is a correlation significant?** Test $H_0: \rho = 0$ with:

$$
t = \frac{r\sqrt{n - 2}}{\sqrt{1 - r^2}}, \qquad \text{df} = n - 2
$$

**Worked example.** $r = 0.7$ from $n = 30$ pairs:

$$
t = \frac{0.7\sqrt{28}}{\sqrt{0.51}} = \frac{3.703}{0.714} = 5.19
$$

With 28 df, $|t| > 2.05$ is significant at 5% — a correlation of 0.7 on 30 points is highly significant. **But:** significance is not causation, and with $n = 30$ any spurious correlation from a few influential points can reach 0.7. Always look at the scatterplot.

**The famous cautions:**

- Correlation measures linear (or monotonic) association, not dependence in general — two variables can be perfectly related non-linearly with $r = 0$.
- $r = 0.7$ on 10 points means almost nothing (df = 8, critical $t_{0.975,8} = 2.306$, $t = 0.7\sqrt8/\sqrt{0.51} = 2.77$ — still "significant" but fragile); on 1,000 points it is overwhelming. Sample size is part of the statistic.

## 9. Choosing a test: the decision map

| Question | Statistic | Distribution |
|---|---|---|
| Is a mean different from zero? (σ known) | $z$ | normal |
| Is a mean different from zero? (σ estimated) | $t$ | $t_{n-1}$ |
| Are two means different? | $t$ (Welch) | $t_{\nu}$ |
| Is a count distribution fair? | $\chi^2$ | $\chi^2_{k-1}$ |
| Are two variables independent in a table? | $\chi^2$ | $\chi^2_{(r-1)(c-1)}$ |
| Are two variances different? | $F$ | $F_{(n_1-1, n_2-1)}$ |
| Is a correlation non-zero? | $t$ | $t_{n-2}$ |
| Are $k$ group means equal? | $F$ (ANOVA) | $F_{(k-1, n-k)}$ |

The whole table is the same shape: *effect ÷ standard error, compared against a known distribution.*

## 10. Common confusions

| Confusion | The truth |
|---|---|
| "t and z are interchangeable" | Only for large $n$. For $n < 30$ the t's wider tails matter. |
| "The p-value is the probability the null is true" | It is the probability of the data under the null. Different quantity entirely. |
| "$P(X \le 45)$ with a normal approximation needs no correction" | Without the continuity correction the approximation is noticeably worse; with it, it is excellent. |
| "Binomial mean is $n$" | Mean is $np$; variance is $np(1-p)$. $n$ alone is the number of trials. |
| "More tests = more confidence" | Every extra test multiplies the chance of a false positive. Adjust (Bonferroni, FDR) or pre-register. |
| "Significant difference = big difference" | With $n = 10{,}000$, a 0.01% difference is "significant". Look at effect size. |
| "Variance of the mean is $\sigma^2$" | It is $\sigma^2/n$ — the standard error shrinks with $\sqrt n$. |

## 11. Practice

1. $X \sim \text{Binomial}(50, 0.2)$: mean and standard deviation? *(Answer: mean $= 10$, sd $= \sqrt{50 \cdot 0.2 \cdot 0.8} = \sqrt8 \approx 2.83$.)*
2. A return of $+3.2\%$ with mean $0.2\%$ and vol $1.5\%$: z-score? *(Answer: $z = 3.0/1.5 = 2.0$ — a 2-sigma event, ~4.6% two-sided.)*
3. 25 returns, mean $0.5\%$, $s = 2.5\%$. Is the mean distinguishable from zero at 5%? *(Answer: $t = 0.5/(2.5/5) = 1.0$; critical $t_{0.975,24} = 2.06$; no.)*
4. A coin flipped 200 times shows 120 heads. Using the normal approximation with continuity correction, is this surprising? *(Answer: $X \sim \mathcal N(100, 50)$, $z = (119.5 - 100)/\sqrt{50} = 2.76$; p ≈ 0.006 — surprising at 1%. Note the correction: without it, $z = 2.83$.)*
5. Which statistic would you use to test whether default rates are independent of sector? *(Answer: χ² test on the contingency table, df = (r−1)(c−1).)*
6. Two strategies have return volatilities $s_1 = 6\%$ (26 obs) and $s_2 = 4\%$ (31 obs). Is one significantly riskier? *(Answer: $F = 36/16 = 2.25$; critical $F_{0.975}(25, 30) \approx 2.12$; marginally significant at 5% — worth checking with more data.)*
