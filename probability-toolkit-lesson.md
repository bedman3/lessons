# Probability Toolkit: Distributions & Expectation

> The working knowledge that keeps coming back: the core distributions and their stories, moment generating functions, and the expectation tricks that turn hard problems into quick arithmetic.

## 1. The distribution zoo: know the story, not just the formula

Every distribution models a specific data-generating story. If you know the story, you can pick the right model and recall the key facts.

### 1.1 Discrete distributions

| Distribution | PMF | Mean | Variance | Story |
|---|---|---|---|---|
| Bernoulli(p) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | One coin flip. |
| Binomial(n, p) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $n$ independent flips; $k$ heads. |
| Poisson(λ) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Count of rare events in a fixed window (defaults, trades, jumps). |
| Geometric(p) | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | Flips until the first success. Memoryless. |
| Categorical / Multinomial | — | — | — | One roll of a $K$-sided die / $n$ rolls. |

### 1.2 Continuous distributions

| Distribution | PDF (or key fact) | Mean | Variance | Story |
|---|---|---|---|---|
| Uniform(a, b) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | Every value equally likely. |
| Exponential(λ) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Time until the next event of a Poisson process. Memoryless. |
| Normal(μ, σ²) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Sum of many small independent effects (CLT). |
| Log-normal(μ, σ²) | $X = e^Y$, $Y \sim \mathcal N(\mu,\sigma^2)$ | $e^{\mu + \sigma^2/2}$ | $e^{2\mu+\sigma^2}(e^{\sigma^2}-1)$ | Asset prices; strictly positive, right-skewed. |
| Student-t(ν) | — | 0 (ν>1) | $\nu/(\nu-2)$ (ν>2) | Heavy-tailed returns; fat tails. |
| Beta(α, β) | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | — | Distribution on (0,1); beliefs about a probability. |
| Gamma(α, β) | $\propto x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Waiting time for α events; conjugate for Poisson rate. |

**The two facts to internalise:** the exponential is the *only* memoryless continuous distribution (the time already waited tells you nothing about the remaining wait), and the log-normal mean carries the famous $\sigma^2/2$ correction: $E[e^Y] = e^{\mu + \sigma^2/2}$.

## 2. Moment generating functions: the distribution's ID card

The MGF is defined as:

$$
M_X(t) = \mathbb{E}[e^{tX}]
$$

**Why it matters:** the MGF uniquely identifies the distribution, and its derivatives at zero give the moments:

$$
M_X'(0) = \mathbb{E}[X], \qquad M_X''(0) = \mathbb{E}[X^2], \qquad \mathbb{E}[X^k] = M_X^{(k)}(0)
$$

**Worked examples:**

- Bernoulli: $M(t) = 1 - p + p e^t$. Check: $M'(0) = p = E[X]$ ✓.
- Normal: $M(t) = e^{\mu t + \frac{1}{2}\sigma^2 t^2}$. Setting $\mu = 0, \sigma = 1$: $M''(0) = 1 = E[X^2]$ ✓ (and $E[X^4] = 3$ — the famous kurtosis of the normal).
- Exponential: $M(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$. $M'(0) = 1/\lambda$ ✓.

**The superpower — sums of independent variables:**

$$
M_{X+Y}(t) = M_X(t)\,M_Y(t) \quad \text{(independent } X, Y)
$$

Because expectation of a product of independent variables is the product of expectations. Consequences:

- Sum of $n$ independent Poissons with rates $\lambda_i$: Poisson with rate $\sum \lambda_i$. *(Check: $e^{\lambda_i(e^t - 1)}$ multiplied over $i$.)*
- Sum of independent normals: normal with mean and variance added: $\mathcal N(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$.
- The log-normal mean: $E[e^Y] = M_Y(1) = e^{\mu + \sigma^2/2}$.

## 3. Expectation tricks that do the heavy lifting

### 3.1 Linearity (no independence needed)

$$
\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]
$$

Linearity always holds — it is the single most useful identity in probability.

### 3.2 When does $E[XY] = E[X]E[Y]$?

Only if $X$ and $Y$ are **independent** (or merely uncorrelated — and independence is strictly stronger). A classic trap:

**Example.** Let $X$ be uniform on $\{-1, 0, 1\}$ and $Y = X^2$. Then $E[X] = 0$, so $E[X]E[Y] = 0$, but $E[XY] = E[X^3] = 0$ too — uncorrelated yet clearly dependent ($Y$ is a function of $X$). Correlation zero does not mean independence.

### 3.3 The tower rule

$$
\mathbb{E}[X] = \mathbb{E}\big[\mathbb{E}[X \mid Y]\big]
$$

Compute an expectation by first averaging over a conditioning variable. **Worked example:** the expected number of rolls of a die until the first 6, given that every roll counts. Condition on the first roll: with probability $1/6$ the count is 1; with probability $5/6$ we restart. So $E = \frac{1}{6}\cdot 1 + \frac{5}{6}(1 + E)$, giving $E = 6$. The same trick solves infinite sums that would be painful otherwise.

### 3.4 Order statistics: the expected max and min

For $n$ i.i.d. Uniform(0,1) variables:

$$
\mathbb{E}[\max] = \frac{n}{n+1}, \qquad \mathbb{E}[\min] = \frac{1}{n+1}
$$

**Why:** $P(\max \le x) = x^n$, so the density is $n x^{n-1}$, and $\int_0^1 x \cdot n x^{n-1} dx = \frac{n}{n+1}$.

**Worked example:** the expected maximum of two dice.

$$
\mathbb{E}[\max] = \sum_{k=1}^{6} k \cdot \frac{2k - 1}{36}
= \frac{1\cdot1 + 2\cdot3 + 3\cdot5 + 4\cdot7 + 5\cdot9 + 6\cdot11}{36}
= \frac{161}{36} \approx 4.472
$$

### 3.5 The coupon collector

To collect all $n$ coupons: the expected number of draws is $n \cdot H_n = n(1 + \frac12 + \cdots + \frac1n)$.

**Why:** the waiting time for the $k$-th new coupon is geometric with success probability $(n-k+1)/n$, and expectations of geometrics add.

**Worked example:** $n = 5$: $5(1 + \frac12 + \frac13 + \frac14 + \frac15) = 5 \cdot \frac{137}{60} = \frac{137}{12} \approx 11.42$ draws. The last coupon alone costs an expected 5 draws.

## 4. Variance and covariance identities

| Identity | Statement |
|---|---|
| Variance | $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ |
| Scaling | $\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)$ |
| Sum | $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y)$ |
| Independence | Sum reduces to $\operatorname{Var}(X) + \operatorname{Var}(Y)$ |
| Law of total variance | $\operatorname{Var}(Y) = \mathbb{E}[\operatorname{Var}(Y\mid X)] + \operatorname{Var}(\mathbb{E}[Y\mid X])$ |

**Worked example:** a portfolio with two uncorrelated assets, weights $w_1 = w_2 = 0.5$, volatilities $\sigma_1 = 20\%$, $\sigma_2 = 30\%$:

$$
\sigma_p^2 = 0.25(0.04) + 0.25(0.09) = 0.0325 \Rightarrow \sigma_p \approx 18.03\%
$$

The portfolio is less volatile than either asset — diversification in numbers.

## 5. Sums, averages, and the CLT

For i.i.d. $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

$$
\bar X_n = \frac{1}{n}\sum_i X_i, \qquad
\mathbb{E}[\bar X_n] = \mu, \qquad
\operatorname{Var}(\bar X_n) = \frac{\sigma^2}{n}
$$

The central limit theorem sharpens this: regardless of the underlying distribution,

$$
\frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal N(0, 1)
$$

**Worked example:** 100 fair coin flips, $X$ = number of heads. $X \sim \text{Binomial}(100, 0.5)$; exactly, $P(40 \le X \le 60) = \sum_{k=40}^{60}\binom{100}{k}0.5^{100} \approx 0.9648$. Via the CLT with a continuity correction: $P(39.5 \le X \le 60.5)$ where $X \approx \mathcal N(50, 25)$:

$$
P\left(\frac{39.5 - 50}{5} \le Z \le \frac{60.5 - 50}{5}\right) = P(-2.1 \le Z \le 2.1) \approx 0.9643
$$

The approximation is accurate to 3 decimals with $n = 100$.

## 6. Heavy tails: when the normal is the wrong model

The normal is everywhere because of the CLT — but financial returns have **fat tails**: extreme moves happen far more often than a normal predicts.

| Model | Tail behaviour | kurtosis (excess) |
|---|---|---|
| Normal | $e^{-x^2/2}$ — decays super-fast | 0 |
| Student-t | polynomial decay $x^{-\nu-1}$ | $6/(\nu-4)$ |
| Cauchy (t with ν=1) | so heavy the mean does not exist | ∞ |

A normal model with $\sigma = 2\%$ predicts a daily move worse than $-6\sigma$ essentially never; markets see them regularly. The practical responses: model returns with t or mixture distributions, use realised/EWMA volatility instead of historical, and stress-test with jumps.

## 7. The toolkit in one page

| Situation | Model | Key fact |
|---|---|---|
| Count of events | Poisson | mean = variance = λ |
| Time until next event | Exponential | memoryless |
| Sum of many small effects | Normal | CLT |
| Positive, right-skewed price | Log-normal | $E[e^Y] = e^{\mu + \sigma^2/2}$ |
| Heavy tails | Student-t | finite variance only for ν > 2 |
| Belief about a probability | Beta | conjugate to Bernoulli |
| Waiting time for k events | Gamma | conjugate to Poisson |
| Sum of independent | MGFs multiply | Poisson + Poisson = Poisson; normal + normal = normal |

## 8. Practice

1. A die is rolled until a 6 appears. Expected number of rolls? *(Answer: 6 — geometric with $p = 1/6$, mean $1/p$.)*
2. Expected maximum of two independent Uniform(0,1) draws? *(Answer: $2/3$.)*
3. $X \sim \mathcal N(2, 9)$. What is $E[X^2]$? *(Answer: $\operatorname{Var} + \mu^2 = 9 + 4 = 13$.)*
4. Independent $X \sim \text{Poisson}(2)$, $Y \sim \text{Poisson}(3)$. Distribution of $X + Y$? *(Answer: $\text{Poisson}(5)$ — MGFs multiply.)*
5. You flip a fair coin until you see a head. What is the probability it takes more than 5 flips? *(Answer: $2^{-5} = 1/32$ — memoryless: five tails in a row.)*
6. A stock's log-return is $\mathcal N(0.05, 0.04)$. Expected gross return? *(Answer: $e^{0.05 + 0.02} = e^{0.07} \approx 1.0725$ — the $\sigma^2/2$ correction matters.)*
