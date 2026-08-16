# Chapter 2 — Distributions & Moments: Shape, Centre, Spread, and Tails

A distribution is more than a formula. It is a story about how outcomes arise. The binomial counts successes in a fixed number of trials; the Poisson counts arrivals in an interval; the exponential measures waiting time. Choosing a distribution means choosing a data-generating story and accepting its assumptions.

## 2.1 Expectation: the long-run balancing point

For a discrete variable,

$$
E[X]=\sum_x x\,p_X(x),
$$

and for a continuous variable,

$$
E[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx.
$$

Expectation is a probability-weighted average, not necessarily a likely outcome. The expected value of a fair six-sided die is $3.5$, which can never appear on one roll.

More generally,

$$
E[g(X)]=\sum_xg(x)p_X(x)
$$

or its integral equivalent. We do not need to derive the distribution of $g(X)$ first.

Linearity is the workhorse:

$$
E[aX+bY+c]=aE[X]+bE[Y]+c.
$$

It does not require independence.

## 2.2 Variance and covariance

Variance measures squared distance from the mean:

$$
\operatorname{Var}(X)=E[(X-E[X])^2]=E[X^2]-E[X]^2.
$$

Standard deviation $\sigma_X$ restores the units of $X$:

$$
\sigma_X=\sqrt{\operatorname{Var}(X)}.
$$

Covariance measures linear co-movement:

$$
\operatorname{Cov}(X,Y)=E[(X-E[X])(Y-E[Y])].
$$

Correlation removes scale:

$$
\rho_{XY}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

For a linear combination,

$$
\operatorname{Var}(aX+bY)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y)+2ab\operatorname{Cov}(X,Y).
$$

This is the algebra behind portfolio risk, regression uncertainty, and correlated simulation.

## 2.3 Moments, skewness, and tails

The $k$th raw moment is $E[X^k]$; the $k$th central moment is $E[(X-E[X])^k]$.

- First moment: location.
- Second central moment: variance.
- Standardized third central moment: skewness.
- Standardized fourth central moment: kurtosis.

Moments are useful when they exist. A heavy-tailed distribution may have a finite mean but infinite variance, or neither. Sample estimates can then behave very differently from the familiar Gaussian case.

The moment-generating function (MGF), when finite near zero, is

$$
M_X(t)=E[e^{tX}].
$$

Derivatives at zero recover moments, and independence turns sums into products: $M_{X+Y}(t)=M_X(t)M_Y(t)$. Characteristic functions $E[e^{itX}]$ always exist and play the same structural role more generally.

## 2.4 Bernoulli and binomial distributions

A Bernoulli variable records one success:

$$
X\sim\operatorname{Bernoulli}(p),\qquad P(X=1)=p.
$$

Because $X^2=X$,

$$
E[X]=p,\qquad \operatorname{Var}(X)=p-p^2=p(1-p).
$$

If $X_1,\ldots,X_n$ are independent Bernoulli$(p)$ variables, their sum

$$
S_n=\sum_{i=1}^nX_i\sim\operatorname{Binomial}(n,p).
$$

Linearity gives

$$
E[S_n]=np,
$$

and independence removes covariance terms:

$$
\operatorname{Var}(S_n)=np(1-p).
$$

The binomial requires a fixed number of trials, a common success probability, and independence. Credit defaults, conversions, or fills often violate at least one of those assumptions through common shocks or changing conditions.

## 2.5 Geometric and Poisson distributions

The geometric distribution models the trial count $T$ until the first success:

$$
P(T=k)=(1-p)^{k-1}p,\qquad k=1,2,\ldots
$$

It is memoryless:

$$
P(T>s+t\mid T>s)=P(T>t).
$$

The Poisson distribution models an event count $N$ in an interval when events arrive independently at a constant average rate:

$$
P(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\qquad E[N]=\operatorname{Var}(N)=\lambda.
$$

Variance much larger than the mean signals overdispersion relative to Poisson, often caused by varying intensity or clustering.

## 2.6 Uniform and exponential distributions

For $X\sim\operatorname{Uniform}(a,b)$,

$$
f_X(x)=\frac{1}{b-a},\qquad a\le x\le b,
$$

with

$$
E[X]=\frac{a+b}{2},\qquad \operatorname{Var}(X)=\frac{(b-a)^2}{12}.
$$

For $T\sim\operatorname{Exponential}(\lambda)$,

$$
f_T(t)=\lambda e^{-\lambda t},\qquad t\ge0.
$$

Its survival function is $P(T>t)=e^{-\lambda t}$. Therefore

$$
P(T>s+t\mid T>s)
=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}
=e^{-\lambda t},
$$

showing continuous memorylessness. A constant hazard rate is strong; real arrival and failure processes often have time-varying hazards.

## 2.7 Gaussian and Student's t distributions

The Gaussian distribution is defined by mean $\mu$ and variance $\sigma^2$:

$$
f_X(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
$$

It appears because sums of many small contributions often become approximately Gaussian, not because all data are inherently normal.

Standardization produces

$$
Z=\frac{X-\mu}{\sigma}\sim N(0,1).
$$

Student's t distribution is symmetric like the Gaussian but has heavier tails. If $Z\sim N(0,1)$, $U\sim\chi^2_\nu$, and they are independent, then

$$
T=\frac{Z}{\sqrt{U/\nu}}\sim t_\nu.
$$

The random denominator captures uncertainty from estimating variance. As degrees of freedom $\nu$ grow, $t_\nu$ approaches the standard normal.

## 2.8 A practical distribution map

| Data-generating story | Candidate distribution | Question to challenge |
|---|---|---|
| One binary outcome | Bernoulli | Is success probability stable? |
| Successes among fixed trials | Binomial | Are trials independent? |
| Trials until first success | Geometric | Is memorylessness plausible? |
| Event count in an interval | Poisson | Is the rate constant; is variance near mean? |
| Waiting time at constant rate | Exponential | Is the hazard constant? |
| Bounded equally likely value | Uniform | Why should density be flat? |
| Sum of many mild contributions | Gaussian | Are tails and dependence mild enough? |
| Gaussian centre with extra tail uncertainty | Student's t | What mechanism produces heavy tails? |

## 2.9 Worked example: portfolio mean and variance

Let returns $R_1,R_2$ have volatilities $20\%$ and $10\%$, correlation $0.3$, and equal portfolio weights. The covariance is

$$
\operatorname{Cov}(R_1,R_2)=0.3(0.2)(0.1)=0.006.
$$

Portfolio variance is

$$
\begin{aligned}
\operatorname{Var}(R_p)
&=(0.5)^2(0.2)^2+(0.5)^2(0.1)^2\\
&\quad+2(0.5)(0.5)(0.006)\\
&=0.0155.
\end{aligned}
$$

So volatility is $\sqrt{0.0155}\approx12.45\%$. Averaging volatilities would give 15%, which ignores covariance and is wrong.

## 2.10 Computational view

Simulation is useful for checking intuition, not proving assumptions:

```python
import random

def bernoulli_sample(p, n):
    return [int(random.random() < p) for _ in range(n)]

sample = bernoulli_sample(0.3, 10_000)
mean = sum(sample) / len(sample)
```

The empirical mean should approach $p$, while its run-to-run error shrinks at roughly $1/\sqrt n$. Chapter 4 explains why.

## 2.11 Failure modes

- **Selecting by shape alone:** similar histograms can arise from different processes.
- **Equating zero correlation with independence:** nonlinear dependence can have zero covariance.
- **Using variance when it does not exist:** heavy tails can invalidate familiar error estimates.
- **Assuming Gaussian tails:** small central differences can hide enormous tail differences.
- **Confusing parameter and observation:** $p$ is a model parameter; a Bernoulli outcome is 0 or 1.
- **Ignoring dependence in sums:** covariance terms determine aggregation risk.

## 2.12 Knowledge checks

1. Derive the variance of a Bernoulli variable using $X^2=X$.
2. A count series has mean 4 and variance 19. Why might Poisson be inadequate?
3. Two assets have zero correlation. Must a portfolio of them be free of dependence risk?
4. If $X_1,\ldots,X_n$ are independent with variance $\sigma^2$, find the variance of their average.
5. Why does a t distribution produce wider intervals than a Gaussian at low degrees of freedom?

### Solution outlines

1. $E[X^2]-E[X]^2=p-p^2=p(1-p)$.
2. Poisson requires equal mean and variance; the data are overdispersed, perhaps from varying intensity or clustering.
3. No. Zero correlation removes linear covariance only; nonlinear or tail dependence can remain.
4. $\operatorname{Var}(\bar X)=n\sigma^2/n^2=\sigma^2/n$.
5. Estimating scale adds uncertainty; the t distribution allocates more probability to tail outcomes.

## 2.13 What to retain

- A distribution encodes a data-generating story and assumptions.
- Expectation is linear; variance aggregation depends on covariance.
- Gaussian behaviour is often an approximation to aggregated noise.
- Student's t reflects uncertainty in estimated scale and has heavier tails.
- Tail behaviour and dependence matter at least as much as centre and spread.

Next: [Chapter 3 — Joint, Conditional & Dependent Variables](ch3-joint-conditional-viewer.html) explains how variables interact and how information changes probability.
