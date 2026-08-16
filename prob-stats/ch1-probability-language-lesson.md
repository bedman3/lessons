# Chapter 1 — Probability Language: From Outcomes to Random Variables

Probability begins with a simple problem: the future is uncertain, but we still need a precise language for describing what could happen. That language has three layers:

1. a **sample space** lists possible outcomes;
2. **events** describe questions about those outcomes;
3. a **random variable** turns outcomes into numbers we can model.

The distinction between discrete and continuous random variables then determines whether probability is represented by masses or densities. Most later confusion comes from mixing those two worlds.

## 1.1 Experiments, outcomes, and events

A random experiment is any repeatable setup whose outcome is not known in advance. The **sample space** $\Omega$ is the set of all possible outcomes.

- Coin toss: $\Omega=\{H,T\}$.
- Two dice: $\Omega=\{(i,j):i,j\in\{1,\ldots,6\}\}$.
- Tomorrow's return: $\Omega$ may be modelled as all real numbers.

An **event** is a subset of $\Omega$. For two dice, “the total exceeds 9” is

$$
A=\{(4,6),(5,5),(5,6),(6,4),(6,5),(6,6)\}.
$$

Events can be combined with set operations:

- $A^c$: $A$ does not occur;
- $A\cup B$: at least one of $A$ and $B$ occurs;
- $A\cap B$: both occur;
- $A\setminus B$: $A$ occurs but $B$ does not.

This is not decorative notation. “A borrower defaults or becomes delinquent,” “a strategy loses money and volatility rises,” and “a classifier is wrong on subgroup $B$” are all event operations.

## 1.2 The probability axioms

A probability measure $P$ assigns numbers to events and obeys three rules:

1. $P(A)\ge 0$ for every event $A$;
2. $P(\Omega)=1$;
3. for disjoint events $A_1,A_2,\ldots$,

$$
P\left(\bigcup_i A_i\right)=\sum_iP(A_i).
$$

Useful consequences follow rather than being separate laws:

$$
P(A^c)=1-P(A),
$$

and

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B).
$$

The subtraction prevents the overlap from being counted twice.

## 1.3 Random variables are measuring rules

A **random variable** $X$ is a function from outcomes to numbers:

$$
X:\Omega\rightarrow\mathbb{R}.
$$

The randomness is in the outcome $\omega$; the mapping $X(\omega)$ is deterministic. With two dice, $X(i,j)=i+j$ maps 36 detailed outcomes to totals from 2 to 12. In finance, a path of market events may map to a daily P&L. In ML, an observed example may map to its loss.

This distinction matters because models usually describe $X$, not every microscopic element of $\Omega$.

## 1.4 Discrete variables: probability mass

A discrete random variable takes a finite or countably infinite collection of values. Its **probability mass function** (PMF) is

$$
p_X(x)=P(X=x).
$$

The PMF is nonnegative and sums to one:

$$
\sum_xp_X(x)=1.
$$

### Worked example: a Bernoulli variable

Let $X=1$ if a trade fills and $X=0$ otherwise. If the fill probability is $p$,

$$
P(X=x)=p^x(1-p)^{1-x},\qquad x\in\{0,1\}.
$$

For $p=0.7$, $P(X=1)=0.7$ and $P(X=0)=0.3$. Each point can carry positive probability.

## 1.5 Continuous variables: probability density

A continuous random variable is described by a **probability density function** (PDF) $f_X$. Probabilities are areas:

$$
P(a<X\le b)=\int_a^b f_X(x)\,dx.
$$

The density satisfies

$$
f_X(x)\ge0,\qquad \int_{-\infty}^{\infty}f_X(x)\,dx=1.
$$

The essential distinction is

$$
f_X(x)\ne P(X=x).
$$

For a continuous $X$,

$$
P(X=x)=\int_x^x f_X(u)\,du=0.
$$

This does not mean $x$ is impossible. It means a single point has zero width. Every individual point can have probability zero while an interval containing infinitely many points has positive probability.

### Worked example: uniform density

Suppose $X\sim\operatorname{Uniform}(0,10)$. Its density is $f_X(x)=0.1$ on $[0,10]$. Then

$$
P(2<X<5)=\int_2^5 0.1\,dx=0.3.
$$

The value $0.1$ is density per unit of $x$, not a 10% probability at each point. Changing the unit changes the numerical density but not interval probabilities.

## 1.6 One object works for both: the CDF

The **cumulative distribution function** is

$$
F_X(x)=P(X\le x).
$$

Every random variable has a CDF. It is nondecreasing, right-continuous, approaches 0 as $x\to-\infty$, and approaches 1 as $x\to\infty$.

- For discrete variables, jumps in $F_X$ are probability masses.
- For continuous variables, $F_X(x)=\int_{-\infty}^x f_X(u)\,du$, and where differentiable, $f_X(x)=F_X'(x)$.

For any distribution,

$$
P(a<X\le b)=F_X(b)-F_X(a).
$$

The exact use of $<$ or $\le$ matters for discrete variables because points can have mass. It does not change probabilities for a fully continuous variable.

## 1.7 Mixed distributions

Real models need not be purely discrete or continuous. Suppose a limit order has a 30% probability of not filling, giving execution quantity $Q=0$. Conditional on filling, $Q$ is continuously distributed between 0 and 100.

Then the CDF has a jump of $0.3$ at zero plus a smooth part above zero. There is no single ordinary PDF that represents the point mass correctly. Thinking through the CDF avoids forcing the model into the wrong category.

Mixture models, zero-inflated counts, default recovery, and censored observations often have this structure.

## 1.8 Transforming a random variable

If $Y=g(X)$, its distribution is induced by the distribution of $X$ and the mapping $g$.

For a one-to-one differentiable transformation $y=g(x)$,

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
$$

The derivative corrects for stretching or compressing the horizontal axis.

### Worked transformation

Let $X\sim\operatorname{Uniform}(0,1)$ and $Y=-\log X$. For $y\ge0$,

$$
P(Y\le y)=P(-\log X\le y)=P(X\ge e^{-y})=1-e^{-y}.
$$

Therefore $f_Y(y)=e^{-y}$: $Y$ is exponential with rate 1. This inverse-CDF idea is also a basic simulation method.

## 1.9 Quantiles and inverse probability questions

The $q$-quantile is

$$
F_X^{-1}(q)=\inf\{x:F_X(x)\ge q\}.
$$

It answers the inverse question: “What threshold is exceeded only $1-q$ of the time?” A 99% loss quantile is the basis of Value at Risk. Classification thresholds and prediction intervals use the same logic.

For discrete distributions, an exact equality $F_X(x)=q$ may not exist, which is why the infimum definition matters.

## 1.10 Computational view

A CDF can be approximated from observations $x_1,\ldots,x_n$ by the empirical CDF

$$
\widehat F_n(x)=\frac{1}{n}\sum_{i=1}^n\mathbf 1\{x_i\le x\}.
$$

```python
def empirical_cdf(sample, threshold):
    return sum(x <= threshold for x in sample) / len(sample)
```

Unlike a histogram, the empirical CDF does not require bin-width choices. It is often the clearest first diagnostic for comparing distributions.

## 1.11 Failure modes and common traps

- **Treating density as mass:** $f(x)$ can exceed 1; only integrated probability must lie in $[0,1]$.
- **Ignoring support:** a Gaussian approximation can assign probability to impossible negative values.
- **Confusing an outcome with an event:** $X=3$ is shorthand for the event $\{\omega:X(\omega)=3\}$.
- **Assuming everything has a PDF:** discrete and mixed distributions may not.
- **Losing the Jacobian:** transformed densities must account for changed scale.
- **Reading probability as causality:** a distribution describes uncertainty, not why outcomes occur.

## 1.12 Knowledge checks

1. A PDF equals 2 on $[0,0.5]$ and 0 elsewhere. Is this valid? What is $P(X<0.1)$?
2. If $X$ is continuous, how can $P(0<X<1)>0$ when every point has probability zero?
3. Let $X$ take values $-1,0,2$ with probabilities $0.2,0.5,0.3$. Write its CDF.
4. A model assigns probability 0.15 to default and otherwise uses a continuous recovery rate. Is the recovery variable discrete, continuous, or mixed?
5. If $X\sim\operatorname{Uniform}(0,1)$ and $Y=X^2$, find $F_Y(y)$ for $0\le y\le1$.

### Solution outlines

1. Yes: its area is $2(0.5)=1$. The probability is $\int_0^{0.1}2\,dx=0.2$.
2. Probability is countably additive, not obtained by assigning positive mass to each of uncountably many points. Intervals receive probability through area.
3. $F(x)=0$ below $-1$, $0.2$ on $[-1,0)$, $0.7$ on $[0,2)$, and $1$ at and above 2.
4. Mixed: it has a point mass for default plus a continuous component otherwise.
5. $F_Y(y)=P(X^2\le y)=P(X\le\sqrt y)=\sqrt y$.

## 1.13 What to retain

- Events are subsets of possible outcomes; random variables measure outcomes numerically.
- Discrete probabilities are masses. Continuous probabilities are areas under densities.
- The CDF unifies discrete, continuous, and mixed distributions.
- A density at a point is not the probability of that point.
- Transformations change distributions according to how they stretch probability mass.

Next: [Chapter 2 — Distributions & Moments](ch2-distributions-moments-viewer.html) develops the recurring distribution families and the summaries used to compare them.
