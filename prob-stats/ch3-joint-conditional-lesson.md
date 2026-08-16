# Chapter 3 — Joint, Conditional & Dependent Variables

Single-variable distributions answer questions such as “How volatile is this return?” Most real decisions ask how variables move together and what changes when new information arrives. Joint distributions contain that complete relationship; marginals, conditionals, covariance, and regression are different views of it.

## 3.1 Joint and marginal distributions

For discrete variables $X$ and $Y$, the joint PMF is

$$
p_{X,Y}(x,y)=P(X=x,Y=y).
$$

The marginal distribution of $X$ sums out $Y$:

$$
p_X(x)=\sum_y p_{X,Y}(x,y).
$$

For continuous variables, replace sums by integrals:

$$
f_X(x)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dy.
$$

“Marginalizing” means ignoring a dimension while preserving probability. In a financial panel, we might marginalize over sectors to study an unconditional return distribution; in a latent-variable model, we integrate out hidden states.

## 3.2 Conditional distributions

For events $A$ and $B$ with $P(B)>0$,

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

Conditioning restricts attention to worlds where $B$ occurred, then renormalizes probability to total one.

For discrete variables,

$$
p_{X\mid Y}(x\mid y)=\frac{p_{X,Y}(x,y)}{p_Y(y)}.
$$

For continuous variables, the same ratio uses densities:

$$
f_{X\mid Y}(x\mid y)=\frac{f_{X,Y}(x,y)}{f_Y(y)}.
$$

Although $P(Y=y)=0$ can hold continuously, conditional density is defined rigorously through regular conditional distributions. The ratio formula is the practical version when densities exist.

## 3.3 Multiplication, total probability, and Bayes

Rearranging the conditional definition gives the multiplication rule:

$$
P(A\cap B)=P(A\mid B)P(B).
$$

If $B_1,\ldots,B_k$ partition the sample space, the law of total probability is

$$
P(A)=\sum_{j=1}^kP(A\mid B_j)P(B_j).
$$

Bayes' theorem reverses the conditioning direction:

$$
P(B_j\mid A)=\frac{P(A\mid B_j)P(B_j)}{\sum_iP(A\mid B_i)P(B_i)}.
$$

The numerator is likelihood times prior; the denominator makes the posterior probabilities sum to one.

### Worked example: updating default risk

Suppose 2% of borrowers default. A warning signal appears for 80% of defaulters and 10% of non-defaulters. Then

$$
P(W)=0.80(0.02)+0.10(0.98)=0.114.
$$

Therefore

$$
P(D\mid W)=\frac{0.80(0.02)}{0.114}\approx0.140.
$$

The warning raises risk from 2% to 14%, but does not make default more likely than not. The low base rate still matters.

## 3.4 Independence and conditional independence

$X$ and $Y$ are independent if their joint distribution factorizes:

$$
f_{X,Y}(x,y)=f_X(x)f_Y(y).
$$

Equivalently, learning $Y$ does not change the distribution of $X$.

Conditional independence is different:

$$
X\perp Y\mid Z.
$$

It says $X$ and $Y$ become independent once $Z$ is known. Stocks can co-move unconditionally because both load on the market, yet their idiosyncratic returns may be approximately independent conditional on the market factor.

Conditioning can also create dependence. If two independent strategies must together achieve a fixed total P&L, observing one constrains the other. Selection and collider bias arise from this mechanism.

## 3.5 Independence versus zero correlation

Independence with finite second moments implies zero covariance. The converse is generally false.

Let $X\sim\operatorname{Uniform}(-1,1)$ and $Y=X^2$. Then $Y$ is completely determined by $X$, so they are dependent. But symmetry gives

$$
\operatorname{Cov}(X,Y)=E[X^3]-E[X]E[X^2]=0.
$$

Correlation sees linear association, not all dependence. This matters when risks interact through absolute returns, squared returns, thresholds, or tails.

For jointly Gaussian variables, zero covariance does imply independence. That is a special property of the multivariate Gaussian, not a general rule.

## 3.6 Conditional expectation

The conditional expectation

$$
E[X\mid Y]
$$

is itself a random variable: before observing $Y$, it is a function of the unknown $Y$; after observing $Y=y$, it becomes a number.

It is the best mean-squared-error predictor of $X$ using information in $Y$:

$$
E[X\mid Y]=\arg\min_{g(Y)}E[(X-g(Y))^2].
$$

This makes conditional expectation central to regression, filtering, forecasting, and derivatives pricing.

## 3.7 The tower property

The law of iterated expectations says

$$
E[E[X\mid Y]]=E[X].
$$

Average the conditional means across possible $Y$ values and the unconditional mean returns.

More generally, if $\mathcal G\subseteq\mathcal H$ are information sets,

$$
E[E[X\mid\mathcal H]\mid\mathcal G]=E[X\mid\mathcal G].
$$

Once a forecast has been compressed to the best forecast under richer information, projecting it onto poorer information is equivalent to forecasting directly with the poorer information.

### Worked expectation without a full distribution

Let $N$ be the number of trades and $X_i$ their P&Ls. If conditional on $N$, the $X_i$ have common mean $\mu$, then

$$
E\left[\sum_{i=1}^{N}X_i\mid N\right]=N\mu.
$$

Taking expectations again gives

$$
E\left[\sum_{i=1}^{N}X_i\right]=E[N]\mu.
$$

This is Wald's identity under suitable independence and integrability assumptions.

## 3.8 Total variance and covariance decomposition

The law of total variance separates within-group uncertainty from between-group uncertainty:

$$
\operatorname{Var}(X)=E[\operatorname{Var}(X\mid Y)]+\operatorname{Var}(E[X\mid Y]).
$$

- $E[\operatorname{Var}(X\mid Y)]$: noise remaining within each state.
- $\operatorname{Var}(E[X\mid Y])$: variation explained by state-dependent means.

A related identity is

$$
\operatorname{Cov}(X,Z)
=E[\operatorname{Cov}(X,Z\mid Y)]
+\operatorname{Cov}(E[X\mid Y],E[Z\mid Y]).
$$

Apparent correlation may therefore come from within-state co-movement, common variation across states, or both.

## 3.9 Transformations and joint simulation

If $Z_1,Z_2$ are independent standard Gaussians, correlated Gaussian variables can be built as

$$
X=Z_1,\qquad Y=\rho Z_1+\sqrt{1-\rho^2}Z_2.
$$

Then $\operatorname{Corr}(X,Y)=\rho$. In matrix form, Cholesky factors transform independent shocks into correlated ones. This is a core Monte Carlo construction.

Correlation alone does not determine joint tail behaviour outside the Gaussian family. Copulas separate marginal distributions from dependence structure, though their assumptions must also be validated.

## 3.10 Association is not causation

$P(Y\mid X)$ describes an observational relationship. It need not equal the result of intervening to set $X$. Confounding, selection, reverse causality, and measurement timing can all create predictive association without causal effect.

This does not make prediction useless. It means “the feature predicts returns” and “changing the feature would change returns” are different claims.

## 3.11 Computational view

Conditional averages can be estimated by grouping observations:

```python
def grouped_means(rows, group_key, value_key):
    groups = {}
    for row in rows:
        total, count = groups.get(row[group_key], (0.0, 0))
        groups[row[group_key]] = (total + row[value_key], count + 1)
    return {group: total / count for group, (total, count) in groups.items()}
```

In high dimensions, exact groups become sparse. Regression and machine learning estimate smooth or regularized versions of $E[Y\mid X]$.

## 3.12 Failure modes

- **Base-rate neglect:** focusing on sensitivity while ignoring prior prevalence.
- **Conditioning in the wrong direction:** $P(A\mid B)$ is not $P(B\mid A)$.
- **Equating zero correlation with independence:** nonlinear dependence remains invisible.
- **Conditioning on a collider:** selection can create spurious relationships.
- **Using a common correlation across regimes:** dependence can change sharply in stress.
- **Making causal claims from predictive conditionals:** intervention requires stronger design and assumptions.

## 3.13 Knowledge checks

1. A test has 95% sensitivity and 90% specificity for a condition affecting 1% of people. Find the probability of the condition after a positive test.
2. Prove that independence implies zero covariance when second moments exist.
3. Give an example where $X$ and $Y$ are marginally dependent but conditionally independent given $Z$.
4. Explain both terms in the law of total variance using market regimes.
5. If $E[Y\mid X]=2X$ and $E[X]=3$, find $E[Y]$.

### Solution outlines

1. $0.95(0.01)/[0.95(0.01)+0.10(0.99)]\approx0.0876$.
2. Independence gives $E[XY]=E[X]E[Y]$, so covariance is zero.
3. Two stocks sharing a market factor can be dependent marginally and independent conditional on that factor under an ideal factor model.
4. Average within-regime variance is state-specific noise; variance of regime means is uncertainty explained by changing regimes.
5. By the tower property, $E[Y]=E[E[Y\mid X]]=2E[X]=6$.

## 3.14 What to retain

- A joint distribution contains both marginal behaviour and dependence.
- Conditioning restricts the possible world and renormalizes probability.
- Bayes combines likelihood with base rates.
- Conditional expectation is the optimal squared-error forecast.
- Independence, zero correlation, conditional independence, and causality are different ideas.

Next: [Chapter 4 — Samples, Estimators, LLN & CLT](ch4-sampling-clt-viewer.html) explains how uncertain samples produce estimators and sampling distributions.
