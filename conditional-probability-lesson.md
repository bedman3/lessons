# Conditional Probability

> From the chain rule to Bayes' theorem — the mathematical language of learning from data, with the traps and interview questions that matter.

## 1. Why conditional probability matters

Conditional probability is the language of *updating belief in light of evidence*. Every time you ask "given what I have seen, what should I expect?" you are reasoning conditionally.

In machine learning, conditional probability is the foundation: classification estimates $\Pr(Y \mid X)$, regression estimates $\mathbb{E}[Y \mid X]$, and Bayesian inference updates $\Pr(\theta \mid \mathcal{D})$. In finance, it underpins default modelling, scenario analysis, and risk assessment. In everyday reasoning, it is the difference between a correct inference and the prosecutor's fallacy.

The central idea is simple but profound:

> Conditioning is zooming in on the subset of the world consistent with what you now know, and re-normalising so that probabilities still sum to one.

---

## 2. Definitions and the chain rule

### 2.1 Conditional probability

The conditional probability of event $A$ given event $B$ is:

$$
\Pr(A \mid B) = \frac{\Pr(A \cap B)}{\Pr(B)}, \quad \Pr(B) > 0.
$$

You restrict the sample space to $B$ and measure how much of $A$ falls inside it. For continuous random variables, replace probabilities with densities:

$$
f_{X\mid Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}, \quad f_Y(y) > 0.
$$

### 2.2 The chain rule (product rule)

Rearrange the definition to factorise a joint distribution:

$$
\Pr(A, B) = \Pr(A \mid B)\,\Pr(B) = \Pr(B \mid A)\,\Pr(A).
$$

For more than two events, the chain rule extends recursively:

$$
\Pr(A_1, A_2, \ldots, A_n) = \Pr(A_1) \cdot \Pr(A_2 \mid A_1) \cdot \Pr(A_3 \mid A_1, A_2) \cdots \Pr(A_n \mid A_1, \ldots, A_{n-1}).
$$

This factorisation is always valid — no independence assumption required. It is the basis for autoregressive models, sequence modelling, and any situation where you decompose a joint distribution into a product of conditionals.

### 2.3 The law of total probability

If events $B_1, B_2, \ldots, B_k$ form a partition of the sample space (mutually exclusive and exhaustive), then:

$$
\Pr(A) = \sum_{i=1}^{k} \Pr(A \mid B_i)\,\Pr(B_i).
$$

This is also called the **marginalisation** rule. In the continuous case:

$$
f_X(x) = \int f_{X\mid Y}(x \mid y)\,f_Y(y)\,dy.
$$

The law of total probability is how you compute the denominator in Bayes' theorem. It is also how you compute the marginal likelihood of a model by averaging over parameters, and how you compute predictive distributions by averaging over posterior uncertainty.

---

## 3. Bayes' theorem

### 3.1 The theorem

Bayes' theorem inverts a conditional. From $\Pr(A \mid B)\,\Pr(B) = \Pr(B \mid A)\,\Pr(A)$:

$$
\Pr(A \mid B) = \frac{\Pr(B \mid A)\,\Pr(A)}{\Pr(B)}
               = \frac{\Pr(B \mid A)\,\Pr(A)}{\sum_i \Pr(B \mid A_i)\,\Pr(A_i)}.
$$

The numerator is the joint probability (likelihood × prior). The denominator is the normalising constant (total probability of the evidence).

### 3.2 The Bayesian inference framework

With parameters $\theta$ and data $\mathcal{D}$:

$$
\underbrace{\Pr(\theta \mid \mathcal{D})}_{\text{posterior}}
= \frac{\overbrace{\Pr(\mathcal{D} \mid \theta)}^{\text{likelihood}}\;
        \overbrace{\Pr(\theta)}^{\text{prior}}}
       {\underbrace{\Pr(\mathcal{D})}_{\text{evidence}}}.
$$

Each term has a clear role:

| Term | Meaning | Role |
|---|---|---|
| Prior $\Pr(\theta)$ | What you believe about $\theta$ before seeing the data. | Encodes domain knowledge; can be informative or weak. |
| Likelihood $\Pr(\mathcal{D} \mid \theta)$ | How probable the observed data is, under parameter value $\theta$. | The data speaks through the likelihood. |
| Evidence $\Pr(\mathcal{D})$ | The average likelihood over the prior. | Normalises the posterior; critical for model comparison. |
| Posterior $\Pr(\theta \mid \mathcal{D})$ | What you should believe about $\theta$ after seeing the data. | The output of Bayesian inference. |

### 3.3 MLE, MAP, and full Bayesian inference

- **Maximum Likelihood Estimation (MLE):** $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \Pr(\mathcal{D} \mid \theta)$. Drops the prior entirely. Equivalent to minimising the negative log-likelihood.
- **Maximum a Posteriori (MAP):** $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \Pr(\mathcal{D} \mid \theta)\,\Pr(\theta)$. Adds a prior; equivalent to MLE with regularisation (e.g. L2 regularisation = Gaussian prior).
- **Full Bayesian:** works with the entire posterior distribution. Average over parameters when predicting, rather than plugging in a point estimate. This properly accounts for parameter uncertainty.

### 3.4 Worked example: medical testing

A disease affects 1% of the population. A test has 95% sensitivity (true positive rate) and 95% specificity (true negative rate). If someone tests positive, what is the probability they actually have the disease?

**Solution.** Let $D$ = has disease, $+$ = positive test.

Given: $\Pr(D) = 0.01$, $\Pr(+ \mid D) = 0.95$, $\Pr(- \mid \neg D) = 0.95$, so $\Pr(+ \mid \neg D) = 0.05$.

Apply Bayes:

$$
\begin{aligned}
\Pr(+) &= \Pr(+ \mid D)\,\Pr(D) + \Pr(+ \mid \neg D)\,\Pr(\neg D) \\
       &= 0.95 \times 0.01 + 0.05 \times 0.99 = 0.0095 + 0.0495 = 0.059. \\[6pt]
\Pr(D \mid +) &= \frac{\Pr(+ \mid D)\,\Pr(D)}{\Pr(+)}
               = \frac{0.0095}{0.059} \approx 16.1\%.
\end{aligned}
$$

Despite a "95% accurate" test, only ~16% of positives are true positives. The intuition: the disease is so rare that the 5% false positive rate generates many more false positives than the 95% true positive rate generates true positives. This is the **base-rate fallacy**.

### 3.5 Worked example: Monty Hall

Three doors. Behind one is a car; behind the other two, goats. You pick door 1. Monty, who knows what is behind the doors, opens door 3 to reveal a goat. Should you switch to door 2?

**Solution.** Let $C_i$ be "car behind door $i$", and $O_3$ be "Monty opens door 3".

Priors: $\Pr(C_1) = \Pr(C_2) = \Pr(C_3) = 1/3$.

Likelihoods (Monty must open a goat door you did not pick):
- If car behind 1: Monty can open 2 or 3 → $\Pr(O_3 \mid C_1) = 1/2$.
- If car behind 2: Monty must open 3 (cannot open 2, which has the car) → $\Pr(O_3 \mid C_2) = 1$.
- If car behind 3: Monty cannot open the car door → $\Pr(O_3 \mid C_3) = 0$.

By Bayes:

$$
\begin{aligned}
\Pr(C_1 \mid O_3) &= \frac{\Pr(O_3 \mid C_1)\,\Pr(C_1)}{\sum_i \Pr(O_3 \mid C_i)\,\Pr(C_i)}
                    = \frac{(1/2)(1/3)}{(1/2)(1/3) + (1)(1/3) + (0)(1/3)}
                    = \frac{1/6}{1/2} = \frac{1}{3}. \\[6pt]
\Pr(C_2 \mid O_3) &= \frac{(1)(1/3)}{1/2} = \frac{2}{3}.
\end{aligned}
$$

Switching doubles your chance of winning from $1/3$ to $2/3$. The key insight: Monty's action carries information — he *cannot* open the door with the car.

---

## 4. Independence and conditional independence

### 4.1 Independence

Events $A$ and $B$ are independent if:

$$
\Pr(A, B) = \Pr(A)\,\Pr(B) \quad\Longleftrightarrow\quad \Pr(A \mid B) = \Pr(A).
$$

Knowing $B$ tells you nothing about $A$. For random variables, independence means the joint factorises: $f_{X,Y}(x,y) = f_X(x)\,f_Y(y)$.

### 4.2 Conditional independence

$A$ and $B$ are conditionally independent given $C$ if:

$$
\Pr(A, B \mid C) = \Pr(A \mid C)\,\Pr(B \mid C).
$$

This is a fundamentally different statement from marginal independence. Two variables can be:
- Marginally dependent but conditionally independent (the most common ML case — features are correlated, but given the class label they are independent; this is the naïve Bayes assumption).
- Marginally independent but conditionally dependent (classic example: two independent coin flips, and $C$ = "the two flips are the same" — conditioning on $C$ makes them perfectly dependent).

Conditional independence is the engine behind graphical models, naïve Bayes classifiers, and the factorisation assumptions that make high-dimensional learning tractable.

### 4.3 The naïve Bayes classifier

Assume features $X_1, \ldots, X_d$ are conditionally independent given the class $Y$:

$$
\Pr(Y = k \mid X_1, \ldots, X_d) \propto \Pr(Y = k) \prod_{j=1}^{d} \Pr(X_j \mid Y = k).
$$

This is "naïve" because features are rarely truly conditionally independent. Yet it works surprisingly well in practice — the probability *ranking* often survives even when the probability *values* are miscalibrated. When it fails, it is usually because strong feature correlations break the ranking.

### 4.4 Conjugate priors: the Beta-Binomial story

A **conjugate prior** is one that, when multiplied by the likelihood, yields a posterior in the same family. This makes Bayesian updating tractable — you just update the parameters.

**Beta-Binomial.** You have a coin with unknown probability $p$. Your prior: $p \sim \text{Beta}(\alpha, \beta)$. You flip $n$ times and observe $k$ heads. Likelihood: $\text{Binomial}(n, p)$. Posterior:

$$
p \mid \text{data} \sim \text{Beta}(\alpha + k,\; \beta + n - k).
$$

The Beta parameters act as "pseudo-counts": $\alpha$ is like prior heads, $\beta$ is like prior tails. The posterior mean is:

$$
\mathbb{E}[p \mid \text{data}] = \frac{\alpha + k}{\alpha + \beta + n}
= \underbrace{\frac{\alpha + \beta}{\alpha + \beta + n}}_{\text{shrinkage weight}} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}}
+ \underbrace{\frac{n}{\alpha + \beta + n}}_{\text{data weight}} \cdot \underbrace{\frac{k}{n}}_{\text{sample mean}}.
$$

This is **shrinkage**: the posterior mean is a weighted average of the prior mean and the sample mean. With more data, the prior's influence decays.

**Dirichlet-Multinomial.** The multivariate extension. Prior: $\mathbf{p} \sim \text{Dirichlet}(\alpha_1, \ldots, \alpha_K)$. Data: counts $n_1, \ldots, n_K$ in $n$ trials. Posterior: $\mathbf{p} \mid \text{data} \sim \text{Dirichlet}(\alpha_1 + n_1, \ldots, \alpha_K + n_K)$. This is the foundation of Bayesian naïve Bayes and topic models (LDA).

**Normal-Normal (known variance).** Prior: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$. Data: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$. Posterior:

$$
\mu \mid \text{data} \sim \mathcal{N}\!\left(
\frac{\frac{\mu_0}{\tau^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}},\;
\frac{1}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}
\right).
$$

The posterior precision (1/variance) is the sum of the prior precision and the data precision. The posterior mean is a precision-weighted average.

### 4.5 Exchangeability and de Finetti's theorem

A sequence of random variables is **exchangeable** if the joint distribution is invariant to permutation:

$$
\Pr(X_1, \ldots, X_n) = \Pr(X_{\pi(1)}, \ldots, X_{\pi(n)}) \quad \text{for any permutation } \pi.
$$

Exchangeability is weaker than i.i.d. — i.i.d. implies exchangeable, but not vice versa. Crucially, **de Finetti's theorem** says that an infinite exchangeable sequence of binary random variables can be represented as:

$$
\Pr(X_1 = x_1, \ldots, X_n = x_n) = \int_0^1 \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} \, dF(p),
$$

where $F$ is a distribution over $p$. In words: an exchangeable sequence behaves *as if* there is an unknown parameter $p$ drawn from some prior, and observations are i.i.d. given $p$. This is a deep justification for Bayesian modelling — the prior exists not as a subjective belief but as a mathematical consequence of exchangeability.

---

## 5. Information theory and probability

### 5.1 Entropy

Entropy measures the uncertainty in a distribution. For a discrete random variable $X$:

$$
H(X) = -\sum_x \Pr(X = x) \log_2 \Pr(X = x).
$$

It is measured in bits (if log base 2) or nats (if natural log). Entropy is maximised by the uniform distribution and minimised (to zero) by a point mass. It answers: *on average, how many yes/no questions do I need to ask to determine the outcome?*

For a Bernoulli variable with probability $p$: $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$. This is the binary entropy function — it peaks at $p=0.5$ ($H=1$ bit) and goes to zero at $p=0$ and $p=1$.

### 5.2 Joint and conditional entropy

$$
\begin{aligned}
H(X, Y) &= -\sum_{x,y} \Pr(x,y) \log \Pr(x,y) \quad &\text{(joint entropy)} \\
H(Y \mid X) &= -\sum_{x,y} \Pr(x,y) \log \Pr(y \mid x) \quad &\text{(conditional entropy)} \\
H(X,Y) &= H(X) + H(Y \mid X) = H(Y) + H(X \mid Y) \quad &\text{(chain rule for entropy)}
\end{aligned}
$$

If $X$ and $Y$ are independent, $H(Y \mid X) = H(Y)$. If $Y$ is a deterministic function of $X$, $H(Y \mid X) = 0$.

### 5.3 KL divergence

The Kullback-Leibler divergence measures how one distribution $Q$ diverges from a reference distribution $P$:

$$
D_{\mathrm{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}.
$$

Key properties:
- $D_{\mathrm{KL}}(P \parallel Q) \ge 0$, with equality iff $P = Q$ almost everywhere.
- It is **not symmetric**: $D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$.
- $D_{\mathrm{KL}}(P \parallel Q)$ is the expected *extra* information needed to encode samples from $P$ using a code optimised for $Q$.

In ML, minimising $D_{\mathrm{KL}}(P_{\text{data}} \parallel P_{\text{model}})$ with respect to the model parameters is equivalent to maximising the expected log-likelihood. Minimising $D_{\mathrm{KL}}(P_{\text{model}} \parallel P_{\text{data}})$ is different (it leads to mode-seeking rather than mode-covering behaviour) and is generally not what MLE does.

### 5.4 Cross-entropy

Cross-entropy is closely related:

$$
H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{\mathrm{KL}}(P \parallel Q).
$$

Since $H(P)$ is fixed (it depends only on the true data distribution), minimising cross-entropy with respect to $Q$ is equivalent to minimising $D_{\mathrm{KL}}(P \parallel Q)$, which is equivalent to maximum likelihood estimation. This is why cross-entropy is the standard classification loss: it directly estimates $\Pr(Y \mid X)$.

### 5.5 Mutual information

Mutual information measures dependence — how much knowing $X$ reduces uncertainty about $Y$:

$$
I(X; Y) = D_{\mathrm{KL}}\big(\Pr(X,Y) \parallel \Pr(X)\Pr(Y)\big)
         = H(Y) - H(Y \mid X)
         = H(X) + H(Y) - H(X,Y).
$$

It is symmetric, non-negative, and zero iff $X$ and $Y$ are independent. It captures *any* form of dependence, not just linear correlation. In feature selection, $I(X_j; Y)$ measures how informative feature $j$ is about the target — unlike correlation, it works for categorical variables and non-linear relationships.

### 5.6 The data processing inequality

If $X \to Y \to Z$ forms a Markov chain (Z depends on X only through Y), then:

$$
I(X; Z) \le I(X; Y).
$$

Processing data cannot increase information. Every transformation — dimensionality reduction, quantisation, passing through a neural network layer — can only lose or preserve information, never create it. This is a sanity check for ML pipelines: if a later layer has *more* mutual information with the input than an earlier layer, something is wrong.

---

## 6. Probability inequalities

These let you bound probabilities when you do not know the full distribution.

### 6.1 Markov's inequality

For a non-negative random variable $X$ and $a > 0$:

$$
\Pr(X \ge a) \le \frac{\mathbb{E}[X]}{a}.
$$

Crude but universal. Requires only that $X \ge 0$ and you know its mean. Tight only for specific distributions. This is the building block for all other inequalities.

### 6.2 Chebyshev's inequality

For any random variable $X$ with finite mean $\mu$ and variance $\sigma^2$, and $k > 0$:

$$
\Pr(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}.
$$

Equivalently: $\Pr(|X - \mu| \ge \epsilon) \le \frac{\sigma^2}{\epsilon^2}$. The probability of being more than $k$ standard deviations from the mean is at most $1/k^2$. For a Gaussian, the corresponding probability is far smaller (e.g. ~0.3% for $k=3$, vs Chebyshev's 11%). Chebyshev is distribution-free but loose.

### 6.3 Chernoff bound

For a sum of independent random variables, the tail probability decays exponentially:

$$
\Pr\!\left(\sum_{i=1}^n X_i \ge \epsilon\right) \le \min_{t>0} e^{-t\epsilon} \prod_{i=1}^n \mathbb{E}[e^{t X_i}].
$$

This is the workhorse behind most learning-theoretic guarantees: Hoeffding's inequality, PAC bounds, and concentration results. The key insight: the moment generating function $\mathbb{E}[e^{tX}]$ controls how sharply the average concentrates around the mean.

### 6.4 Hoeffding's inequality

For independent bounded random variables $X_i \in [a_i, b_i]$:

$$
\Pr\!\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}[X]\right| \ge \epsilon\right) \le 2\exp\!\left(-\frac{2n^2\epsilon^2}{\sum_i (b_i - a_i)^2}\right).
$$

The sample mean converges to the true mean at rate $O(1/\sqrt{n})$, with exponential confidence. This is the formal basis for the intuition that "more data means better estimates."

### 6.5 Jensen's inequality

For a convex function $\phi$:

$$
\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)].
$$

The inequality reverses for concave functions. Examples: $\mathbb{E}[X^2] \ge (\mathbb{E}[X])^2$ (so variance is non-negative); $\mathbb{E}[\log X] \le \log \mathbb{E}[X]$; and critically for ML, $\mathbb{E}[-\log p(X)] \ge -\log \mathbb{E}[p(X)]$, which underlies the evidence lower bound (ELBO) in variational inference.

---

## 7. Conditional expectation

### 7.1 Definition

The conditional expectation $\mathbb{E}[Y \mid X = x]$ is the average of $Y$ over the conditional distribution $f_{Y\mid X}(y \mid x)$:

$$
\mathbb{E}[Y \mid X = x] = \begin{cases}
\sum_y y \cdot \Pr(Y = y \mid X = x) & \text{(discrete)} \\[6pt]
\int y \, f_{Y\mid X}(y \mid x)\, dy & \text{(continuous)}
\end{cases}
$$

### 7.2 The law of iterated expectations

Also called the **tower rule**:

$$
\mathbb{E}[Y] = \mathbb{E}_X\!\big[\mathbb{E}[Y \mid X]\big].
$$

The unconditional expectation is the average of the conditional expectations, weighted by how often each condition occurs. This is one of the most useful identities in probability — it decomposes a hard unconditional problem into easier conditional ones.

### 7.3 Conditional variance

The law of total variance:

$$
\operatorname{Var}(Y) = \underbrace{\mathbb{E}_X\!\big[\operatorname{Var}(Y \mid X)\big]}_{\text{average within-group variance}}
                      + \underbrace{\operatorname{Var}_X\!\big(\mathbb{E}[Y \mid X]\big)}_{\text{between-group variance}}.
$$

The first term is the unexplained variation (noise), the second is the explained variation (signal). The ratio $\operatorname{Var}_X(\mathbb{E}[Y\mid X]) / \operatorname{Var}(Y)$ is the $R^2$ of regressing $Y$ on $X$ — it measures how much of $Y$'s variance $X$ explains.

---

## 8. Common probability distributions

Every distribution models a particular data-generating story. Know the story, not just the formula.

### 8.1 Discrete distributions

| Distribution | PMF | Parameters | Story |
|---|---|---|---|
| Bernoulli | $\Pr(X=x) = p^x (1-p)^{1-x}$ | $p \in [0,1]$ | One coin flip; $x \in \{0,1\}$. |
| Binomial | $\Pr(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$ | $n \in \mathbb{N}$, $p \in [0,1]$ | $n$ independent coin flips; $k$ successes. |
| Poisson | $\Pr(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda > 0$ | Count of rare events in a fixed interval. |
| Geometric | $\Pr(X=k) = (1-p)^{k-1} p$ | $p \in [0,1]$ | Number of flips until first success. |
| Categorical | $\Pr(X=k) = p_k$ | $\sum p_k = 1$ | One roll of a $K$-sided die. |
| Multinomial | $\Pr(X_1=n_1,\ldots) = \frac{n!}{\prod n_k!} \prod p_k^{n_k}$ | $n$, $p_1,\ldots,p_K$ | $n$ rolls of a $K$-sided die. |

### 8.2 Continuous distributions

| Distribution | PDF | Parameters | Story |
|---|---|---|---|
| Gaussian (Normal) | $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ | $\mu \in \mathbb{R}$, $\sigma^2 > 0$ | Sum of many small independent effects (CLT). |
| Exponential | $f(x) = \lambda e^{-\lambda x}$ | $\lambda > 0$ | Waiting time until the next event in a Poisson process. Memoryless. |
| Beta | $f(x) \propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\alpha,\beta > 0$ | Distribution over probabilities; conjugate prior for Bernoulli/Binomial. |
| Dirichlet | $f(\mathbf{p}) \propto \prod_k p_k^{\alpha_k-1}$ | $\alpha_k > 0$ | Distribution over probability vectors; conjugate prior for Categorical/Multinomial. |
| Gamma | $f(x) \propto x^{\alpha-1} e^{-\beta x}$ | $\alpha,\beta > 0$ | Conjugate prior for Poisson rate and Exponential rate; waiting time for $\alpha$ events. |

### 8.3 The Gaussian in context

Why is the Gaussian everywhere?

1. **Central Limit Theorem:** the sum (or average) of many independent, finite-variance random variables tends toward a Gaussian, regardless of the original distribution.
2. **Maximum entropy:** among all distributions with fixed mean and variance, the Gaussian has the largest entropy — it makes the fewest assumptions.
3. **Closed under linear operations:** linear combinations of Gaussians are Gaussian.
4. **Mathematical convenience:** the log-density is quadratic, making MLE analytically solvable and conjugate to itself.

The practical danger: the Gaussian has exponentially thin tails. In finance, real returns have fat tails — a Gaussian model drastically underestimates extreme events.

---

## 9. Connecting to machine learning

Most supervised learning problems are conditional estimation problems:

| Task | What you are modelling |
|---|---|
| Binary classification | $\Pr(Y = 1 \mid X = x)$ |
| Multi-class classification | $\Pr(Y = k \mid X = x)$ for $k = 1,\ldots,K$ |
| Regression | $\mathbb{E}[Y \mid X = x]$ or the full conditional $f_{Y\mid X}(y \mid x)$ |
| Density estimation | $f_X(x)$ (unconditional) or $f_{X\mid Z}(x \mid z)$ |

From this perspective:

- **Logistic regression** directly models $\Pr(Y=1\mid X=x)$ with a linear function passed through a sigmoid.
- **Linear regression** models $\mathbb{E}[Y\mid X=x]$ as a linear function, assuming Gaussian errors.
- **Neural networks** with softmax outputs model $\Pr(Y=k\mid X=x)$; with linear outputs, they model $\mathbb{E}[Y\mid X=x]$.
- **Decision trees** partition the feature space and estimate a constant conditional mean or class probability in each region.
- **Bayesian methods** place priors on parameters and compute $\Pr(\theta \mid \mathcal{D})$ to capture uncertainty.

The loss function you minimise is typically the negative log-likelihood under the chosen conditional model. This is why understanding conditional probability is prerequisite to understanding ML.

---

## 10. Common traps

| Trap | What it looks like | How to catch it |
|---|---|---|
| Confusing $\Pr(A\mid B)$ with $\Pr(B\mid A)$ | "Most defaulted loans had low income, so low-income borrowers always default." | Write both conditionals explicitly. Draw the 2×2 table. |
| Base-rate neglect | A test that is "95% accurate" for a disease with 1% prevalence — most positives are false. | Always apply Bayes with the true prior. |
| Simpson's paradox | A trend appears in every subgroup but reverses when groups are combined. | Check for a lurking confounding variable. Condition on it. |
| Gambler's fallacy | "I've had five losses; I am due for a win." | Independence means the next trial does not remember the past. |
| Prosecutor's fallacy | $\Pr(\text{evidence} \mid \text{innocent})$ is small, therefore $\Pr(\text{innocent} \mid \text{evidence})$ is small. | This ignores the prior — the prior probability of innocence is overwhelmingly high. |
| Assuming independence | "Features are uncorrelated, so the naïve Bayes assumption holds." | Uncorrelated ≠ independent. Test conditional independence. |
| Ignoring the denominator | Comparing posteriors across models without the evidence $\Pr(\mathcal{D})$. | For model selection, use the Bayes factor: the ratio of evidences. |
| Confusing marginal and conditional independence | Two variables are independent marginally but dependent given a third. | Always state which variables are in the conditioning set. |

---

## 11. Interview practice

### 9.1 The response pattern

For probability questions:

1. **Name the events and their probabilities.** Define clear notation.
2. **Identify what is given and what is asked.** Which conditional direction?
3. **Apply Bayes or the chain rule.** Write the formula before plugging in numbers.
4. **Check for reasonableness.** Does the answer respect base rates? Is it between 0 and 1?
5. **Explain the intuition.** Why did the probability shift the way it did?

### 9.2 Calculation: medical testing

**Question.** Disease prevalence 0.5%, test sensitivity 98%, specificity 97%. A patient tests positive. What is $\Pr(\text{Disease} \mid +)$?

**Answer.**

$$
\Pr(+) = 0.98 \times 0.005 + 0.03 \times 0.995 = 0.0049 + 0.02985 = 0.03475.
$$

$$
\Pr(D \mid +) = \frac{0.0049}{0.03475} \approx 14.1\%.
$$

More than 85% of positives are false positives. The low prevalence dominates the high test accuracy.

### 9.3 Calculation: two tests

**Question.** Same setting. The patient takes a second, independent test and it is also positive. Now what is $\Pr(D \mid ++)$?

**Answer.** Use the posterior from the first test as the new prior:

$$
\Pr(D \mid ++) = \frac{0.98 \times 0.141}{0.98 \times 0.141 + 0.03 \times 0.859} = \frac{0.1382}{0.1382 + 0.0258} \approx 84.3\%.
$$

Two independent positive tests push the probability from ~14% to ~84%. This is how Bayesian updating works: each new piece of evidence multiplies the odds.

### 9.4 Concept: "What does 'conditional' mean?"

**Answer.** It means restricting the sample space to only those outcomes consistent with the condition, then re-normalising so probabilities sum to one. Conditioning is *zooming in* — it is not causation, and it is not a chronological relationship. $\Pr(A \mid B)$ can describe a relationship even when $B$ happens after $A$, as long as $B$ is informative about $A$.

### 9.5 Design: "How would you estimate the probability of default for a new credit applicant?"

**Answer.** "This is a conditional probability estimation problem: $\Pr(\text{Default} \mid \text{features})$. I would start with logistic regression because it directly models this conditional probability with a linear combination of features and produces calibrated probabilities. I would validate calibration with a reliability diagram, check discrimination with AUC, and ensure the training data is representative of the applicant population. If linearity is too restrictive, I would use gradient boosted trees, but I would still check calibration — trees can produce poorly calibrated probabilities."

### 9.6 Trick: "If $\Pr(A \mid B) > \Pr(A)$, does $\Pr(B \mid A) > \Pr(B)$?"

**Answer.** Yes. From the definition:

$$
\Pr(A \mid B) > \Pr(A) \iff \frac{\Pr(A,B)}{\Pr(B)} > \Pr(A) \iff \Pr(A,B) > \Pr(A)\Pr(B).
$$

The condition is symmetric in $A$ and $B$. So $\Pr(B \mid A) = \frac{\Pr(A,B)}{\Pr(A)} > \frac{\Pr(A)\Pr(B)}{\Pr(A)} = \Pr(B)$. Intuitively: if $B$ makes $A$ more likely, then $A$ makes $B$ more likely.

### 9.7 Derivation: law of total probability

**Question.** Derive the denominator in Bayes' theorem.

**Answer.** For a partition $\{B_i\}$:

$$
\Pr(A) = \sum_i \Pr(A, B_i) = \sum_i \Pr(A \mid B_i)\,\Pr(B_i).
$$

The first equality uses the fact that the $B_i$ partition the space: $A = \cup_i (A \cap B_i)$, and these are disjoint. The second applies the chain rule to each term.

### 9.8 Intuition: "Why does naïve Bayes work when its assumptions are wrong?"

**Answer.** The probability *estimates* are often miscalibrated, but the probability *ranking* — which class is more likely — often survives because the errors in the numerator tend to be correlated across classes. For classification, only the $\arg\max$ matters, not the absolute probabilities. Naïve Bayes can also work well with limited data because the independence assumption drastically reduces the number of parameters to estimate, trading bias for variance.

---

## 12. Cheat sheet

| Concept | Remember |
|---|---|
| Conditional probability | $\Pr(A \mid B) = \frac{\Pr(A \cap B)}{\Pr(B)}$ |
| Chain rule | $\Pr(A,B) = \Pr(A \mid B)\,\Pr(B) = \Pr(B \mid A)\,\Pr(A)$ |
| Law of total probability | $\Pr(A) = \sum_i \Pr(A \mid B_i)\,\Pr(B_i)$ |
| Bayes' theorem | $\Pr(A \mid B) = \frac{\Pr(B \mid A)\,\Pr(A)}{\Pr(B)}$ |
| Posterior $\propto$ Likelihood × Prior | $\Pr(\theta \mid \mathcal{D}) \propto \Pr(\mathcal{D} \mid \theta)\,\Pr(\theta)$ |
| Independence | $\Pr(A,B) = \Pr(A)\,\Pr(B)$ |
| Conditional independence | $\Pr(A,B \mid C) = \Pr(A \mid C)\,\Pr(B \mid C)$ |
| Law of iterated expectations | $\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y \mid X]]$ |
| Law of total variance | $\operatorname{Var}(Y) = \mathbb{E}[\operatorname{Var}(Y\mid X)] + \operatorname{Var}(\mathbb{E}[Y\mid X])$ |
| MLE vs MAP vs Full Bayes | MLE maximises likelihood; MAP adds prior; Full Bayes averages. |
| Base-rate fallacy | Ignoring $\Pr(A)$ when computing $\Pr(A \mid B)$. |
| Simpson's paradox | A trend reverses when conditioning on a lurking variable. |

---

## 13. Final checklist

Before using a conditional probability result, verify:

- Have I correctly identified which event is the condition and which is the target? $\Pr(A \mid B)$ is not $\Pr(B \mid A)$.
- Have I accounted for the base rate (prior)? If prevalence is low, a high-accuracy test still produces mostly false positives.
- Are there lurking variables that could create Simpson's paradox or spurious correlations?
- If I am assuming independence or conditional independence, have I tested it? Uncorrelated does not mean independent.
- Is the partition in my law-of-total-probability calculation genuinely exhaustive and mutually exclusive?
- Have I checked that my answer is between 0 and 1, and consistent with the base rates?
- Can I explain the result in plain language — why did the probability shift?
- If this is for an ML model, is the conditional probability calibrated? Does $\hat{p} = 0.8$ mean 80% empirical frequency?

If you can define the events, write Bayes' theorem, compute the normalising constant, and explain why the probability shifted, you have the conditional probability fluency that most quantitative interviews demand.
