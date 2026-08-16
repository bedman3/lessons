# Chapter 3 — Variance Reduction & Monte Carlo Greeks

Monte Carlo error is not determined by path count alone. If an estimator uses more information per simulated path, its variance can fall dramatically without changing the target expectation. Greeks add another layer: differentiating a noisy estimator can amplify variance or introduce bias.

## 3.1 The objective

For an unbiased estimator $\widehat V_N$,

$$
\operatorname{Var}(\widehat V_N)=\frac{\operatorname{Var}(Y)}{N}.
$$

Variance reduction replaces $Y$ with another random variable having the same expectation and smaller variance. A fourfold variance reduction is worth roughly four times as many paths.

Always compare methods at equal computational cost, including extra payoff evaluations and preprocessing.

## 3.2 Antithetic variates

For Gaussian driver $Z$, simulate a pair $Z$ and $-Z$ and average their payoffs:

$$
Y_A=\frac{g(Z)+g(-Z)}{2}.
$$

The estimator remains unbiased because $Z$ and $-Z$ have the same marginal law. Its variance is

$$
\operatorname{Var}(Y_A)
=\frac12\operatorname{Var}(g(Z))
+\frac12\operatorname{Cov}(g(Z),g(-Z)).
$$

Negative covariance helps. For a monotone payoff under a symmetric driver this often works well; for non-monotone payoffs it can help little or even hurt.

## 3.3 Control variates

Suppose $X$ is correlated with payoff $Y$ and $E[X]=\mu_X$ is known. Define

$$
Y_C=Y-\beta(X-\mu_X).
$$

Its expectation remains $E[Y]$. The variance-minimizing coefficient is

$$
\beta^*=\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)},
$$

giving variance reduction factor $1-\rho_{YX}^2$ in the ideal single-control case.

For a European call, discounted terminal stock is a natural control because

$$
E^Q[e^{-rT}S_T]=S_0e^{-dT}.
$$

Estimate $\beta$ on a pilot sample or account for using the same sample. Multiple controls turn the problem into a regression.

## 3.4 Conditional Monte Carlo

Replace a noisy payoff by its conditional expectation given part of the simulation:

$$
E[Y]=E[E[Y\mid Z]].
$$

The law of total variance shows

$$
\operatorname{Var}(E[Y\mid Z])\le\operatorname{Var}(Y).
$$

Analytically integrating the final time step, smoothing a digital payoff, or conditioning on a volatility path can reduce variance and improve Greek regularity.

## 3.5 Stratification and Latin hypercube sampling

Stratified sampling divides the unit interval or state space into regions and samples each deliberately. For $U\sim U(0,1)$, divide into $m$ equal strata and sample once within each. This prevents random clustering.

Latin hypercube sampling stratifies each marginal dimension while permuting combinations. It often improves low-dimensional smooth integrands but does not automatically control high-dimensional interactions.

Allocation can be proportional to stratum probability or optimized toward high-variance strata.

## 3.6 Importance sampling

Rare-event payoffs waste paths under the original measure. If density $q$ samples important regions more often than target density $p$,

$$
E_p[g(X)]
=E_q\left[g(X)\frac{p(X)}{q(X)}\right].
$$

The likelihood ratio $p/q$ restores the correct expectation. A poor proposal creates explosive weights and can increase variance or yield an estimator dominated by a few paths.

Monitor effective sample size, maximum weights, and tail contributions. Importance sampling is a modelling optimization problem, not merely a change of random seed.

## 3.7 Quasi-Monte Carlo

Quasi-random sequences such as Sobol points fill $[0,1]^d$ more evenly than pseudorandom samples. For sufficiently smooth, low-effective-dimensional integrands, convergence can beat the usual $N^{-1/2}$ behaviour.

Randomized scrambling restores replication-based error estimation. Brownian bridge or PCA constructions place important path variation in early quasi-random coordinates, reducing effective dimension.

Discontinuities, poor coordinate ordering, and very high effective dimension can erode gains.

## 3.8 Finite-difference Greeks

A central-difference delta is

$$
\widehat\Delta_h=\frac{\widehat V(S_0+h)-\widehat V(S_0-h)}{2h}.
$$

Small $h$ reduces truncation bias but amplifies independent simulation noise by division by $h$. **Common random numbers** use the same shocks for bumped valuations, making their errors positively correlated so the difference is much less noisy.

Choose $h$ by a stability study, not machine precision. Gamma is even more sensitive because it uses a second difference divided by $h^2$.

## 3.9 Pathwise differentiation

When payoff and simulated path are sufficiently differentiable,

$$
\frac{d}{d\theta}E[g(X_\theta)]
=E\left[g'(X_\theta)\frac{\partial X_\theta}{\partial\theta}\right].
$$

For a European call under GBM,

$$
\Delta=e^{-rT}E\left[\mathbf1_{\{S_T>K\}}\frac{S_T}{S_0}\right].
$$

The payoff kink is harmless with probability one for delta, but discontinuous payoffs such as digitals violate straightforward pathwise differentiation.

## 3.10 Likelihood-ratio method

If parameter $\theta$ changes density $p_\theta(x)$ rather than payoff form,

$$
\frac{d}{d\theta}E_\theta[g(X)]
=E_\theta\left[g(X)\frac{\partial}{\partial\theta}\log p_\theta(X)\right].
$$

This score-function estimator handles discontinuous payoffs but often has higher variance. Hybrid estimators combine pathwise and likelihood-ratio ideas.

## 3.11 Adjoint differentiation

Forward sensitivities propagate one input perturbation through all intermediate states. Reverse-mode or adjoint differentiation propagates one output sensitivity backward to many parameters at cost comparable to a small multiple of one valuation.

It is powerful for portfolios with many risk factors, but path discontinuities, exercise decisions, memory, and implementation consistency remain difficult. Automatic differentiation differentiates the implemented numerical scheme, which may not equal the derivative of the intended continuous model unless discretization is controlled.

## 3.12 Combining methods

Methods can reinforce one another:

- scrambled Sobol points plus Brownian bridge;
- common random numbers plus central differences;
- control variates applied to pathwise Greeks;
- conditional smoothing before differentiating a discontinuous payoff.

Measure variance reduction, bias, and runtime separately. A visually stable estimate can still be biased.

## 3.13 Failure modes

- Assuming antithetic pairs always reduce variance.
- Estimating a control mean from the same uncertain model as the payoff and calling it known.
- Using importance weights with infinite or enormous variance.
- Reporting quasi-Monte Carlo error from an iid formula without randomization.
- Choosing a finite-difference bump from one noisy run.
- Applying pathwise Greeks blindly to discontinuous payoffs.
- Differentiating code while ignoring discretization and exercise-policy changes.

## 3.14 Knowledge checks

1. Derive the optimal control-variate coefficient.
2. Why do common random numbers reduce finite-difference variance?
3. When is likelihood-ratio differentiation preferable to pathwise differentiation?
4. What does effective dimension mean for quasi-Monte Carlo?
5. Why can a smaller finite-difference bump increase total error?

### Solution outlines

1. Differentiate $\operatorname{Var}(Y-\beta X)$ with respect to $\beta$.
2. Positive correlation makes noise cancel in the bumped difference.
3. When the payoff is discontinuous but the density is differentiable in the parameter.
4. Most integrand variation is explained by a small subset or low-order combinations of coordinates.
5. Truncation bias falls, but simulation and floating-point noise are divided by a smaller number.

## 3.15 What to retain

- Variance reduction changes information efficiency, not the pricing target.
- Controls and conditioning exploit known expectations or analytic structure.
- Importance sampling targets rare but valuable regions and must control weights.
- Greek estimators trade differentiability, bias, variance, and cost.
- Every efficiency claim needs equal-cost empirical verification.

Next: [Chapter 4 — The Black–Scholes PDE & Feynman–Kac](ch4-black-scholes-pde-viewer.html).
