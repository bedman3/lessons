# Chapter 2 — Monte Carlo Foundations: Estimation, Paths & Error

Monte Carlo converts an expectation into an average of simulated outcomes. Its appeal is generality: once a payoff can be simulated, even high-dimensional and path-dependent prices become approachable. Its danger is that several errors—sampling, time discretization, model, and implementation—can look alike.

## 2.1 The basic estimator

If

$$
V_0=E^Q[D_T H(X)],
$$

where $D_T$ is a discount factor, simulate independent copies $Y_i=D_T^{(i)}H(X^{(i)})$ and estimate

$$
\widehat V_N=\frac1N\sum_{i=1}^NY_i.
$$

The law of large numbers gives convergence under integrability. With finite variance, the CLT gives

$$
\frac{\widehat V_N-V_0}{s_Y/\sqrt N}\approx N(0,1),
$$

so an approximate 95% interval is

$$
\widehat V_N\pm1.96\frac{s_Y}{\sqrt N}.
$$

The $N^{-1/2}$ rate is slow: reducing standard error by ten requires roughly one hundred times as many paths.

## 2.2 A European call under GBM

Under risk-neutral geometric Brownian motion with dividend yield $d$,

$$
S_T=S_0\exp\left[(r-d-\tfrac12\sigma^2)T+\sigma\sqrt T Z\right],
\qquad Z\sim N(0,1).
$$

The call estimator is

$$
\widehat C_0=e^{-rT}\frac1N\sum_{i=1}^N(S_T^{(i)}-K)^+.
$$

Because the terminal law is exact, this estimator has sampling error but no time-discretization error. It can be checked against the Black–Scholes formula—a valuable implementation benchmark.

```python
def european_call_mc(s0, strike, rate, dividend, vol, expiry, normals):
    payoffs = []
    for z in normals:
        terminal = s0 * exp((rate - dividend - 0.5 * vol**2) * expiry
                            + vol * sqrt(expiry) * z)
        payoffs.append(max(terminal - strike, 0.0))
    mean = sum(payoffs) / len(payoffs)
    return exp(-rate * expiry) * mean
```

## 2.3 Random numbers and transformations

Pseudorandom generators produce deterministic sequences from a seed. Reproducibility requires recording generator type, seed, parallel-stream policy, and transformation method.

Uniform draws can be transformed into other distributions by inverse CDF:

$$
X=F^{-1}(U),\qquad U\sim\operatorname{Uniform}(0,1).
$$

Normal generators often use Box–Muller, rejection, or specialized algorithms. Correlated Gaussian shocks can be constructed with a Cholesky factor $L$ of correlation matrix $\Sigma$:

$$
Z=L\varepsilon,\qquad \varepsilon\sim N(0,I).
$$

A non-positive-definite estimated correlation matrix must be diagnosed or repaired; silently forcing a factorization changes the model.

## 2.4 Simulating stochastic differential equations

For

$$
dX_t=a(X_t,t)dt+b(X_t,t)dW_t,
$$

Euler–Maruyama on step $\Delta t$ is

$$
X_{t+\Delta t}=X_t+a(X_t,t)\Delta t+b(X_t,t)\sqrt{\Delta t}\,Z.
$$

Milstein adds a correction when derivatives of $b$ are available:

$$
\tfrac12 b\,\partial_xb\,[(\Delta W)^2-\Delta t].
$$

Exact transition sampling is preferable when known. Discretizing GBM directly can create negative stock values; simulating log-price or using the exact exponential step preserves positivity.

## 2.5 Strong and weak convergence

**Strong convergence** measures pathwise error using the same Brownian path:

$$
E[|X_T^{\Delta t}-X_T|].
$$

**Weak convergence** measures error in expectations:

$$
|E[g(X_T^{\Delta t})]-E[g(X_T)]|.
$$

Option pricing usually needs weak accuracy, but barrier and stopping features can be sensitive to pathwise errors. Euler–Maruyama typically has strong order $1/2$ and weak order 1 under regularity conditions.

An order-$p$ bias behaves like $C(\Delta t)^p$ asymptotically. Halving the step should reduce bias by roughly $2^p$ once the asymptotic regime is reached.

## 2.6 Error decomposition

For a discretized Monte Carlo estimator,

$$
\widehat V_{N,\Delta t}-V
=\underbrace{(\widehat V_{N,\Delta t}-V_{\Delta t})}_{\text{sampling error}}
+\underbrace{(V_{\Delta t}-V)}_{\text{discretization bias}}.
$$

Increasing paths reduces the first term, not the second. Refining time steps reduces discretization bias but increases work per path. Model error and parameter-estimation error sit outside both terms.

## 2.7 Path-dependent payoffs

Asian options depend on an average, barriers depend on extrema or crossings, and lookbacks depend on path maxima/minima. Simulation must track sufficient state.

Discrete monitoring can miss a barrier crossing between grid points. Brownian-bridge corrections estimate crossing probability conditional on endpoints. Simply adding paths cannot fix a coarse monitoring grid.

For running averages, store the accumulated sum rather than the entire path when only the average is required. This reduces memory without changing the payoff.

## 2.8 Confidence intervals and diagnostics

Report price, standard error, path count, step size, seed policy, and confidence interval. Useful diagnostics include:

- repeat across independent seeds;
- plot estimated price and standard error versus $N$;
- refine the time grid and estimate convergence order;
- benchmark special cases against analytic prices;
- verify discounted underlying martingale conditions;
- check terminal moments and correlations;
- inspect payoff tails and contribution concentration.

A narrow Monte Carlo interval only quantifies simulation sampling error. It does not validate the model or calibration.

## 2.9 Computational budget

If cost is proportional to paths times steps, $C\propto N/\Delta t$. Choosing $N$ and $\Delta t$ requires balancing variance and squared bias:

$$
\operatorname{MSE}\approx\frac{\sigma_Y^2}{N}+C_b^2(\Delta t)^{2p}.
$$

Spending all computation on more paths while leaving dominant discretization bias is wasteful. Pilot runs estimate both components.

## 2.10 Failure modes

- Reporting too many decimal places relative to standard error.
- Using one seed and treating repeatability as statistical validation.
- Increasing path count to fix discretization bias.
- Missing barrier crossings between grid dates.
- Simulating a process with a scheme that violates positivity or boundaries.
- Sharing random streams incorrectly across parallel workers.
- Treating Monte Carlo confidence intervals as model uncertainty intervals.

## 2.11 Knowledge checks

1. Why does a 10× smaller standard error require about 100× paths?
2. Distinguish strong and weak convergence.
3. Name four conceptually different sources of pricing error.
4. Why can exact terminal simulation be insufficient for a barrier option?
5. How would you verify an Euler implementation empirically?

### Solution outlines

1. Standard error scales as $N^{-1/2}$.
2. Strong compares paths; weak compares expected functions of paths.
3. Sampling, time discretization, parameter calibration, model specification, and coding error are distinct examples.
4. The payoff depends on intermediate crossings, which terminal value alone does not reveal.
5. Compare against an exact transition or known price across successively halved steps and estimate the error slope.

## 2.12 What to retain

- Monte Carlo is statistical estimation, so every result needs a standard error.
- Sampling error and discretization bias require different remedies.
- Exact transitions and analytic benchmarks are powerful verification tools.
- Path-dependent payoffs require careful state and crossing treatment.
- Numerical confidence is narrower than model confidence.

Next: [Chapter 3 — Variance Reduction & Greeks](ch3-variance-reduction-greeks-viewer.html).
