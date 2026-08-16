# Chapter 6 — American & Path-Dependent Options

European claims depend only on a fixed terminal payoff. American options add a decision: exercise now or continue. Path-dependent options add memory: the present spot alone no longer summarizes the payoff-relevant past.

Both problems require enlarging the pricing state or the algorithm.

## 6.1 Optimal stopping

For immediate exercise value $h(S_t)$, an American option satisfies

$$
V_t=\sup_{\tau\in\mathcal T_{t,T}}
E^Q\left[e^{-\int_t^\tau r_sds}h(S_\tau)\mid\mathcal F_t\right],
$$

where $\tau$ ranges over admissible stopping times. A valid exercise decision can use current and past information, not the future.

The value is the smallest supermartingale dominating the discounted exercise payoff—the Snell envelope.

## 6.2 Exercise and continuation regions

At each state compare

$$
\text{exercise value}=h(S)
$$

with

$$
\text{continuation value}=E^Q[e^{-r\Delta t}V_{t+\Delta t}\mid S_t=S].
$$

Exercise when the former is at least the latter. Their boundary is a **free boundary** because its location is part of the solution.

For a non-dividend-paying stock, early exercise of an American call is not optimal under standard assumptions: paying the strike early loses interest and sacrifices optionality. American puts can exercise early because receiving strike cash sooner can outweigh remaining optionality.

## 6.3 Binomial-tree solution

At maturity set

$$
V_N=h(S_N).
$$

At earlier nodes compute

$$
C=e^{-r\Delta t}[qV_u+(1-q)V_d]
$$

and

$$
V=\max(h(S),C).
$$

Backward induction automatically produces the exercise region. Recombining trees are simple and transparent, but convergence can oscillate with step count and parameter alignment.

## 6.4 Variational inequality

The American value satisfies

$$
V\ge h,
$$

$$
V_t+\mathcal L^QV-rV\le0,
$$

$$
(V-h)(V_t+\mathcal L^QV-rV)=0.
$$

In the continuation region, the pricing PDE holds with equality. In the exercise region, $V=h$. Numerically this is a linear complementarity problem.

Projected iterative methods solve a PDE step then enforce $V_i\leftarrow\max(V_i,h_i)$. Penalty methods approximate the obstacle with a large penalty. Tolerance and penalty size create additional numerical errors.

## 6.5 Least-squares Monte Carlo

Longstaff–Schwartz estimates continuation values from simulated paths:

1. simulate risk-neutral paths;
2. start from maturity cash flows;
3. move backward through exercise dates;
4. among in-the-money paths, regress discounted future realized cash flow on basis functions of current state;
5. exercise where immediate payoff exceeds fitted continuation;
6. discount chosen pathwise cash flows to today.

For state $S_t$, a basis might include $1,S_t,S_t^2$ or orthogonal polynomials. The regression estimates conditional expectation, not the option payoff directly.

## 6.6 Policy bias and out-of-sample evaluation

If the same paths fit and evaluate the exercise rule, overfitting can inflate estimated value. A cleaner procedure fits the policy on training paths and evaluates it on independent paths.

Any feasible stopping policy produces a **lower bound** because it may exercise suboptimally. Dual martingale methods can produce upper bounds. A tight lower–upper gap is stronger evidence than one regression estimate.

Basis choice, sample size, exercise-date density, and state representation all affect policy quality.

## 6.7 Making a path-dependent payoff Markov

A process is Markov only relative to its state. For an arithmetic Asian option, spot alone is insufficient, but augmented state

$$
(S_t,A_t),\qquad A_t=\int_0^tS_udu,
$$

is Markov under the usual diffusion model. For a lookback, track the running maximum. For a barrier, track whether the barrier has been hit.

State augmentation enables PDEs or dynamic programming but increases dimension. Monte Carlo stores path summaries more naturally.

## 6.8 Barrier options

Barrier payoffs depend on whether the path crosses a level. Discrete grids miss between-step crossings, biasing prices. Conditional on log-price endpoints under GBM, a Brownian bridge gives a crossing probability that can correct monitoring bias.

Near barriers, deltas and gammas can be large or discontinuous. Hedging risk is driven not only by terminal payoff but by the changing probability of activation or knockout.

## 6.9 Asian and lookback options

Asian averaging reduces sensitivity to one terminal print and often lowers volatility value. Geometric-average Asians have analytic formulas under GBM and make excellent controls for arithmetic-average Monte Carlo.

Lookbacks depend on running extrema and are particularly sensitive to time-grid monitoring. Exact or bridge-based simulation of extrema can materially improve accuracy.

## 6.10 Regression design in LSM

Good basis functions approximate continuation value over the states that influence the exercise boundary. Too few create approximation bias; too many overfit noisy realized future cash flows.

Diagnostics include:

- exercise-boundary shape and monotonicity;
- stability across bases and path counts;
- out-of-sample policy value;
- comparison with a tree/PDE in a low-dimensional benchmark;
- lower and upper bounds;
- distribution of exercised times and regression leverage.

Regress only on information available at the exercise time.

## 6.11 Failure modes

- Using future path information in an exercise feature.
- Fitting and valuing an LSM policy on the same paths without bias analysis.
- Assuming more basis functions always improve continuation estimates.
- Missing barrier crossings between simulation dates.
- Forgetting that state augmentation increases PDE dimension.
- Comparing American prices without matching exercise-date conventions.
- Reporting one lower-bound policy value as the exact optimum.

## 6.12 Knowledge checks

1. Why is American pricing an optimal-stopping problem?
2. Explain the complementarity conditions in the exercise and continuation regions.
3. Why does an LSM policy evaluated on new paths give a lower bound?
4. What state augmentation makes an arithmetic Asian Markov?
5. Why can a fine terminal distribution still misprice a barrier option?

### Solution outlines

1. The holder chooses an adapted exercise time to maximize discounted payoff.
2. Value dominates exercise; the PDE holds where continuing; value equals exercise where stopping.
3. It is one feasible, possibly suboptimal stopping policy, so the true supremum cannot be lower.
4. Add the running integral or average to current spot.
5. Barrier value depends on intermediate crossings that terminal calibration does not determine.

## 6.13 What to retain

- American value is exercise versus continuation under an admissible information set.
- The unknown exercise boundary makes the PDE an obstacle problem.
- LSM estimates conditional continuation and needs out-of-sample policy checks.
- Path dependence becomes Markov through sufficient state augmentation.
- Monitoring and policy errors are separate from ordinary Monte Carlo noise.

Next: [Chapter 7 — Volatility, Calibration & Model Risk](ch7-volatility-calibration-model-risk-viewer.html).
