# Markov Chains for Quant Finance

> A practical introduction to the mathematics, the model-risk traps, and the interview questions that matter.

## 1. Why quants care

A Markov chain is a model for a system that moves among a finite or countable set of states. Its defining shortcut is simple:

\[
\Pr(X_{t+1}=j\mid X_t=i, X_{t-1},\ldots,X_0)
=\Pr(X_{t+1}=j\mid X_t=i).
\]

Conditional on the **current state** $X_t$, the model says the older history adds no information about the next state. This is the *Markov property*.

That is an assumption, not a fact about markets. It can be useful because it turns a complicated time series into a tractable state-transition problem. It can also be dangerously wrong when your state leaves out volatility, macro conditions, order-book shape, time since a rating change, or other predictive information.

Typical financial uses include:

- credit-rating migration and default;
- market-regime modelling, usually with states that are not observed directly;
- order-book / execution models based on spread, queue, imbalance, and fill states;
- operational risk, customer churn, and lifecycle models;
- scenario generation where the state is a discretised economic or risk regime.

The central modelling question is therefore not "can I fit a transition matrix?" It is:

> What state representation makes the next-step dynamics approximately memoryless at the decision horizon I care about?

---

## 2. Core objects and notation

We use a **discrete-time**, time-homogeneous chain with states $1,\ldots,K$. A time step might be one day, month, event, or trade; its meaning must be fixed before estimation.

### 2.1 Transition matrix

Let

\[
P_{ij}=\Pr(X_{t+1}=j\mid X_t=i).
\]

The $K\times K$ matrix $P$ is the **transition matrix**. We use the row-vector convention:

- $P_{ij}\ge 0$;
- every row sums to one: \(\sum_j P_{ij}=1\);
- a distribution is a row vector \(\mu_t\), so \(\mu_{t+1}=\mu_tP\).

The $n$-step probability is

\[
\Pr(X_{t+n}=j\mid X_t=i)=(P^n)_{ij}.
\]

This is the Chapman–Kolmogorov idea in matrix form: to get from $i$ to $j$ in two steps, sum over every possible intermediate state.

### 2.2 A small worked example

Suppose a stylised market has two **observed** regimes: Calm $C$ and Stressed $S$, with a daily transition matrix

\[
P=
\begin{pmatrix}
0.95 & 0.05\\
0.20 & 0.80
\end{pmatrix}.
\]

The first row says a calm day is followed by stress with probability 5%; the second says stress persists with probability 80%. The probability of stress two days after a calm day is

\[
(P^2)_{CS}=0.95(0.05)+0.05(0.80)=0.0875.
\]

The extra 4 percentage points relative to 5% arises because the chain can enter stress tomorrow and remain there.

### 2.3 Stationary distribution

A distribution \(\pi\) is **stationary** if

\[
\pi=\pi P, \qquad \sum_i\pi_i=1.
\]

For the two-state example, write \(\pi=(\pi_C,\pi_S)\). The balance condition is

\[
0.05\pi_C=0.20\pi_S,
\]

which gives \(\pi=(0.80,0.20)\). In a long, stable sample, the chain spends 20% of days in stress *under this model*. It does **not** say that the probability of stress is always 20%, nor that the real market has a fixed stationary law.

### 2.4 Conditional, marginal, and transition probability

Keep these distinct in interviews:

- \(P_{ij}\): a conditional, one-step transition probability;
- \(\mu_t(j)\): the unconditional probability of state $j$ at time $t$, given the initial distribution;
- \(\pi_j\): a stationary marginal probability, if a suitable stationary distribution exists.

---

## 3. The chain properties that become modelling traps

The words below are not just theory. Each reveals a potential error in a financial interpretation.

| Property / trap | Mathematical meaning | Quant consequence | Diagnostic and response |
|---|---|---|---|
| Absorbing state | \(P_{ii}=1\) | Default is often modelled as absorbing. Recovery, cure, restructuring, and exit rules then need explicit treatment. | Check whether "default" is truly terminal at the stated horizon. Add post-default states if the use case needs recoveries. |
| Reducible chain | Some states cannot communicate | A portfolio may split into classes that never transition under the fitted matrix; stationary conclusions can depend on where it starts. | Draw the transition graph; identify communicating classes. Do not report one global long-run distribution blindly. |
| Periodic chain | Returns to a state only in multiples of an integer $d>1$ | A stylised alternating state produces oscillating probabilities rather than convergence. | Check cycle structure or examine powers of $P$. Use an aperiodic state definition / time scale if convergence is needed. |
| Non-ergodic / no unique limiting law | Usually reducibility or periodicity prevents ordinary convergence | "Long-run average" or "equilibrium risk" can be ill-defined or initial-state dependent. | Establish irreducibility and aperiodicity before using a unique stationary distribution. |
| Non-stationary transitions | $P$ changes with calendar time or covariates | A matrix fitted in low-rate years can fail in recession or policy shifts. | Estimate rolling matrices; condition on macro covariates; compare held-out transition frequencies. |
| State aggregation | Important information is omitted from $X_t$ | Coarse ratings or return bins can create fake memory: transitions depend on age, volatility, or path. | Test whether richer current states reduce residual dependence; use duration or covariates if they do. |
| Sparse transitions | Some counts are near zero | Maximum-likelihood rows become noisy; impossible-looking transitions may be sampling artefacts. | Report counts and uncertainty, shrink / pool sensibly, and stress rare-event assumptions. |

### 3.1 Absorption and expected time to default

Suppose transient states come first and absorbing states last. Partition the matrix as

\[
P=\begin{pmatrix}Q&R\\0&I\end{pmatrix}.
\]

Here $Q$ describes movements among non-absorbing states. The **fundamental matrix** is

\[
N=(I-Q)^{-1}=I+Q+Q^2+\cdots.
\]

Its entry $N_{ij}$ is the expected number of visits to transient state $j$ when starting in $i$. The vector of expected times to absorption is

\[
\mathbf{t}=N\mathbf{1}.
\]

This is useful for default or churn models, but only if absorption is economically meaningful. A credit process that lets firms emerge from default is not an absorbing chain unless "default" has been defined as a terminal accounting event.

### 3.2 The first important finance warning: returns are rarely Markov by themselves

A raw return sequence $r_t$ generally does not become well modelled by \(\Pr(r_{t+1}\mid r_t)\). Volatility clustering, jumps, intraday seasonality, leverage effects, liquidity, and latent information often matter. A better state may be something like

\[
X_t=(\text{volatility bucket},\ \text{trend bucket},\ \text{liquidity bucket},\ \text{macro regime}).
\]

But adding variables increases data needs and can create sparse cells. State design is a bias–variance trade-off, not a quest for maximum granularity.

---

## 4. Building a useful model: start with the decision

Before fitting $P$, write down five things.

1. **Decision:** What will the model support—one-year expected credit loss, execution routing, stress testing, or regime-aware allocation?
2. **State:** What information is available at $t$, and why should it be sufficient for the next transition?
3. **Clock:** Calendar time or event time? Daily, monthly, trade-to-trade? Your estimated probabilities change with this choice.
4. **Target:** One-step prediction, multi-horizon scenarios, long-run occupancy, first-passage risk, or control?
5. **Loss from error:** Is a missed default, an under-estimated tail transition, or a wrongly delayed execution most costly?

### 4.1 Estimation in the simplest case

For observed states, let $n_{ij}$ be the number of observed transitions from $i$ to $j$. The maximum-likelihood estimator is

\[
\widehat P_{ij}=\frac{n_{ij}}{\sum_j n_{ij}}.
\]

This is intuitive: each row is a multinomial probability estimate. Yet direct frequency estimates are often fragile in finance because the important transitions—deep downgrade, default, liquidity collapse—are scarce.

Practical alternatives include:

- pooling related obligors or regimes while checking that the pooling assumption is defensible;
- Bayesian / Dirichlet smoothing, with transparent priors;
- shrinkage toward an industry matrix or a structural baseline;
- parameterising $P$ using covariates rather than estimating a separate unconditioned matrix;
- bootstrapping, block bootstrapping, or Bayesian posterior simulation to express uncertainty.

Never "fix" an implausible estimated row by silently forcing a preferred probability. If you impose monotonicity, a floor, a stress overlay, or a through-the-cycle adjustment, document it as a model choice and test its impact.

### 4.2 What good validation looks like

A credible validation set goes beyond checking that rows sum to one.

- **Transition-frequency calibration:** group out-of-sample observations by current state and compare realised next-state frequencies with predicted rows.
- **Multi-step calibration:** compare $P^h$ with realised $h$-horizon transitions; one-step fit does not guarantee multi-step fit.
- **Markov check:** after conditioning on $X_t$, test whether $X_{t-1}$, duration in the state, volatility, or macro variables still predict $X_{t+1}$. Residual predictive power indicates a missing state variable or a non-Markov process.
- **Stability / regime check:** compare matrices across time windows, sectors, rating agencies, rate cycles, and stressed periods. Use formal uncertainty rather than only eyeballing differences.
- **Plausibility / monotonicity:** for ordered credit ratings, verify that the structure is economically sensible; do not confuse plausibility with empirical validation.
- **Sensitivity:** perturb rare-transition estimates, time-step choices, and state boundaries. Report the impact on the downstream quantity—PD, VaR contribution, execution cost—not only on entries of $P$.
- **Benchmark:** compare to a simpler persistence model, an empirical marginal model, and any relevant conditional model. A Markov chain earns its complexity by improving a decision-relevant score.

---

## 5. Quant-finance applications

### 5.1 Credit-rating migration

**State.** Credit ratings such as AAA, AA, …, CCC, Default. The time step is commonly one year or one month.

**Use.** Estimate migration probabilities, cumulative default probability, expected loss, and portfolio rating distribution. Default is often set as absorbing for a specific risk horizon.

**Mechanics.** If today's rating distribution is \(\mu_0\), then the $h$-period distribution is \(\mu_0P^h\). If Default is state $D$, \((P^h)_{iD}\) is the model-implied cumulative probability of reaching default by $h$ when default is absorbing.

**Key trap.** A rating alone is rarely sufficient. Time spent in rating, outlook/watch status, leverage, sector, and macro conditions can all predict transitions. Also distinguish:

- **point-in-time** matrices, responsive to conditions; from
- **through-the-cycle** matrices, smoothed for capital or planning.

Using one as though it were the other is a model-governance failure.

### 5.2 Regime models for returns and volatility

**State.** A low-vol / high-vol or risk-on / risk-off regime. The state may be latent rather than observed.

**Use.** Conditional risk forecasts, scenario generation, asset allocation overlays, and regime-dependent return distributions.

**Key trap.** If the regime is inferred from the same noisy returns it is meant to explain, state uncertainty matters. Treating estimated labels as ground truth gives overconfident parameters and optimistic backtests.

**Better extension.** Use a Hidden Markov Model (HMM): latent $X_t$ follows a Markov chain, while the observable $Y_t$ is generated from a state-dependent distribution—for example, \(Y_t\mid X_t=k\sim\mathcal N(\mu_k,\sigma_k^2)\), or a heavier-tailed alternative. Estimate states probabilistically and evaluate both filtered real-time predictions and smoothed in-sample explanations.

### 5.3 Order-book and execution models

**State.** Spread, best-quote queue sizes, imbalance, recent market-order sign, and perhaps short-term volatility—usually binned to avoid a huge state space.

**Use.** Estimate fill probability, adverse-selection risk, short-horizon price move, or the expected cost of passive versus aggressive execution.

**Key trap.** Sampling in calendar time can hide event dynamics. A trade-time or quote-update-time chain may be more appropriate. But the order book is highly non-stationary across time of day and events; include session state or condition the matrix. Avoid accidentally using future book information in state construction.

### 5.4 Stress testing and scenario simulation

**State.** Discrete macroeconomic, funding, liquidity, or portfolio-loss regimes.

**Use.** Simulate paths, estimate time spent under stress, and calculate first-passage or cumulative-loss metrics.

**Key trap.** A historical matrix can understate an unprecedented stress transition. For stress testing, the right question is not merely "what did the MLE estimate?" but "what transition structure is coherent under the scenario?" Use controlled overlays, expert judgement, and sensitivity analysis—and identify them clearly as non-historical assumptions.

---

## 6. Advanced extensions: what changes and why

| Tool | When it helps | Main idea | Watch-out |
|---|---|---|---|
| Continuous-time Markov chain (CTMC) | Events happen irregularly; credit intensity and operational processes | A generator $G$ has non-negative off-diagonals and rows summing to zero; \(P(t)=e^{tG}\). | Not every empirical discrete matrix is exactly $e^{G}$ for a valid generator; time-homogeneity can still fail. |
| HMM / Markov-switching model | Regimes are not directly observed | Latent chain plus state-dependent observation model. Forward filtering estimates real-time state probabilities. | Label switching, local optima, state uncertainty, and unrealistic emission distributions. |
| Non-homogeneous chain | Transitions move with macro variables, age, time of day, or season | $P_t$, or $P(z_t)$, changes with covariates. | More flexibility can overfit; future covariates may be unavailable at forecast time. |
| Semi-Markov / duration model | Sojourn time matters | Transition chance depends on time already spent in state. | A simple chain implies geometric holding times, often unrealistic for ratings and regimes. |
| Markov decision process (MDP) | You control actions such as execution aggressiveness | Transitions depend on state and action, \(P(j\mid i,a)\); optimise expected reward. | Backtest the policy with costs, constraints, and realistic market impact; do not confuse prediction with control. |
| Coupled / factor models | Many obligors or assets move together | Common latent factors or conditional matrices create dependence. | Independent-chain portfolio simulations can dramatically understate clustered defaults and liquidity stress. |

### 6.1 Continuous-time chains in one paragraph

For a CTMC, the generator $G$ satisfies $G_{ij}\ge0$ for $i\ne j$, $G_{ii}=-\sum_{j\ne i}G_{ij}$, and

\[
P(t)=\exp(tG).
\]

The off-diagonal $G_{ij}$ is an instantaneous transition intensity, not a probability. For a small interval \(\Delta t\), \(P_{ij}(\Delta t)\approx G_{ij}\Delta t\) when $i\ne j$. CTMCs avoid arbitrary resampling of irregular events, but they introduce an embedding/calibration problem: a monthly transition matrix may not correspond exactly to any valid continuous-time generator.

### 6.2 The state-space explosion problem

Adding state variables is the natural response when the Markov property fails. But if you use 10 volatility bins, 10 trend bins, 5 liquidity bins, 4 macro states, and 5 time-of-day buckets, you now have 10,000 states. Most transition rows will be empty.

Ways to manage this:

- begin with the smallest state that passes decision-relevant diagnostics;
- pool or hierarchically model similar states;
- use features in a conditional transition model rather than a giant Cartesian-product state;
- use regularisation and out-of-sample selection;
- keep a state-definition version history—small boundary changes alter estimated transitions.

---

## 7. Interview practice

### 7.1 The response pattern that interviewers like

For a modelling question, answer in this order:

1. **Define the state and clock.** "I would model monthly rating states, with Default absorbing for this one-year ECL calculation."
2. **State the assumption.** "Conditional on current rating and selected covariates, the next state is independent of older history."
3. **Show the mechanics.** "Estimate row-wise transition probabilities, then use $P^h$ for an $h$-period horizon."
4. **Name the failure mode.** "Rating age and the macro regime may violate the simple Markov assumption."
5. **Say how you would test or improve it.** "Test residual duration / macro predictability, compare out-of-sample calibration, then use a semi-Markov or conditional model if material."

This is much stronger than launching directly into eigenvectors or saying "I would use an HMM" without explaining why.

### 7.2 Calculation question: two-step transition

**Question.** A chain has

\[
P=\begin{pmatrix}0.7&0.3\\0.4&0.6\end{pmatrix}.
\]

What is the probability of being in state 2 after two steps, starting in state 1?

**Answer outline.** Sum the two paths:

\[
0.7(0.3)+0.3(0.6)=0.39.
\]

Or compute \((P^2)_{12}\). Say why multiplication works: it sums over all intermediate states.

### 7.3 Derivation question: stationary distribution

**Question.** For the same matrix, find the stationary distribution.

**Answer outline.** Let \(\pi=(a,1-a)\). The first component of \(\pi=\pi P\) gives

\[
a=0.7a+0.4(1-a),
\]

so $0.7a=0.4$, hence \(a=4/7\) and \(\pi=(4/7,3/7)\). Check non-negativity and that the components sum to one. If asked about convergence, note that this finite chain is irreducible and aperiodic because all entries are positive.

### 7.4 Concept question: "How would you model rating migration?"

**Strong answer.** "Start with rating grades plus Default as states and select the horizon based on the risk use case. Estimate a row-stochastic matrix from transitions, use $P^h$ for multi-period migration, and quantify uncertainty because default rows are sparse. But I would not assume ratings alone are Markov without testing it: migration depends on sector, macro regime, watch status, and rating age. I would validate one- and multi-horizon calibration by rating and economic regime. Depending on the application, I would add covariates, use a duration model, or distinguish point-in-time from through-the-cycle matrices. For a portfolio, I would also add dependence; independent obligor chains miss default clustering."

### 7.5 Critique question: "The one-year matrix fits perfectly. Is the model good?"

**Answer outline.** No. Ask:

- Does it calibrate out of sample, including stressed windows?
- Does $P^h$ match $h$-period migration?
- Are enough observations available in rare rows and tail transitions?
- Is the current state sufficient, or does history / duration add predictive power?
- Is the time-homogeneity assumption reasonable?
- Does the model improve the actual business decision versus a simpler benchmark?

### 7.6 Design question: "How do you make a non-Markov process Markov?"

**Answer outline.** Enlarge the state to include the predictive history or a sufficient statistic. For an AR(2)-like process, \(X_t=(r_t,r_{t-1})\) can be Markov even when $r_t$ alone is not. In finance, include volatility, duration, macro conditions, or order-book features only if they are available at decision time and improve out-of-sample prediction. Then mention the cost: state-space explosion and sparse data.

### 7.7 Trick question: "Does a stationary distribution mean the process is stationary?"

**Answer outline.** No. A chain may possess a stationary distribution, but its actual distribution is stationary only when it is initialised at that distribution. In addition, an estimated financial process may be non-stationary even if a fitted constant matrix has a stationary vector.

### 7.8 Coding / practical interview prompt

**Prompt.** "You receive a table of entity, date, and rating. How would you estimate and validate a monthly transition matrix?"

**Answer outline.**

1. Sort each entity by date; align valid consecutive monthly observations; define treatment for missing, withdrawn, and default labels before counting.
2. Count only valid $t\to t+1$ transitions; keep the count matrix next to the probability matrix.
3. Row-normalise, flag zero-count rows, and decide whether pooling / smoothing is justified.
4. Split train and test chronologically, not randomly; calculate one- and multi-month calibration by initial rating.
5. Compare stable and stressed windows; quantify uncertainty for sparse default transitions.
6. Version the state mapping, data exclusions, and any matrix overlays.

---

## 8. Cheat sheet

| Item | Remember |
|---|---|
| One-step dynamics | \(\mu_{t+1}=\mu_tP\) under the row-vector convention. |
| $h$-step dynamics | \(\mu_{t+h}=\mu_tP^h\). |
| Stationary distribution | Solve \(\pi=\pi P\) and \(\sum_i\pi_i=1\). |
| Unique limiting distribution (finite-state intuition) | Look for irreducible + aperiodic chain. |
| Absorbing-chain times | Partition $P$, then \(N=(I-Q)^{-1}\), \(\mathbf t=N\mathbf1\). |
| CTMC | \(P(t)=e^{tG}\); $G$'s off-diagonals are intensities, not probabilities. |
| Markov assumption | The next step depends only on a well-designed current state—not merely on the most recent raw observation. |
| Interview red flag | Treating an in-sample transition matrix or stationary vector as proof of a stable real-world process. |

## 9. Final modelling checklist

Before relying on a Markov-chain result, be able to answer all of these:

- What exactly is the state, what information was discarded, and why is that acceptable?
- What does one time step mean, and is calendar time or event time appropriate?
- Are rows estimated from enough data—especially for adverse transitions?
- Is Default / exit genuinely absorbing for the decision horizon?
- Do past history, duration, covariates, or calendar regime still predict the next state after conditioning on the proposed state?
- Is the matrix stable enough for the forecast horizon? If not, what conditional or scenario overlay is justified?
- Does $P^h$ calibrate, not just $P$?
- Are cross-entity dependencies material for the portfolio quantity being reported?
- Does the chain improve a decision-relevant outcome over a simpler benchmark?

If you can explain those choices plainly, carry out the matrix calculations, and identify where the assumptions break, you have the practical Markov-chain fluency most quant interviews are looking for.
