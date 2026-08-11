# Stochastic Processes: Random Walks, Martingales & Brownian Motion

> From coin flips to continuous time: how randomness accumulates over time, which processes are fair games, and why the log of a stock price is the model that makes everything tractable.

## 1. The symmetric random walk

Start at $S_0 = 0$. Each step, flip a fair coin: up $+1$ with probability $1/2$, down $-1$ with probability $1/2$:

$$
S_n = X_1 + X_2 + \cdots + X_n, \qquad X_i = \pm 1
$$

**The two facts that define the process:**

$$
\mathbb{E}[S_n] = 0, \qquad \operatorname{Var}(S_n) = n
$$

The mean stays flat while the standard deviation grows like $\sqrt n$. This is the single most important number to internalise about randomness over time: **fluctuation grows with the square root of time, not with time.** After 10,000 steps the walk is typically $\pm 100$ from start — not $\pm 10{,}000$.

## 2. Hitting probabilities: the gambler's ruin

A gambler starts with $i$ units, plays until reaching $N$ (wins) or 0 (ruin). Win probability per round $p$, lose probability $q = 1 - p$. The probability of ruin is:

$$
P(\text{ruin} \mid i) = \begin{cases}
\dfrac{(q/p)^i - (q/p)^N}{1 - (q/p)^N} & p \neq q \\[10pt]
1 - \dfrac{i}{N} & p = q
\end{cases}
$$

**Why (sketch):** let $r_i$ be the ruin probability starting at $i$. By conditioning on the first step, $r_i = p r_{i+1} + q r_{i-1}$, with $r_0 = 1$, $r_N = 0$. Solving this linear recurrence gives the formula. The boundary conditions are doing the work.

**Worked examples:**

- Fair game, start 90 of 100: $P(\text{ruin}) = 1 - 90/100 = 10\%$. A fair game still ends in ruin 10% of the time — and someone else's 90% win is your 10% loss.
- Slightly unfair, $p = 0.45$, start 90 of 100: with $(q/p)^{90}$ astronomically large, the formula collapses to roughly $1 - (q/p)^{-10} \approx 1 - 0.8 = 0.19$... let us compute precisely: $(q/p) = 0.55/0.45 = 11/9 \approx 1.2222$. Then

$$
r = \frac{(11/9)^{90} - (11/9)^{100}}{1 - (11/9)^{100}} \approx 1 - \frac{1}{(11/9)^{10}} \approx 1 - 0.133 = 0.867
$$

A 5-cent edge per round turns a 10% ruin probability into an 87% one. **The drift dominates at long horizons — small edges compound.**

**Expected duration of the fair game** (start $i$, target $N$): $i(N - i)$ steps. Starting mid-way ($i = N/2$) the game lasts $N^2/4$ steps on average.

## 3. Martingales: the fair games

A process $M_n$ is a **martingale** (with respect to the information up to $n$) if:

$$
\mathbb{E}[M_{n+1} \mid \mathcal F_n] = M_n
$$

In words: *given everything you know now, tomorrow's expected value is today's value.* A martingale has no drift — it is a fair game, and no strategy based on the past can improve expected wealth.

**The three canonical examples:**

1. **The symmetric random walk:** $\mathbb{E}[S_{n+1} \mid S_n] = S_n$. Fair.
2. **The product martingale:** if the steps are $+1$ (prob $p$) / $-1$ (prob $q$), then $\left(\frac{q}{p}\right)^{S_n}$ is a martingale. *(Check: multiply the current value by $q/p$ with probability $p$ and by $p/q$ with probability $q$ — the expected multiplier is $p(q/p) + q(p/q) = q + p = 1$.)* This is the engine behind the gambler's-ruin formula: the product martingale is constant in expectation, so its boundary values pin down $r_i$.
3. **Discounted stock prices under the risk-neutral measure:** $e^{-rt} S_t$ is a martingale. This is the mathematical content of "no arbitrage" — see the options pricing lesson.

**Optional stopping (stated, not proven):** if a bounded-time strategy stops a martingale at a fair stopping time, the stopped value has the same expectation as the start. The gambler's ruin formula is optional stopping applied to the product martingale.

## 4. Brownian motion: the continuous-time limit

Take a random walk and shrink it: step size $\sqrt{\Delta t}$, step frequency $1/\Delta t$. As $\Delta t \to 0$, the walk converges to **Brownian motion** (the Wiener process) $W_t$ with the defining properties:

1. **$W_0 = 0$** and $W_t$ is continuous.
2. **Independent increments:** $W_{t_2} - W_{t_1}$ is independent of $W_{t_1} - W_{t_0}$ for non-overlapping intervals.
3. **Normal increments:** $W_{t_2} - W_{t_1} \sim \mathcal N(0,\ t_2 - t_1)$ — variance grows linearly with elapsed time.

The scaling laws carry over from the random walk:

$$
\mathbb{E}[W_t] = 0, \qquad \operatorname{Var}(W_t) = t, \qquad \operatorname{Cov}(W_s, W_t) = \min(s, t)
$$

Brownian motion is continuous but nowhere differentiable — its paths wiggle so much that no slope exists. Formally, $(dW_t)^2 = dt$, a fact whose consequences show up in Ito's lemma below.

## 5. Geometric Brownian motion: the standard stock model

If the *log* of the price is a Brownian motion with drift, the price itself follows **geometric Brownian motion** (GBM):

$$
\frac{dS_t}{S_t} = \mu\, dt + \sigma\, dW_t
$$

The percentage return per unit time has drift $\mu$ and volatility $\sigma$. The solution (via Ito's lemma) is:

$$
S_t = S_0 \exp\!\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

**The two numbers to keep straight:**

- **Mean of the price:** $\mathbb{E}[S_t] = S_0 e^{\mu t}$ — grows at the *arithmetic* rate $\mu$.
- **Mean of the log-price:** $\mathbb{E}[\ln S_t] = \ln S_0 + (\mu - \sigma^2/2)t$ — grows at the *geometric* rate $\mu - \sigma^2/2$.

The gap $\sigma^2/2$ is the volatility drag: because $e^{x}$ is convex, the average of the price is larger than the price of the average log-return (Jensen's inequality). It is also why a 50% loss requires a 100% gain to recover — arithmetic vs geometric returns.

**The log-return view:** over one period, $\ln(S_{t+1}/S_t) \sim \mathcal N(\mu - \sigma^2/2,\ \sigma^2)$ — log-returns are normal, returns are log-normal, prices are always positive.

## 6. Ito's lemma: the chain rule of stochastic calculus

In ordinary calculus, $d(\ln S) = dS/S$. In stochastic calculus there is a correction:

$$
d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\, dW_t
$$

The general statement: if $X_t$ satisfies $dX_t = a\, dt + b\, dW_t$ and $f$ is twice differentiable, then

$$
df(X_t) = \left(f'(X_t)a + \tfrac12 f''(X_t) b^2\right) dt + f'(X_t) b\, dW_t
$$

The extra $\tfrac12 f'' b^2 dt$ term is the second-order effect: over an infinitesimal time, $(dW_t)^2 = dt$ does not vanish (unlike $(dt)^2$), so the Taylor expansion keeps the second derivative. This is the correction that produces the $-\sigma^2/2$ in the log-price.

**Worked example.** For GBM with constant $\mu, \sigma$ and $f(x) = \ln x$: $f' = 1/x$, $f'' = -1/x^2$, $b = \sigma S$, so

$$
df = \left(\frac{1}{S}\mu S - \frac{1}{2}\frac{1}{S^2}\sigma^2 S^2\right)dt + \frac{1}{S}\sigma S\, dW = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\, dW
$$

Integrating from 0 to $t$ gives the GBM solution above.

## 7. Markov vs martingale: two different "no memory" properties

| | Statement | Finance meaning |
|---|---|---|
| **Markov** | $P(X_{t+1} \mid \text{history}) = P(X_{t+1} \mid X_t)$ | The *distribution* of the next step depends only on the current state. |
| **Martingale** | $E[X_{t+1} \mid \text{history}] = X_t$ | The *expectation* of the next step is the current value. |

Markov is about the full conditional distribution; martingale is only about the conditional mean. A process can be one without the other: a random walk with drift is Markov but not a martingale; a process whose distribution depends on the whole path (e.g., GARCH) can still be a martingale. Under the risk-neutral measure, prices are martingales; under the physical measure, they are typically Markov but not martingales.

## 8. The toolkit in one page

| Process | Definition | Key facts |
|---|---|---|
| Random walk | $S_n = \sum X_i$, $X_i = \pm1$ | $E[S_n] = 0$, $\operatorname{Var}(S_n) = n$ |
| Gambler's ruin | absorb at 0 or N | $r_i = \frac{(q/p)^i - (q/p)^N}{1-(q/p)^N}$; fair: $1 - i/N$ |
| Martingale | $E[M_{n+1} \mid \mathcal F_n] = M_n$ | no drift; no strategy beats it in expectation |
| Brownian motion | $dW_t \sim \mathcal N(0, dt)$, independent increments | $\operatorname{Var}(W_t) = t$; $(dW)^2 = dt$ |
| GBM | $dS/S = \mu dt + \sigma dW$ | $S_t = S_0 e^{(\mu - \sigma^2/2)t + \sigma W_t}$ |
| Ito's lemma | $df = (f'a + \tfrac12 f''b^2)dt + f'b\, dW$ | the $\sigma^2/2$ correction |

## 9. Practice

1. A random walk of 2,500 fair steps: what is the typical distance from the start? *(Answer: $\sqrt{2500} = 50$.)*
2. Fair game, start 75, target 100: probability of ruin? *(Answer: $1 - 75/100 = 25\%$.)*
3. Same game with $p = 0.48$: ruin probability? *(Answer: $q/p = 0.52/0.48 = 13/12 \approx 1.0833$. Write $r = \frac{(13/12)^{75} - (13/12)^{100}}{1 - (13/12)^{100}}$ and divide top and bottom by $(13/12)^{100}$: $r = \frac{1 - (12/13)^{25}}{1 - (12/13)^{100}}$. Now $(12/13)^{25} \approx 0.1356$ and $(12/13)^{100} \approx 0.00034$, so $r \approx \frac{0.8644}{0.9997} \approx 0.865$. A 2-cent edge per round turns the fair-game 25% ruin probability into ~86.5%.)*

4. A stock has $\mu = 10\%$, $\sigma = 20\%$ annually, $S_0 = 100$. Expected price in one year? *(Answer: $100 e^{0.10} \approx 110.52$.)* The median price? *(Answer: $100 e^{0.10 - 0.02} \approx 108.33$ — the mean exceeds the median by the volatility drag.)*
5. Show that $(q/p)^{S_n}$ is a martingale for the biased walk. *(Answer: conditional expected multiplier is $p \cdot q/p + q \cdot p/q = q + p = 1$.)*
6. Verify numerically that the sample mean of many independent GBM paths approximates $S_0 e^{\mu t}$. *(Use a Monte Carlo: simulate 100,000 paths of daily GBM for one year, average the terminal prices, compare to $S_0 e^{\mu t}$.)*
