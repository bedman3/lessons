# Options Pricing: Risk-Neutral Valuation & Black-Scholes

> From payoffs and no-arbitrage to the binomial model and the Black-Scholes formula: why prices are expectations under a changed probability, and how every Greek tells you something about risk.

## 1. Payoffs and notation

An option gives the right (not obligation) to trade at a fixed strike $K$ on a fixed date $T$. The payoff at expiry for a stock priced $S_T$:

| Instrument | Payoff at $T$ | Profit picture |
|---|---|---|
| Call | $\max(S_T - K, 0)$ | Gains above $K$, loses only the premium below |
| Put | $\max(K - S_T, 0)$ | Gains below $K$ |
| Forward | $S_T - F$ | Linear: both directions |
| Long stock | $S_T - S_0$ | Linear |

Paying the premium makes options *non-linear*: the payoff diagram is kinked at $K$. That kink is the entire business — convexity in the payoff, protection on one side, participation on the other.

## 2. No-arbitrage: the only law that matters

An arbitrage is a portfolio with:
1. zero initial cost,
2. non-negative payoff in every state,
3. strictly positive payoff in at least one state.

The no-arbitrage assumption — such portfolios do not exist in functioning markets — is the foundation of all pricing. Its immediate consequences:

**Forward price.** To guarantee delivery of a stock at $T$, borrow $S_0$, buy the stock, owe $S_0 e^{rT}$ at $T$. The forward must price at:

$$
F = S_0 e^{rT}
$$

If $F$ were higher, you could sell the forward and buy the stock with borrowed money for a riskless profit.

**Put-call parity.** A portfolio of call + bond and put + stock have identical payoffs at expiry:

$$
C - P = S_0 - K e^{-rT}
$$

**Worked example.** $S_0 = 100$, $K = 105$, $r = 5\%$, $T = 1$:

$$
C - P = 100 - 105e^{-0.05} = 100 - 99.879 = 0.121
$$

If the market quotes $C - P = 0.50$ instead, buy the cheap side and sell the rich side for a locked-in profit. Put-call parity is an identity, not a model — it must hold regardless of the stock's future distribution.

## 3. The binomial model: pricing by replication

Assume the stock moves to one of two values over one period: $S_u = S_0 u$ (up) or $S_d = S_0 d$ (down), with $d < 1 < u$ (after accounting for the risk-free rate $r$). We want the call price $C$ with strike $K$.

**Step 1 — replicate the call with Δ shares and $B$ cash:**

$$
\Delta S_u + B e^{r\Delta t} = C_u, \qquad
\Delta S_d + B e^{r\Delta t} = C_d
$$

**Step 2 — solve for the hedge ratio Δ:**

$$
\Delta = \frac{C_u - C_d}{S_u - S_d}
$$

This is the famous **delta** — shares per option. It is a ratio of payoff differences to price differences, and it is why option pricing is a *hedging* problem, not a forecasting problem: the hedge removes the stock's randomness entirely.

**Step 3 — rearrange into the risk-neutral expectation.** Solving both equations and simplifying:

$$
C = e^{-r\Delta t}\Big(p^* C_u + (1 - p^*) C_d\Big), \qquad
p^* = \frac{e^{r\Delta t} - d}{u - d}
$$

$p^*$ is the **risk-neutral probability**: the value of $p$ for which the stock's expected return under $p^*$ equals the risk-free rate. The option price is the *discounted expected payoff under the risk-neutral measure* — the real-world probability of up-moves never appears.

**Worked example.** $S_0 = 100$, $u = 1.1$, $d = 0.9$, $r = 5\%$, $K = 100$, one year.

$$
p^* = \frac{e^{0.05} - 0.9}{1.1 - 0.9} = \frac{1.05127 - 0.9}{0.2} = 0.75635
$$

$$
C_u = \max(110 - 100, 0) = 10, \qquad C_d = \max(90 - 100, 0) = 0
$$

$$
C = e^{-0.05}\big(0.75635 \cdot 10 + 0.24365 \cdot 0\big) = 0.95123 \cdot 7.5635 = 7.19
$$

Delta: $\Delta = (10 - 0)/(110 - 90) = 0.5$ shares.

**Two-step binomial:** repeat recursively backward from the terminal payoffs. With $n$ steps the binomial price converges to the Black-Scholes price as $n \to \infty$ — the binomial model is Black-Scholes in miniature.

## 4. Black-Scholes: the continuous-time limit

**Assumptions:** the stock follows geometric Brownian motion $dS/S = \mu dt + \sigma dW$; constant risk-free rate; no dividends; no transaction costs; continuous trading.

Under these assumptions the no-arbitrage price of a European call is:

$$
C = S_0\, N(d_1) - K e^{-rT}\, N(d_2)
$$

$$
d_1 = \frac{\ln(S_0/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T
$$

and for the put, by put-call parity:

$$
P = K e^{-rT} N(-d_2) - S_0\, N(-d_1)
$$

**Reading the formula:**

- $N(d_1)$ is the probability (under the risk-neutral measure) that the option ends in the money — adjusted; it is also the hedge ratio **delta**.
- $N(d_2)$ is the probability of ending in the money unadjusted.
- The formula is a *replicating portfolio*: own $N(d_1)$ shares, borrow $K e^{-rT} N(d_2)$ dollars.

**Worked example.** $S_0 = 100$, $K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1$.

$$
d_1 = \frac{0 + (0.05 + 0.02)}{0.2} = 0.35, \qquad d_2 = 0.15
$$

$$
N(0.35) = 0.6368, \qquad N(0.15) = 0.5596
$$

$$
C = 100(0.6368) - 100 e^{-0.05}(0.5596) = 63.68 - 95.123(0.5596) = 63.68 - 53.23 = 10.45
$$

Delta of this call: $N(d_1) = 0.6368$ — to hedge one call, short 0.6368 shares; the hedge is self-financing because the hedge ratio is updated continuously.

## 5. The Greeks: sensitivities that manage risk

Each Greek is a partial derivative of the price — Taylor expansion in disguise:

| Greek | Definition | What it measures | Sign intuition |
|---|---|---|---|
| Delta | $\frac{\partial C}{\partial S} = N(d_1)$ | Price sensitivity to the stock | Between 0 and 1 for a call |
| Gamma | $\frac{\partial^2 C}{\partial S^2}$ | How fast delta changes | Large near the money, near expiry |
| Theta | $\frac{\partial C}{\partial t}$ | Time decay | Usually negative for long options |
| Vega | $\frac{\partial C}{\partial \sigma}$ | Sensitivity to volatility | Always positive for long calls/puts |
| Rho | $\frac{\partial C}{\partial r}$ | Sensitivity to rates | Small for equity options |

**Worked example (theta/vega intuition).** An at-the-money 1-month option on a 20%-vol stock: gamma and vega are at their largest, theta bites hardest. The combination is the trader's daily P&L statement:

$$
dC \approx \Delta\, dS + \tfrac12 \Gamma\, dS^2 + \Theta\, dt + \mathcal V\, d\sigma
$$

Delta-hedged (long stock hedged by shorting $\Delta$ shares), the remaining P&L is driven by $\Gamma$ (moves) and $\Theta$ (time) — which is why gamma scalping is a trade and not a gamble: the expected value of $\Gamma dS^2/2$ exactly offsets $-\Theta dt$ under the risk-neutral measure.

## 6. Implied volatility: the market's own view

Invert the Black-Scholes formula: given the market price, solve for the $\sigma$ that reproduces it. This **implied volatility** is the market's collective assessment of future uncertainty. Two empirical facts:

1. **The volatility smile:** implied vol is not constant across strikes — it is typically U-shaped (low near the money, higher in the wings). The log-normal model does not fully describe real markets; the smile is the market pricing fat tails and crash risk.
2. **Term structure:** implied vol varies with maturity. The whole surface (strike × maturity) is a richer object than any single number — and the surface is what options desks actually trade.

The smile's existence is a caution: Black-Scholes is a *pricing convention* (how the market quotes) and a *hedging framework*, not a literal description of the world.

## 7. The toolkit in one page

| Tool | Formula | When |
|---|---|---|
| Forward | $F = S_0 e^{rT}$ | No-arbitrage pricing |
| Put-call parity | $C - P = S_0 - Ke^{-rT}$ | Identities across instruments |
| Binomial price | $C = e^{-r\Delta t}(p^*C_u + (1-p^*)C_d)$, $p^* = \frac{e^{r\Delta t} - d}{u-d}$ | Discrete trees, American options, intuition |
| Black-Scholes call | $S_0 N(d_1) - Ke^{-rT}N(d_2)$ | Continuous-time European pricing |
| Delta hedge | $\Delta = N(d_1) = \frac{\partial C}{\partial S}$ | The hedging recipe |
| Implied volatility | invert BS for $\sigma$ | The market's volatility quote |

## 8. Practice

1. $S_0 = 50$, $K = 50$, $r = 4\%$, $T = 1$. What is $C - P$? *(Answer: $50 - 50e^{-0.04} = 50 - 48.04 = 1.96$.)*
2. One-step binomial: $S_0 = 100$, $u = 1.2$, $d = 0.8$, $r = 10\%$, $K = 100$. Find the call price. *(Answer: $p^* = (e^{0.1} - 0.8)/(0.4) = 0.7632$; $C_u = 20$, $C_d = 0$; $C = e^{-0.1}(0.7632 \cdot 20) = 0.9048 \cdot 15.264 = 13.81$; $\Delta = 20/40 = 0.5$.)*
3. Black-Scholes sanity: what happens to the call price as $\sigma \to 0$? *(Answer: $d_1, d_2 \to \infty$ when $S_0 > Ke^{-rT}$, so $C \to S_0 - Ke^{-rT}$ — the intrinsic value of the forward-like claim; with zero volatility the option is a forward.)*
4. A call has delta 0.6 and gamma 0.05. You are long 1,000 calls. How many shares hedge the delta? *(Answer: short 600 shares.)* If the stock jumps +1, roughly what is the new delta? *(Answer: $0.6 + 0.05 \cdot 1 = 0.65$ — gamma updates the hedge.)*
5. Why does the real-world drift $\mu$ not appear in the Black-Scholes price? *(Answer: the option is priced by replication — a delta hedge removes the stock's randomness, so the price cannot depend on the expected return of an asset the hedge does not hold. The only distributional input that survives is volatility.)*
