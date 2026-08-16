# Chapter 1 — Risk-Neutral Pricing, Martingales & Numeraires

Derivative pricing begins with a constraint, not a forecast: two portfolios with the same future cash flows must have the same price today. Otherwise one can buy the cheaper portfolio, sell the expensive one, and lock in profit without net risk.

Risk-neutral probability is the mathematical representation of this no-arbitrage constraint. It is not the claim that investors are indifferent to risk.

## 1.1 A one-period market

Let a stock cost $S_0$ today and be worth $S_u$ or $S_d$ next period. A risk-free dollar grows to $R=1+r$ dollars. A derivative pays $H_u$ or $H_d$.

Replicate the derivative with $\Delta$ shares and $B$ dollars in the bank:

$$
\Delta S_u+BR=H_u,\qquad
\Delta S_d+BR=H_d.
$$

Solving,

$$
\Delta=\frac{H_u-H_d}{S_u-S_d},
$$

and the no-arbitrage price is $H_0=\Delta S_0+B$. The derivative's expected payoff under the real probabilities never entered. Replication fixed the price.

## 1.2 State prices and risk-neutral probabilities

Let $\pi_u,\pi_d$ be prices today of securities paying one dollar in only the up or down state. Then

$$
H_0=\pi_uH_u+\pi_dH_d.
$$

Because a risk-free dollar pays one in either state,

$$
\pi_u+\pi_d=\frac1R.
$$

Define $q=R\pi_u$ and $1-q=R\pi_d$. Then

$$
H_0=\frac1R\left[qH_u+(1-q)H_d\right].
$$

$q$ behaves like a probability and is determined by making the discounted stock a fair game:

$$
S_0=\frac1R[qS_u+(1-q)S_d],
$$

so

$$
q=\frac{RS_0-S_d}{S_u-S_d}.
$$

No arbitrage requires $0<q<1$, equivalent to $S_d<RS_0<S_u$.

## 1.3 Worked binomial price

Take $S_0=100$, $S_u=120$, $S_d=90$, and $R=1.05$. A call with strike 100 pays 20 or 0. Then

$$
q=\frac{105-90}{120-90}=0.5,
$$

and

$$
C_0=\frac{0.5(20)+0.5(0)}{1.05}=9.5238.
$$

Replication gives $\Delta=20/30=2/3$. The bank position satisfies $B(1.05)=-60$, so $B=-57.1429$ and $C_0=(2/3)100-57.1429=9.5238$.

Expectation under $q$ and replication are two descriptions of the same price.

## 1.4 Equivalent martingale measures

A probability measure $Q$ is **equivalent** to the real-world measure $P$ if they agree on which events are impossible. It is a **martingale measure** relative to the money-market numeraire when discounted tradable prices are martingales:

$$
\widetilde S_t=\frac{S_t}{B_t},\qquad
E^Q[\widetilde S_T\mid\mathcal F_t]=\widetilde S_t.
$$

The fundamental theorem of asset pricing links:

- absence of arbitrage to existence of an equivalent martingale measure;
- market completeness to uniqueness of that measure, under technical conditions.

If many martingale measures exist, some claims cannot be replicated uniquely. No-arbitrage gives a price range or requires an additional preference, calibration, or hedging criterion.

## 1.5 Conditional pricing

For a replicable payoff $H_T$ and deterministic short rate $r$,

$$
V_t=e^{-r(T-t)}E^Q[H_T\mid\mathcal F_t].
$$

Conditioning matters: the price is updated using all information available at $t$. Under stochastic rates, discounting remains inside the expectation:

$$
V_t=E^Q\left[e^{-\int_t^T r_sds}H_T\mid\mathcal F_t\right].
$$

This equation is a computational recipe only after the risk-neutral dynamics and payoff are specified.

## 1.6 From real-world to risk-neutral dynamics

Suppose under $P$,

$$
dS_t=\mu S_tdt+\sigma S_tdW_t^P.
$$

In the Black–Scholes market, changing to $Q$ replaces the expected return $\mu$ by the risk-free rate:

$$
dS_t=rS_tdt+\sigma S_tdW_t^Q.
$$

The change in Brownian drift encodes the market price of risk. Volatility remains because it determines hedge ratios and payoff dispersion. The physical drift is essential for forecasting and portfolio choice, but not for pricing a perfectly replicable claim.

With a continuous dividend yield $d$, the risk-neutral stock drift becomes $r-d$.

## 1.7 Numeraires and measure changes

A **numeraire** is a strictly positive traded asset used as the unit of account. Under the measure associated with numeraire $N_t$, any tradable price expressed in units of $N_t$ is a martingale:

$$
\frac{V_t}{N_t}
=E^{Q^N}\left[\frac{V_T}{N_T}\middle|\mathcal F_t\right].
$$

Choosing the money-market account gives the usual risk-neutral measure. Choosing a zero-coupon bond maturing at $T$ gives the $T$-forward measure, often simplifying interest-rate and forward-payoff calculations.

Changing numeraire changes probabilities and drift terms, but not an arbitrage-free price when calculations are consistent.

## 1.8 Martingales are relative to information and units

A process $M_t$ is a martingale if

$$
E[M_t\mid\mathcal F_s]=M_s,\qquad s\le t.
$$

This “fair game” property depends on the measure, filtration, and numeraire. An undiscounted stock is generally not a martingale under the money-market risk-neutral measure; the discounted stock is.

Confusing these choices is a common source of incorrect drifts and discount factors.

## 1.9 Pricing versus forecasting

| Question | Typical measure | Central object |
|---|---|---|
| What return distribution may occur? | Physical $P$ | Estimated real-world dynamics |
| What price avoids arbitrage? | Pricing measure $Q$ | Discounted expected replicating payoff |
| How should risk be hedged? | Both may matter | Sensitivities and residual P&L |
| How should capital be allocated? | Physical plus preferences | Expected utility or risk objective |

Risk-neutral probabilities are usually inferred from market prices. They contain risk premia and should not be read directly as real-world event forecasts.

## 1.10 Failure modes

- Treating $Q$ as a psychological belief that investors are risk-neutral.
- Discounting outside an expectation when interest rates are stochastic and correlated with the payoff.
- Using physical drift inside a replication price.
- Forgetting dividends, funding, collateral, or tradability assumptions.
- Claiming a unique price in an incomplete market without an extra criterion.
- Calling an undiscounted asset a martingale without naming measure and numeraire.

## 1.11 Knowledge checks

1. Derive the binomial risk-neutral probability and its no-arbitrage bounds.
2. Why can a risk-averse market still use risk-neutral pricing?
3. What does uniqueness of the martingale measure imply?
4. Why does changing numeraire change drift but not price?
5. Distinguish a pricing probability from a real-world forecast.

### Solution outlines

1. Enforce $S_0=R^{-1}[qS_u+(1-q)S_d]$; $q\in(0,1)$ iff $S_d<RS_0<S_u$.
2. Risk preferences affect underlying prices and risk premia; replication removes claim-specific risk from its arbitrage-free price.
3. Under suitable conditions, every contingent claim is replicable and has a unique arbitrage-free price.
4. Payoffs and prices are expressed in a new unit and the associated measure changes consistently; converting back preserves value.
5. Pricing probabilities reproduce traded prices under no-arbitrage; physical probabilities describe actual frequencies under a model.

## 1.12 What to retain

- Replication is the economic argument; risk-neutral expectation is its probabilistic form.
- Discounted tradables are martingales under the appropriate pricing measure.
- Measure, filtration, and numeraire must always be named.
- Physical and pricing measures answer different questions.
- Incomplete markets require more than no-arbitrage to choose one price.

Next: [Chapter 2 — Monte Carlo Foundations](ch2-monte-carlo-foundations-viewer.html).
