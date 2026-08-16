# Chapter 4 — The Black–Scholes PDE & Feynman–Kac

The Black–Scholes formula is one solution to a deeper statement: a perfectly hedged derivative portfolio must earn the risk-free rate. That argument produces a partial differential equation. Risk-neutral expectation and the PDE are two representations of the same no-arbitrage price.

## 4.1 Model and claim

Assume a stock with continuous dividend yield $d$ follows

$$
dS_t=\mu S_tdt+\sigma S_tdW_t
$$

and the money-market account grows at rate $r$. Let a European derivative have value $V(t,S)$ and terminal payoff

$$
V(T,S)=\Phi(S).
$$

Assumptions include continuous trading, frictionless markets, known constant volatility and rates, and sufficient smoothness. The derivation is conditional on these idealizations.

## 4.2 Itô's lemma for the derivative

Applying Itô's lemma,

$$
dV
=\left(V_t+\mu SV_S+\tfrac12\sigma^2S^2V_{SS}\right)dt
+\sigma SV_SdW_t.
$$

Subscripts denote partial derivatives. The $V_{SS}$ term is the Itô correction caused by quadratic variation.

## 4.3 Delta hedging removes local randomness

Form a portfolio long one derivative and short $\Delta$ shares:

$$
\Pi=V-\Delta S.
$$

The stock pays dividend yield $d$, so the short-stock position affects financing. Choose

$$
\Delta=V_S.
$$

The $dW_t$ terms cancel. Over an infinitesimal interval the portfolio is locally riskless under the model, so no arbitrage requires it to earn $r$.

After accounting for dividends, the resulting PDE is

$$
V_t+(r-d)SV_S+\frac12\sigma^2S^2V_{SS}-rV=0.
$$

The physical drift $\mu$ disappeared. It affects the stock's forecast, but a continuously replicated claim is priced by hedge mechanics.

## 4.4 Interpreting every term

- $V_t$: time decay holding spot fixed.
- $(r-d)SV_S$: risk-neutral carry acting through delta.
- $\tfrac12\sigma^2S^2V_{SS}$: convexity exposure to quadratic variation.
- $-rV$: required financing return on derivative value.

Rearranging gives the delta-hedged identity

$$
V_t+\frac12\sigma^2S^2V_{SS}
=r(V-SV_S)+dSV_S.
$$

Theta and gamma are linked; they are not independent sources of P&L in the ideal model.

## 4.5 Terminal and boundary conditions

The PDE runs backward from terminal payoff. For a call,

$$
V(T,S)=(S-K)^+.
$$

Natural asymptotic boundaries are

$$
V(t,0)=0,
$$

and for large $S$,

$$
V(t,S)\approx Se^{-d(T-t)}-Ke^{-r(T-t)}.
$$

Numerical domains truncate $S$ at $S_{\max}$, so this asymptotic relation becomes an approximate boundary. A bad boundary can contaminate interior values.

## 4.6 Feynman–Kac representation

Consider the risk-neutral process

$$
dS_t=(r-d)S_tdt+\sigma S_tdW_t^Q.
$$

Feynman–Kac states that the PDE solution is

$$
V(t,S)
=E^Q\left[e^{-r(T-t)}\Phi(S_T)\mid S_t=S\right].
$$

The PDE is local and deterministic in state space; the expectation is global and probabilistic over paths. They agree because both encode the same generator and terminal condition.

## 4.7 Generator viewpoint

For a diffusion $dX=a(X,t)dt+b(X,t)dW$, its generator acting on smooth $f$ is

$$
\mathcal Lf=a f_x+\frac12b^2f_{xx}.
$$

Risk-neutral pricing solves

$$
V_t+\mathcal L^QV-rV=0.
$$

This form generalizes to multiple assets, stochastic rates, local volatility, and jump generators. Dimensionality determines whether grids remain practical.

## 4.8 Replication error in discrete time

Continuous delta hedging is an ideal limit. With discrete rebalancing, a local P&L approximation is

$$
\Delta V-\Delta\,\Delta S
\approx V_t\Delta t+\frac12V_{SS}(\Delta S)^2.
$$

Realized quadratic variation need not match model-implied $\sigma^2S^2\Delta t$. Transaction costs, jumps, liquidity, parameter error, and finite rebalancing produce residual hedging P&L.

Gamma-rich positions are especially sensitive to realized-versus-implied variance and rebalancing frequency.

## 4.9 Choosing PDE or Monte Carlo

PDE grids are attractive for one to three state variables, early-exercise boundaries, and full surfaces of values and Greeks. Monte Carlo handles high-dimensional factors and complex path dependence more naturally, but ordinary forward simulation struggles with optimal stopping and very accurate Greeks.

Trees, transform methods, regression Monte Carlo, and hybrid solvers occupy the space between.

## 4.10 Beyond Black–Scholes

If volatility depends on $S$ and $t$, replace $\sigma$ with local volatility $\sigma_{loc}(S,t)$ in the PDE. If volatility is a separate stochastic factor, the PDE gains another dimension and an unhedgeable volatility risk unless another traded asset completes the market.

Jumps prevent perfect hedging with stock and cash alone. The pricing equation becomes an integro-differential equation and requires a jump-risk pricing specification.

## 4.11 Failure modes

- Believing the PDE proves the assumptions rather than follows from them.
- Dropping the Itô second-derivative term.
- Leaving physical drift $\mu$ in the pricing PDE.
- Forgetting dividend yield or other carry.
- Applying terminal conditions at the wrong time direction.
- Using continuous-hedging claims to describe real discrete P&L without residual risk.
- Choosing PDE grids for a high-dimensional state without confronting dimensionality.

## 4.12 Knowledge checks

1. Derive the PDE by choosing the delta that cancels $dW$.
2. Why does physical drift disappear?
3. What are the terminal and large-spot conditions for a European call?
4. State the Feynman–Kac representation in words.
5. Why does discrete hedging leave gamma-related error?

### Solution outlines

1. Apply Itô to $V$, form $V-V_SS$, include dividends, and equate locally riskless return to $r$.
2. Stock risk is eliminated locally by replication, so its risk premium is irrelevant to the claim's no-arbitrage price.
3. $(S-K)^+$ at maturity and discounted forward intrinsic behaviour for large $S$.
4. The solution of a backward pricing PDE equals a discounted conditional expectation under the diffusion generated by its operator.
5. Finite intervals expose the hedge to realized quadratic variation and jumps between rebalances.

## 4.13 What to retain

- Delta hedging and no-arbitrage produce the pricing PDE.
- Gamma enters through Itô's quadratic-variation term.
- Feynman–Kac equates the PDE with risk-neutral expectation.
- Boundaries and terminal conditions are part of the pricing problem.
- Real hedging error measures departures from continuous-model assumptions.

Next: [Chapter 5 — Finite-Difference Methods](ch5-finite-differences-viewer.html).
