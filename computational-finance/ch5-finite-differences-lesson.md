# Chapter 5 — Finite-Difference Methods: Grids, Stability & Verification

A pricing PDE turns an unknown function into local relationships among neighbouring states and times. Finite differences replace derivatives by those relationships on a grid, converting the PDE into repeated linear-algebra problems.

The formulas are short. Reliable use depends on boundaries, stability, payoff smoothness, convergence, and independent verification.

## 5.1 The grid

For the Black–Scholes PDE, choose spot nodes

$$
S_i=i\Delta S,\qquad i=0,\ldots,M,
$$

and time nodes $t_n=n\Delta t$. Write $V_i^n\approx V(t_n,S_i)$. Terminal values are known:

$$
V_i^N=\Phi(S_i).
$$

The computation proceeds backward from $n=N$ to $0$.

Central spatial differences are

$$
V_S\approx\frac{V_{i+1}^n-V_{i-1}^n}{2\Delta S},
$$

$$
V_{SS}\approx\frac{V_{i+1}^n-2V_i^n+V_{i-1}^n}{(\Delta S)^2}.
$$

Taylor expansion shows both have second-order spatial truncation error for smooth $V$.

## 5.2 Time direction

It is often clearer to use time to maturity $\tau=T-t$. Then

$$
V_\tau=\frac12\sigma^2S^2V_{SS}+(r-d)SV_S-rV,
$$

with initial condition at $\tau=0$ equal to the payoff. The equation now advances forward in $\tau$ even though calendar-time pricing runs backward.

Mixing these directions is a common sign error.

## 5.3 Explicit scheme

Approximate $V_\tau$ with a forward difference and use spatial values at the current layer:

$$
V_i^{n+1}=a_iV_{i-1}^n+b_iV_i^n+c_iV_{i+1}^n.
$$

The coefficients follow directly from substituting finite differences into the PDE. Explicit stepping is simple and parallel across nodes, but stable only when $\Delta t$ is sufficiently small relative to spatial resolution and diffusion strength.

For a heat equation, the familiar restriction is roughly

$$
\frac{\sigma^2S^2\Delta t}{(\Delta S)^2}\lesssim1.
$$

Because the coefficient grows with $S^2$, the most restrictive node can determine the global time step. Violating stability produces oscillations or explosive values even when the code is algebraically correct.

## 5.4 Implicit scheme

Backward Euler evaluates spatial derivatives at the new time layer:

$$
A V^{n+1}=V^n,
$$

where $A$ is tridiagonal for a one-dimensional PDE. Each step requires solving a tridiagonal system, which is $O(M)$ with the Thomas algorithm.

Backward Euler is first-order in time and strongly damps high-frequency errors. It is unconditionally stable for standard diffusion problems, meaning stability does not impose an explicit-style time-step restriction. Accuracy still requires refinement.

“Unconditionally stable” does not mean “accurate for any grid.”

## 5.5 Crank–Nicolson

Crank–Nicolson averages the spatial operator between two layers:

$$
\left(I-\frac{\Delta t}{2}L\right)V^{n+1}
=\left(I+\frac{\Delta t}{2}L\right)V^n.
$$

It is second-order in time for smooth solutions and usually the default for vanilla problems. Nonsmooth terminal payoffs can generate oscillations near the strike. **Rannacher smoothing** uses a few half-sized backward-Euler steps before switching to Crank–Nicolson.

This illustrates a general lesson: formal order assumes smoothness that option payoffs may not initially have.

## 5.6 Boundary conditions and truncation

The continuous spot domain is $[0,\infty)$; the grid ends at $S_{\max}$. Boundaries may use known values, asymptotic formulas, or derivative conditions.

For a European call,

$$
V(\tau,0)=0,
$$

$$
V(\tau,S_{\max})\approx S_{\max}e^{-d\tau}-Ke^{-r\tau}.
$$

$S_{\max}$ must be far enough that this approximation does not distort the region of interest. Grid refinement with a fixed poor boundary can converge to the wrong truncated problem.

## 5.7 Consistency, stability, and convergence

- **Consistency:** local finite-difference equations approach the PDE as steps shrink.
- **Stability:** numerical errors do not grow without control.
- **Convergence:** the grid solution approaches the PDE solution.

For well-posed linear problems, the Lax equivalence principle connects consistent and stable schemes to convergence. In practice, perform a refinement table:

| Grid | Price | Change | Estimated order |
|---|---:|---:|---:|
| $(M,N)$ | $V_1$ | — | — |
| $(2M,2N)$ | $V_2$ | $|V_2-V_1|$ | — |
| $(4M,4N)$ | $V_3$ | $|V_3-V_2|$ | $\log_2\frac{|V_2-V_1|}{|V_3-V_2|}$ |

The observed order should match theory only after reaching the asymptotic regime.

## 5.8 Nonuniform grids and coordinate changes

Concentrate nodes near the strike, barriers, or expected exercise boundary where curvature is largest. Nonuniform finite-difference weights must reflect local spacing.

Log-price $x=\log S$ converts multiplicative diffusion into a constant diffusion coefficient and can simplify stability analysis. Further transformations map Black–Scholes to a heat equation, useful for theory and benchmarks.

Coordinate changes alter boundary placement and interpolation; they do not remove the need for validation.

## 5.9 Greeks from a grid

At a node,

$$
\Delta_i\approx\frac{V_{i+1}-V_{i-1}}{S_{i+1}-S_{i-1}},
$$

and gamma uses a second difference. Greeks amplify grid noise and interpolation error. A price can appear converged while gamma near a payoff kink remains unstable.

Check Greek convergence separately and align important states with nodes when possible.

## 5.10 Verification ladder

1. Test derivative stencils on functions with known derivatives.
2. Compare European prices and Greeks with Black–Scholes.
3. Check put–call parity and monotonicity.
4. Confirm nonnegative values and sensible bounds.
5. Refine $S_{\max}$, $\Delta S$, and $\Delta t$ separately.
6. Compare explicit, implicit, and Crank–Nicolson in a stable regime.
7. Inspect residuals of the discrete PDE.

Agreement between two implementations sharing the same mistake is not independent verification.

## 5.11 Failure modes

- Marching in the wrong time direction.
- Using an explicit grid outside its stability region.
- Interpreting unconditional stability as unrestricted accuracy.
- Ignoring payoff nonsmoothness and Crank–Nicolson oscillations.
- Refining the interior while leaving a contaminating boundary fixed.
- Interpolating prices smoothly but reporting noisy Greeks.
- Validating only at one parameter point.

## 5.12 Knowledge checks

1. Derive the central second-derivative stencil by Taylor expansion.
2. Why is backward Euler more damping than Crank–Nicolson?
3. What problem does Rannacher smoothing address?
4. Distinguish stability from accuracy.
5. Why should $S_{\max}$ be included in convergence studies?

### Solution outlines

1. Add expansions at $S+\Delta S$ and $S-\Delta S$; odd derivatives cancel, leaving $V_{SS}(\Delta S)^2$.
2. It fully evaluates diffusion at the new layer and suppresses high-frequency modes more strongly.
3. Oscillations caused by applying a second-order time scheme immediately to a nonsmooth payoff.
4. Stability controls error growth; accuracy controls closeness to the desired solution.
5. The truncated boundary is itself an approximation that can dominate interior error.

## 5.13 What to retain

- Finite differences turn a pricing PDE into sparse linear algebra.
- Explicit, implicit, and Crank–Nicolson trade simplicity, damping, and time accuracy.
- Stability is necessary but not sufficient for accuracy.
- Boundaries and payoff smoothness determine practical behaviour.
- Prices and Greeks need separate refinement evidence.

Next: [Chapter 6 — American & Path-Dependent Options](ch6-american-path-dependent-viewer.html).
