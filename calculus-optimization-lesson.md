# Calculus & Optimization for Quantitative Finance

> Derivatives measure rates of change; gradients point uphill; Lagrange multipliers find constrained bests. This is the calculus toolkit used to price, hedge, and optimise — with every move worked by hand.

## 1. Derivatives: the rate of change

The derivative of $f$ at $x$ is the instantaneous rate of change:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

It answers: *if $x$ moves a tiny amount, how much does $f$ move?* That is the core question of risk: how much does the portfolio value move when the underlying moves?

| Function | Derivative |
|---|---|
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $u(x)v(x)$ | $u'v + uv'$ (product rule) |
| $u(v(x))$ | $u'(v(x))\,v'(x)$ (chain rule) |
| $u(x)/v(x)$ | $(u'v - uv')/v^2$ (quotient rule) |

**The chain rule is the most used rule in quantitative finance.** It says: *the rate of change through a chain of effects is the product of the rates at each link.* A derivative's delta, a bond's duration, and a model's sensitivity all are chain-rule products.

**Worked example.** $f(x) = e^{x^2}$:

$$
f'(x) = e^{x^2} \cdot 2x = 2x e^{x^2}
$$

**Worked example (the log trick).** $f(x) = x^x$. Take logs, differentiate, solve:

$$
\ln f = x\ln x \implies \frac{f'}{f} = \ln x + 1 \implies f' = x^x(\ln x + 1)
$$

## 2. Partial derivatives and the gradient

With multiple variables, take the derivative with respect to one at a time, holding the others fixed:

$$
\frac{\partial f}{\partial x} = \text{rate of change in the } x\text{-direction}
$$

The **gradient** collects all partial derivatives into a vector:

$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)
$$

Two facts to carry forever:

1. **The gradient points in the direction of steepest ascent** — of all directions to step, this one increases $f$ fastest.
2. **$-\nabla f$ points in the direction of steepest descent** — which is exactly why gradient descent (the engine of machine learning) steps opposite the gradient.

**Worked example.** $f(x, y) = x^2 + 3xy + y^2$:

$$
\nabla f = (2x + 3y,\; 3x + 2y)
$$

At $(1, 1)$: $\nabla f = (5, 5)$ — increasing $f$ fastest in the $(1,1)$ direction, decreasing it fastest in the $(-1,-1)$ direction.

## 3. Taylor expansion: linear and quadratic approximations

Near a point $a$, any smooth function can be approximated by polynomials:

$$
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x-a)^2 + \frac{f'''(a)}{6}(x-a)^3 + \cdots
$$

**Why this is everywhere in finance:**

- **Delta hedging** is the linear term: $f(S + \Delta S) \approx f(S) + f'(S)\Delta S$. The first-order term is exactly how much the derivative moves with the underlying.
- **Gamma** is the quadratic term: when the second derivative is large, the linear approximation understates risk — this is why convexity matters in both bonds and options.
- The multi-variable version uses the Hessian $H$ (matrix of second partials):

$$
f(x) \approx f(a) + \nabla f(a)^T (x-a) + \frac{1}{2}(x-a)^T H(a)\,(x-a)
$$

**Worked example.** Approximate $\sqrt{4.01}$ using the Taylor series of $f(x) = \sqrt{x}$ at $a = 4$. We have $f(4) = 2$, $f'(4) = \frac{1}{4}$, $f''(4) = -\frac{1}{32}$:

$$
\sqrt{4.01} \approx 2 + \frac{1}{4}(0.01) - \frac{1}{64}(0.01)^2 \approx 2 + 0.0025 - 0.0000016 = 2.0024984
$$

Actual: $\sqrt{4.01} = 2.0024984\ldots$ — the quadratic approximation is accurate to seven digits.

## 4. Convexity: the shape that guarantees a best answer

A function is **convex** if the chord lies above the graph:

$$
f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda) f(y), \qquad \lambda \in [0, 1]
$$

Equivalently (for twice-differentiable functions): $f''(x) \ge 0$ everywhere (or the Hessian is positive semi-definite in multiple dimensions).

**Why convexity is the golden property:** a convex function has *exactly one* (possibly flat) minimum, and any local minimum is global. Gradient descent on a convex function cannot get stuck. Many finance problems are engineered to be convex on purpose (least-squares objectives, Markowitz with a fixed covariance) precisely so the maths is tractable.

**Jensen's inequality** is the analytical consequence: for convex $f$,

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
$$

It reverses for concave $f$. The pricing implications are everywhere: $\mathbb{E}[e^X] \ge e^{\mathbb{E}[X]}$ (log is concave), so the expected price of a log-normal asset is *not* the price built from the mean log-return — the volatility correction is Jensen's gap.

## 5. Unconstrained optimization

To minimise (or maximise) a smooth function: **set the gradient to zero, then check the second derivative.**

1. **First-order condition:** $\nabla f(x^*) = 0$ — no direction improves locally.
2. **Second-order condition:** the Hessian at $x^*$ positive definite → local minimum; negative definite → local maximum; indefinite → saddle point.

**Worked example.** Minimise $f(x) = x^2 - 4x + 7$.

- $f'(x) = 2x - 4 = 0 \Rightarrow x = 2$.
- $f''(x) = 2 > 0$ → minimum.
- $f(2) = 4 - 8 + 7 = 3$.

**Worked example (two variables).** Minimise $f(x, y) = x^2 + y^2 - 2x - 4y + 10$.

- $\nabla f = (2x - 2,\; 2y - 4) = (0, 0) \Rightarrow x = 1, y = 2$.
- Hessian $= \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$, positive definite → minimum.
- $f(1, 2) = 1 + 4 - 2 - 8 + 10 = 5$.

## 6. Constrained optimization: Lagrange multipliers

When the variables must satisfy a constraint $g(x) = c$, build the **Lagrangian**:

$$
\mathcal{L}(x, \lambda) = f(x) - \lambda\big(g(x) - c\big)
$$

and solve the system:

$$
\nabla_x \mathcal{L} = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0
$$

The condition $\nabla f = \lambda \nabla g$ has a clean reading: **at the optimum, the objective's gradient is parallel to the constraint's gradient** — there is no feasible direction left that improves $f$. The multiplier $\lambda$ itself is the *shadow price*: how much the optimum improves if the constraint is relaxed by one unit.

**Worked example.** Minimise $x^2 + y^2$ subject to $x + y = 4$.

- $\mathcal{L} = x^2 + y^2 - \lambda(x + y - 4)$.
- $\partial\mathcal{L}/\partial x = 2x - \lambda = 0 \Rightarrow x = \lambda/2$; $\partial\mathcal{L}/\partial y = 2y - \lambda = 0 \Rightarrow y = \lambda/2$.
- Constraint: $x + y = \lambda = 4 \Rightarrow \lambda = 4$, so $x = y = 2$, minimum value $8$.

**Worked example (utility).** Maximise $U(x, y) = x^{1/2} y^{1/2}$ subject to the budget $2x + y = 6$.

- $\partial U/\partial x = \frac{1}{2}x^{-1/2}y^{1/2} = \lambda \cdot 2$; $\partial U/\partial y = \frac{1}{2}x^{1/2}y^{-1/2} = \lambda \cdot 1$.
- Ratio: $y/x = 2 \Rightarrow y = 2x$. Budget: $2x + 2x = 6 \Rightarrow x = 1.5$, $y = 3$.
- $U_{\max} = \sqrt{4.5} \approx 2.121$.

This is the same structure as portfolio problems: maximise a utility/return objective subject to a budget or risk constraint — the tangency condition $\mathrm{MU}_x / P_x = \mathrm{MU}_y / P_y$ is Lagrange in disguise.

## 7. Newton's method: root-finding as repeated linearisation

To solve $f(x) = 0$, iterate:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

Each step replaces the curve by its tangent line and jumps to where the tangent crosses zero.

**Worked example.** Solve $x^2 - 2 = 0$ (i.e. compute $\sqrt2$) starting from $x_0 = 1$:

$$
x_1 = 1 - \frac{-1}{2} = 1.5, \qquad
x_2 = 1.5 - \frac{0.25}{3} = 1.4167, \qquad
x_3 = 1.4167 - \frac{0.00694}{2.8333} \approx 1.41422
$$

$\sqrt2 = 1.41421\ldots$ — three steps to six digits. Newton's method doubles correct digits per step (quadratic convergence). In finance it is the engine behind **implied volatility** inversion: find the volatility $\sigma$ such that the model price equals the market price.

## 8. The toolkit in one page

| Tool | Question it answers | Finance use |
|---|---|---|
| Derivative $f'$ | How much does $f$ change per unit of $x$? | Delta, duration, sensitivities |
| Gradient $\nabla f$ | Which direction changes $f$ fastest? | Gradient descent, optimisation |
| Taylor / Hessian | What is the local shape of $f$? | Gamma, convexity, risk approximation |
| Convexity | Is there a unique best answer? | Convex objectives, Jensen's inequality |
| Lagrange multipliers | Best value under a constraint? | Portfolio allocation, utility maximisation |
| Newton's method | Solve $f(x) = 0$ quickly? | Implied volatility, root finding |

## 9. Practice

1. Differentiate $f(x) = \ln(x^2 + 1)$. *(Answer: $\frac{2x}{x^2+1}$ — chain rule.)*
2. Find the gradient of $f(x, y) = e^{xy}$ at $(1, 2)$. *(Answer: $(ye^{xy}, xe^{xy}) = (2e^2, e^2)$.)*
3. Minimise $f(x) = x^2 + \frac{1}{x}$ for $x > 0$. *(Answer: $f' = 2x - 1/x^2 = 0 \Rightarrow x = 2^{-1/3} \approx 0.794$; $f'' = 2 + 2/x^3 > 0$, minimum.)*
4. Use the linear Taylor term to approximate $e^{0.02}$. *(Answer: $1 + 0.02 = 1.02$; exact $1.0202$ — the quadratic term contributes the extra $0.0002$.)*
5. Minimise $3x + 4y$ subject to $x^2 + y^2 = 25$. *(Answer: Lagrange gives $x = 3, y = 4$ (and the negative version), minimum value $25$; check $\nabla f = (3,4) = \lambda(2x, 2y)$ with $\lambda = 1/2$ at $(3,4)$.)*
