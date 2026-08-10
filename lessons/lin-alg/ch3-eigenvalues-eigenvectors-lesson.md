# Chapter 3 — Eigenvalues & Eigenvectors: The Hidden Directions

> Most matrices are diagonal matrices in disguise. This chapter shows you how to find the disguise — and why it matters for powers, PCA, PageRank, and physics.

## 3.1 The definition, said in words first

A generic machine $A$ stretches arrows *and* spins them. But there are special directions — **eigenvectors** — that the machine only stretches, never spins. The **eigenvalue** $\lambda$ is the stretch factor:

$$
A\mathbf v = \lambda \mathbf v, \qquad \mathbf v \neq 0
$$

- $\lambda = 2$: the arrow doubles.
- $\lambda = -1$: the arrow flips exactly backwards.
- $\lambda = 0.5$: the arrow shrinks to half.
- $\lambda = 0$: the arrow is destroyed (it lies in the nullspace).

> Every vector is stretched and rotated by $A$ — *except* the eigenvectors, which survive direction-unchanged. The eigenvalue says by how much.

## 3.2 Worked example: find the hidden directions

$$
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
$$

**Step 1 — write the condition for a non-zero solution.** $A\mathbf v = \lambda\mathbf v$ means $(A - \lambda I)\mathbf v = 0$. A non-zero $\mathbf v$ exists only if the machine $A - \lambda I$ collapses something:

$$
\det(A - \lambda I) = 0
$$

**Step 2 — the characteristic polynomial:**

$$
\det\begin{pmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{pmatrix}
= (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1)
$$

Eigenvalues: $\lambda_1 = 3$, $\lambda_2 = 1$.

**Step 3 — eigenvectors.** For $\lambda = 3$, solve $(A - 3I)\mathbf v = 0$:

$$
\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0
\implies -v_1 + v_2 = 0 \implies \mathbf v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

Check: $A(1,1) = (3,3) = 3(1,1)$ ✓. For $\lambda = 1$: $(A - I)\mathbf v = 0$ gives $v_1 + v_2 = 0$, so $\mathbf v_2 = (1, -1)$. Check: $A(1,-1) = (1,-1)$ ✓.

**Read the result:** along the direction $(1,1)$ the machine stretches ×3; along $(1,-1)$ it leaves lengths alone. In these two hidden directions, $A$ acts like the diagonal matrix $\operatorname{diag}(3, 1)$.

## 3.3 Sanity checks that never lie

| Check | For our example |
|---|---|
| Sum of eigenvalues = trace | $3 + 1 = 4 = 2 + 2$ ✓ |
| Product of eigenvalues = determinant | $3 \times 1 = 3 = 4 - 1$ ✓ |
| Distinct eigenvalues ⇒ independent eigenvectors | $(1,1), (1,-1)$ are independent ✓ |

These two checks catch 90% of hand-computation errors instantly. Always run them.

## 3.4 Diagonalization: making the disguise explicit

If $A$ has $n$ independent eigenvectors, collect them as columns of $V$ and the eigenvalues on the diagonal of $\Lambda$:

$$
A = V \Lambda V^{-1}
$$

**Why this is the best way to understand $A$:**

1. Change to the eigenvector basis ($V^{-1}$).
2. Stretch each coordinate independently by its $\lambda$ ($\Lambda$).
3. Change back ($V$).

**The payoff — powers of matrices:**

$$
A^k = V \Lambda^k V^{-1}, \qquad A^k \mathbf v = \sum_i \lambda_i^k c_i \mathbf v_i
$$

To raise $A$ to the 100th power, raise the eigenvalues to the 100th power. For a Markov chain's transition matrix $P$: $P^n$ is $n$-step probabilities, and since $|\lambda_i| < 1$ for $i \neq 1$, the chain converges to the dominant eigenvector — the stationary distribution. *PageRank is exactly this computation on the web's transition matrix.*

## 3.5 Symmetric matrices: the special, beautiful case

If $A = A^T$:

1. **All eigenvalues are real** (no complex numbers — this is why covariances behave).
2. **Eigenvectors of distinct eigenvalues are orthogonal.**
3. **$A$ diagonalizes with an orthogonal $Q$** ($Q^T Q = I$):

$$
A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i \, \mathbf q_i \mathbf q_i^T
$$

The spectral theorem turns $A$ into a sum of rank-1 machines, each stretching one orthogonal direction. This is the mathematical engine of:

- **PCA:** the covariance matrix $\Sigma = \frac{1}{N-1}X^T X$ is symmetric positive semi-definite; its eigenvectors are the principal directions of maximum variance, and $\lambda_i$ is the variance along direction $i$.
- **The Hessian:** at a minimum of a smooth function it is symmetric positive definite — all $\lambda_i > 0$.
- **Quadratic forms:** $x^T A x$ defines an ellipsoid whose axis lengths are $1/\sqrt{\lambda_i}$.

## 3.6 Worked example: a 3×3 (all real, clean numbers)

$$
A = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}
$$

**Characteristic polynomial** (expand along row 1):

$$
\det(A - \lambda I) = (2-\lambda)\big((2-\lambda)^2 - 1\big) - 1(2-\lambda) = (2-\lambda)\big(\lambda^2 - 4\lambda + 2\big)
$$

Roots: $\lambda = 2$ and $\lambda = 2 \pm \sqrt{2}$.

**Sanity:** sum $= 2 + (2 + \sqrt2) + (2 - \sqrt2) = 6$ = trace ✓. Product $= 2(4 - 2) = 4$ = determinant (computed in Chapter 2) ✓.

**Eigenvectors:**

- $\lambda = 2$: solve $(A - 2I)\mathbf v = 0$: $v_1 + 0 \cdot v_2 = 0$? From row 1: $0 \cdot v_1 + v_2 + 0 = 0 \Rightarrow v_2 = 0$; row 3: $v_2 + 0 \cdot v_3 = 0 \Rightarrow v_2 = 0$; so $\mathbf v = (1, 0, -1)$. Check: $A(1,0,-1) = (2, 0, -2) = 2(1,0,-1)$ ✓.
- $\lambda = 2 + \sqrt2$: $(- \sqrt2) v_1 + v_2 = 0 \Rightarrow v_2 = \sqrt2 \, v_1$; $v_2 - \sqrt2 \, v_3 = 0 \Rightarrow v_3 = v_1$. So $\mathbf v = (1, \sqrt2, 1)$. Check: $A(1,\sqrt2,1) = (2+\sqrt2, 2\sqrt2 + 2, 2+\sqrt2) = (2+\sqrt2)(1, \sqrt2, 1)$ ✓.
- $\lambda = 2 - \sqrt2$: analogously $\mathbf v = (1, -\sqrt2, 1)$.

All three eigenvectors are mutually orthogonal (the matrix is symmetric) — verify: $(1,0,-1)\cdot(1,\sqrt2,1) = 0$ ✓, $(1,0,-1)\cdot(1,-\sqrt2,1) = 0$ ✓, $(1,\sqrt2,1)\cdot(1,-\sqrt2,1) = 1 - 2 + 1 = 0$ ✓.

## 3.7 When diagonalization fails

Not every matrix has $n$ independent eigenvectors. A **defective** matrix like $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has eigenvalue 1 twice but only one eigenline — it shears along that line. No diagonalization exists; you need Jordan form (rarely needed in practice). For all the cases that matter in data work — symmetric matrices — the spectral theorem always saves you.

## 3.8 Common gotchas

| Gotcha | The truth |
|---|---|
| "Eigenvalues of $A$ are the same as singular values of $A$" | Only for symmetric positive semi-definite matrices. In general $\sigma_i = \sqrt{\lambda_i(A^T A)}$. |
| "$\det(A) > 0$ means positive definite" | No — positive definite means *all* $\lambda_i > 0$. |
| "Eigenvectors are unique" | Only up to scaling (and sign). $(1,1)$ and $(-1,-1)$ are the same eigenvector. |
| "Complex eigenvalues mean the matrix is bad" | Not at all — rotations have complex eigenvalues. It means there is no real invariant direction. |

## 3.9 Practice

1. Find the eigenvalues and eigenvectors of $\begin{pmatrix} 4 & 1 \\ 1 & 4 \end{pmatrix}$. *(Answer: $\lambda = 5$ with $(1,1)$, $\lambda = 3$ with $(1,-1)$; trace $8 = 5+3$ ✓, det $15 = 5\cdot3$ ✓.)*
2. If $A$ has eigenvalues $2, -1$ and $A = V\Lambda V^{-1}$, what are the eigenvalues of $A^{10}$? *(Answer: $2^{10}, (-1)^{10} = 1024, 1$.)*
3. The covariance matrix $\Sigma$ has eigenvalues $5, 1, 0.2$. What fraction of variance does the first principal component explain? *(Answer: $5 / (5+1+0.2) \approx 80.6\%$.)*
4. A Markov transition matrix has eigenvalue 1 and a second eigenvalue $0.8$. After $n$ steps, how fast does the distribution converge to stationarity? *(Answer: at rate $0.8^n$ — the second eigenvalue controls the speed.)*
