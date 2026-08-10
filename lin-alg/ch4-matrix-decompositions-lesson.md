# Chapter 4 — Decompositions: Factor Everything

> Every matrix is a product of simpler matrices. Choose the right factorization and the hard problem — solving, inverting, compressing, ranking — becomes easy. This chapter covers the five decompositions that matter, each with a worked example and a "when to use it" rule.

## 4.1 Why factor at all?

A matrix $A$ is a machine, but a *generic* machine is opaque. Factorization rewrites it as a pipeline of simple machines:

$$
A = \underbrace{L}_{ \text{triangles} } U, \qquad
A = Q R, \qquad
A = Q \Lambda Q^T, \qquad
A = U \Sigma V^T
$$

Once factored, the hard operations become trivial: solving $Ax = b$ becomes two triangular solves; $A^{100}$ becomes $\lambda_i^{100}$; a rank-$k$ approximation is a truncation. **A decomposition is not a trick — it is the matrix's anatomy laid out for surgery.**

## 4.2 LU decomposition (for solving)

**What:** $A = LU$ with $L$ lower-triangular (ones on the diagonal) and $U$ upper-triangular.

**Where it comes from:** Gaussian elimination records itself. The elimination matrix $E$ from Chapter 2 multiplies to give $U = EA$; its inverse is $L$.

**Worked example.** $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$. Elimination: row 2 ← row 2 − 3·row 1 gives $U = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$; the multiplier 3 goes into $L$:

$$
L = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}, \qquad
U = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}, \qquad
LU = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix} = A \quad\checkmark
$$

**Why you want it:** to solve $Ax = b$ for many different $b$'s (same $A$), factor once, then:

1. Solve $L\mathbf y = \mathbf b$ by forward substitution.
2. Solve $U\mathbf x = \mathbf y$ by back-substitution.

Each solve is $O(n^2)$ instead of $O(n^3)$. In practice use **partial pivoting** ($PA = LU$, $P$ a permutation) so the pivots are never tiny — that is what `np.linalg.solve` / `scipy.linalg.lu` do under the hood.

## 4.3 Cholesky decomposition (for symmetric positive definite)

**What:** $A = LL^T$, $L$ lower-triangular with positive diagonal. Exists iff $A$ is symmetric positive definite. Half the work of LU and rock-solid numerically.

**Worked example.** $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$. Solve for $L = \begin{pmatrix} \ell_{11} & 0 \\ \ell_{21} & \ell_{22} \end{pmatrix}$:

- $\ell_{11}^2 = 4 \Rightarrow \ell_{11} = 2$
- $\ell_{21}\ell_{11} = 2 \Rightarrow \ell_{21} = 1$
- $\ell_{21}^2 + \ell_{22}^2 = 3 \Rightarrow \ell_{22}^2 = 2 \Rightarrow \ell_{22} = \sqrt2$

$$
L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt2 \end{pmatrix}, \qquad
LL^T = \begin{pmatrix} 4 & 2 \\ 2 & 1 + 2 \end{pmatrix} = A \quad\checkmark
$$

**Uses:** testing positive definiteness (if Cholesky fails, it is not PD); sampling from $\mathcal{N}(\mu, \Sigma)$ via $\mu + L\mathbf z$ with $\mathbf z \sim \mathcal{N}(0, I)$; computing $\log \det \Sigma = 2\sum_i \log L_{ii}$; solving normal equations without forming $X^T X$ badly.

## 4.4 QR decomposition (for stable least squares and orthonormal bases)

**What:** $A = QR$, $Q$ orthogonal ($Q^T Q = I$), $R$ upper-triangular. The columns of $Q$ are an orthonormal basis of $A$'s column space — the answer to "which perpendicular axes span what this machine reaches?"

**Worked example** (Gram–Schmidt on columns of $A = \begin{pmatrix} 4 & 1 \\ 3 & 2 \end{pmatrix}$):

- $q_1 = \frac{(4,3)}{\|(4,3)\|} = (0.8, 0.6)$, $r_{11} = 5$.
- Project out: $v_2 = (1,2) - ((0.8,0.6)\cdot(1,2))(0.8,0.6) = (1,2) - 2(0.8,0.6) = (-0.6, 0.8)$.
- $q_2 = (-0.6, 0.8)$, $r_{12} = 2$, $r_{22} = \|v_2\| = 1$.

$$
Q = \begin{pmatrix} 0.8 & -0.6 \\ 0.6 & 0.8 \end{pmatrix}, \qquad
R = \begin{pmatrix} 5 & 2 \\ 0 & 1 \end{pmatrix}, \qquad
QR = \begin{pmatrix} 4 & 1 \\ 3 & 2 \end{pmatrix} = A \quad\checkmark
$$

**Why you want it:** least squares without forming $A^T A$. Since $Q$ is orthogonal, $\|Ax - b\| = \|Rx - Q^Tb\|$, so solve $R\mathbf x = Q^T\mathbf b$ by back-substitution — numerically stable where the normal equations ($A^T A$, condition number squared) are not. (In production, Householder reflections do the factoring; Gram–Schmidt here is just for intuition.)

## 4.5 Eigendecomposition (for powers, PCA, Markov chains)

**What:** $A = V\Lambda V^{-1}$ — or, for symmetric $A$, $A = Q\Lambda Q^T$ with orthogonal $Q$ (Chapter 3).

**Worked example.** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ has $\lambda = 3, 1$ with orthonormal eigenvectors $q_1 = \frac{1}{\sqrt2}(1,1)$, $q_2 = \frac{1}{\sqrt2}(1,-1)$:

$$
A = \frac{1}{\sqrt2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}
    \frac{1}{\sqrt2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

Verify one term of the spectral sum: $3\, q_1q_1^T + 1\, q_2q_2^T = \frac{3}{2}\begin{pmatrix}1&1\\1&1\end{pmatrix} + \frac{1}{2}\begin{pmatrix}1&-1\\-1&1\end{pmatrix} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ ✓

**Uses:** matrix powers and exponentials; the stationary distribution of Markov chains; PCA; spectral clustering; solving linear ODEs. For *square, symmetric* matrices this is the decomposition of choice.

## 4.6 SVD — the decomposition that always exists (the crown jewel)

**What:** *every* matrix — square or rectangular, singular or not — factors as:

$$
A = U \Sigma V^T
$$

$U$ and $V$ orthogonal; $\Sigma$ diagonal with $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$.

**The one-sentence intuition (rotate–scale–rotate):**

> Every matrix does exactly three things: rotate (or reflect) the input space via $V^T$, stretch the axes by the singular values via $\Sigma$, rotate (or reflect) the output space via $U$.

**Worked example.** $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ (symmetric). Its eigendecomposition is also its SVD: eigenvalues $4, 2$ with eigenvectors $\frac{1}{\sqrt2}(1,1), \frac{1}{\sqrt2}(1,-1)$:

$$
A = \frac{1}{\sqrt2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}
    \frac{1}{\sqrt2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    = U\Sigma V^T
$$

Check: $\sigma_1^2 = 16 = \lambda_1(A^TA)$ ✓, $\sigma_2^2 = 4 = \lambda_2(A^TA)$ ✓, $\det A = 8 = 4 \cdot 2 = \prod \sigma_i$ ✓.

**As a sum of rank-1 machines:**

$$
A = \sigma_1 \mathbf u_1\mathbf v_1^T + \sigma_2 \mathbf u_2\mathbf v_2^T + \cdots
$$

**The Eckart–Young theorem (why SVD is the compression king):** the best rank-$k$ approximation to $A$ is the sum of its first $k$ SVD terms. The error is $\sqrt{\sum_{i>k}\sigma_i^2}$. This one theorem powers PCA, image compression, recommender systems (matrix factorization), LSA, and LoRA fine-tuning in LLMs.

**Worked example (compression).** $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ with rank-1 approximation:

$$
A_1 = 4 \cdot \frac{(1,1)}{\sqrt2}\frac{(1,1)}{\sqrt2}^T = 2\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}
$$

Error: $\|A - A_1\|_F = \sqrt{2^2} = 2 = \sigma_2$ ✓ — one singular value kept, one dropped.

## 4.7 Choosing the right decomposition

| You need… | Use… | Why |
|---|---|---|
| Solve $Ax = b$ (square) | LU | Fast, stable with pivoting; one factor, many solves |
| Solve PD systems / sample Gaussians | Cholesky | Half the work of LU, ultra-stable, tests PD |
| Least squares, orthonormal basis | QR | Avoids ill-conditioned $A^TA$ |
| Powers, Markov chains, PCA (symmetric) | Eigendecomposition | $A^k = Q\Lambda^k Q^T$; the spectral theorem |
| Anything at all — rank, nullspace, low-rank approx, condition number | **SVD** | Always exists, always stable; Eckart–Young |
| Solve ill-conditioned problems | SVD + truncate | Kill the tiny singular values, not the solution |

**Rule of thumb:** when in doubt, SVD. It computes rank, range, nullspace, pseudoinverse ($A^+ = V\Sigma^+ U^T$), and the condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ — all from one factorization.

## 4.8 Common gotchas

| Gotcha | The truth |
|---|---|
| "$\det(A) \neq 0$ means $Ax = b$ is safe" | Determinant does not measure *how* invertible. Use $\kappa(A)$; if $\kappa > 10^8$, expect garbage. |
| "Eigenvalues and singular values are the same thing" | Only for symmetric PSD matrices. SVD exists for every matrix; eigendecomposition does not. |
| "Form $X^T X$ for least squares" | Squares the condition number. Use QR or SVD on $X$ directly. |
| "Rank = number of non-zero eigenvalues" | Compute rank as the number of singular values above a tolerance — eigenvalues are the wrong tool for non-symmetric matrices. |
| "QR by Gram–Schmidt in production" | Classic Gram–Schmidt is numerically unstable; use Householder. Gram–Schmidt is for intuition only. |

## 4.9 Practice

1. Write the LU of $\begin{pmatrix} 3 & 1 \\ 9 & 5 \end{pmatrix}$. *(Answer: $L = \begin{pmatrix}1&0\\3&1\end{pmatrix}$, $U = \begin{pmatrix}3&1\\0&2\end{pmatrix}$; check $LU = A$ ✓.)*
2. Cholesky of $\begin{pmatrix} 9 & 3 \\ 3 & 10 \end{pmatrix}$? *(Answer: $L = \begin{pmatrix}3&0\\1&3\end{pmatrix}$: $9=3^2$ ✓, $3=3\cdot1$ ✓, $10=1+9$ ✓.)*
3. The SVD of a matrix has singular values $5, 1, 0.1$. What is the condition number, and what fraction of the Frobenius energy is in the rank-1 approximation? *(Answer: $\kappa = 5/0.1 = 50$; energy fraction $= 25/(25+1+0.01) \approx 96.1\%$.)*
4. When would you prefer QR over SVD for least squares? *(Answer: when $X$ is tall and full column rank and speed matters — QR is cheaper; use SVD when $X$ may be rank-deficient.)*

---

## The four chapters in one paragraph

Matrices are machines (Ch 1). You must be fluent in their mechanics — solving, inverting, rank, determinants (Ch 2). Most machines are diagonal in hidden directions — the eigenvectors — which unlocks powers, PCA, and Markov chains (Ch 3). And every machine has a clean anatomy — LU, Cholesky, QR, eigendecomposition, SVD — so that any hard operation becomes a few simple steps (Ch 4). That is linear algebra. Everything else is detail.
