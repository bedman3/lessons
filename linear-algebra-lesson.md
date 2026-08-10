# Linear Algebra for Machine Learning & Quant Finance

> From determinants to SVD — the geometric intuition, the decompositions, and the memory-friendly way to understand why they all matter.

## 1. Why linear algebra matters

Almost everything in quantitative work is linear algebra in disguise:

- **Linear regression:** $\hat{\beta} = (X^T X)^{-1} X^T y$ — pure matrix algebra.
- **PCA:** eigendecomposition of the covariance matrix.
- **Neural networks:** every layer is $a = g(Wx + b)$ — a matrix multiply followed by a nonlinearity.
- **Portfolio optimisation:** $\min_w w^T \Sigma w$ subject to $w^T \mathbf{1} = 1$ — quadratic form with linear constraints.
- **PageRank:** the dominant eigenvector of the web graph's transition matrix.
- **Diffusion models and graph learning:** spectral decompositions of the graph Laplacian.

The goal of this lesson is not to memorise formulas but to build **geometric intuition** — to see a matrix and feel what it does to space.

---

## 2. Vectors and matrices: geometric thinking

### 2.1 Vectors are arrows (and points)

A vector $v \in \mathbb{R}^n$ can be thought of as:
- A **point** in $n$-dimensional space.
- An **arrow** from the origin to that point.

Two operations define a vector space:
- **Addition:** $v + w$ — put the tail of $w$ at the head of $v$ (the parallelogram rule).
- **Scalar multiplication:** $\alpha v$ — stretch or shrink $v$ by $\alpha$.

### 2.2 Linear combinations and span

A **linear combination** of vectors $v_1, \ldots, v_k$ is:

$$
\alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_k v_k.
$$

The **span** of $\{v_1, \ldots, v_k\}$ is the set of all possible linear combinations — the subspace these vectors "reach." The span of one vector is a line through the origin. The span of two independent vectors is a plane.

### 2.3 Linear independence

Vectors are **linearly independent** if no vector in the set can be written as a linear combination of the others. Equivalently:

$$
\alpha_1 v_1 + \cdots + \alpha_k v_k = 0 \implies \alpha_1 = \cdots = \alpha_k = 0.
$$

Geometric intuition: $k$ independent vectors span a $k$-dimensional subspace. If one vector lies in the span of the others, the set is **dependent** and the dimension drops.

### 2.4 Basis and dimension

A **basis** for a vector space is a set of linearly independent vectors that span the entire space. Every vector in the space can be written **uniquely** as a linear combination of the basis vectors. The number of vectors in any basis is the **dimension** of the space.

The standard basis for $\mathbb{R}^n$: $e_1 = (1,0,\ldots,0)$, $e_2 = (0,1,\ldots,0)$, etc. But any $n$ independent vectors form a basis — and different bases give different coordinate representations of the same underlying vector.

### 2.5 Dot product and orthogonality

The dot product (inner product) of two vectors:

$$
v \cdot w = v^T w = \sum_{i=1}^{n} v_i w_i = \|v\| \,\|w\| \cos\theta.
$$

where $\|v\| = \sqrt{v \cdot v}$ is the Euclidean length. The dot product measures **alignment**: it is maximised when $v$ and $w$ point in the same direction ($\cos\theta = 1$), zero when they are perpendicular ($\cos\theta = 0$), and negative when they point in opposite directions.

Two vectors are **orthogonal** if $v \cdot w = 0$. They are **orthonormal** if additionally $\|v\| = \|w\| = 1$. An orthogonal matrix $Q$ has $Q^T Q = I$ — its columns form an orthonormal basis. Multiplying by $Q$ preserves lengths and angles: it is a rotation (or reflection).

---

## 3. Matrices as linear transformations

### 3.1 The key mental model

**A matrix $A \in \mathbb{R}^{m \times n}$ is a function that takes a vector $x \in \mathbb{R}^n$ and produces a vector $Ax \in \mathbb{R}^m$.**

This is the most important sentence in linear algebra. A matrix is not a grid of numbers — it is an action. It stretches, rotates, reflects, projects, or collapses space.

### 3.2 Matrix multiplication as composition

If $A$ is $m \times n$ and $B$ is $n \times p$, then $C = AB$ is $m \times p$ with:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}.
$$

Interpretation: the $j$-th column of $C$ is $A$ applied to the $j$-th column of $B$. Matrix multiplication is **function composition**: applying $B$ then $A$ is $AB$. This is why $(AB)x = A(Bx)$.

**Four ways to think about $C = AB$:**

1. **Dot product view:** $C_{ij}$ = dot product of row $i$ of $A$ with column $j$ of $B$.
2. **Column view:** columns of $C$ = linear combinations of columns of $A$, with weights from $B$.
3. **Row view:** rows of $C$ = linear combinations of rows of $B$, with weights from $A$.
4. **Sum of rank-1 matrices:** $C = \sum_{k=1}^{n} (\text{column } k \text{ of } A)(\text{row } k \text{ of } B)$.

### 3.3 The four fundamental subspaces

For a matrix $A \in \mathbb{R}^{m \times n}$:

| Subspace | Notation | Dimension | Intuition |
|---|---|---|---|
| Column space (range) | $\mathcal{R}(A)$ | $\text{rank}(A) = r$ | Every possible output $Ax$. The subspace that $A$ "hits." |
| Row space | $\mathcal{R}(A^T)$ | $r$ | Every possible $A^T y$. The subspace that $A^T$ "hits." |
| Nullspace (kernel) | $\mathcal{N}(A)$ | $n - r$ | Vectors $x$ such that $Ax = 0$. The directions $A$ "kills." |
| Left nullspace | $\mathcal{N}(A^T)$ | $m - r$ | Vectors $y$ such that $A^T y = 0$. |

**Fundamental theorem of linear algebra:** The row space and nullspace are orthogonal complements in $\mathbb{R}^n$; the column space and left nullspace are orthogonal complements in $\mathbb{R}^m$. Every vector in $\mathbb{R}^n$ decomposes uniquely into a row-space component plus a nullspace component.

### 3.4 Rank

The **rank** of $A$ is the dimension of its column space — the number of linearly independent columns (which equals the number of linearly independent rows). A matrix is **full rank** if:
- $\text{rank}(A) = \min(m, n)$ — the maximum possible.
- Full column rank ($r = n$): columns are independent; $A^T A$ is invertible; nullspace is $\{0\}$.
- Full row rank ($r = m$): rows are independent; $A A^T$ is invertible.

If $r < \min(m,n)$, the matrix is **rank-deficient** — it collapses some dimensions.

---

## 4. The determinant

### 4.1 Geometric definition

**The determinant of a square matrix $A \in \mathbb{R}^{n \times n}$ is the signed volume of the $n$-dimensional parallelepiped formed by its columns.**

For a $2 \times 2$ matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the columns span a parallelogram. Its signed area is:

$$
\det(A) = ad - bc.
$$

For a $3 \times 3$ matrix, $\det(A)$ is the signed volume of the parallelepiped.

### 4.2 What the determinant tells you

| $\det(A)$ | Meaning |
|---|---|
| $\det(A) = 0$ | $A$ is singular — it collapses at least one dimension. Columns are linearly dependent. $A^{-1}$ does not exist. |
| $\det(A) > 0$ | $A$ preserves orientation. |
| $\det(A) < 0$ | $A$ reverses orientation (like a reflection). |
| $|\det(A)| > 1$ | $A$ expands volume. |
| $0 < |\det(A)| < 1$ | $A$ shrinks volume. |
| $\det(A) = 1$ | $A$ is volume-preserving (all rotations have determinant 1). |

### 4.3 Key properties

1. **Multiplicative:** $\det(AB) = \det(A)\det(B)$. The volume change of composing two transformations equals the product of their individual volume changes.
2. **Product of eigenvalues:** $\det(A) = \prod_{i=1}^{n} \lambda_i$. The determinant is the product of all eigenvalues (counting multiplicity). This is the most useful computational fact.
3. **Transpose invariant:** $\det(A) = \det(A^T)$.
4. **Inverse:** $\det(A^{-1}) = 1/\det(A)$.
5. **Triangular matrices:** the determinant is the product of the diagonal entries.
6. **Row operations:** swapping two rows flips the sign; multiplying a row by $c$ multiplies the determinant by $c$; adding a multiple of one row to another leaves the determinant unchanged.

### 4.4 Computing determinants

For $n > 3$, never compute the determinant by the permutation formula ($O(n!)$). Use:

1. **Row reduction to triangular form** — track row swaps (sign flips) and row scalings; then the determinant is the product of the pivots. $O(n^3)$.
2. **Product of eigenvalues** — if you already have the eigenvalues.
3. **Laplace / cofactor expansion** — $O(n!)$; only use for $n \le 3$ or sparse/theoretical matrices. The formula:

$$
\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(M_{ij}),
$$

where $M_{ij}$ is the matrix $A$ with row $i$ and column $j$ removed.

### 4.5 The determinant in practice

- If $\det(A)$ is very small (near zero), $A$ is **ill-conditioned** — nearly singular. Solving $Ax = b$ will amplify errors.
- In ML, $\det(\Sigma)$ appears in the Gaussian log-likelihood via $\log\det\Sigma$. For high-dimensional covariance matrices, computing this directly is unstable; use the Cholesky decomposition: $\log\det\Sigma = 2\sum_i \log L_{ii}$.
- The Jacobian determinant $|\det(J)|$ appears in change-of-variables (normalising flows, reparameterisation).

---

## 5. Eigenvalues and eigenvectors

### 5.1 The defining equation

For a square matrix $A \in \mathbb{R}^{n \times n}$, an **eigenvector** $v \neq 0$ and its **eigenvalue** $\lambda$ satisfy:

$$
A v = \lambda v.
$$

The matrix $A$ acts on $v$ by merely **scaling** it — $A$ does not change $v$'s direction. This is the most profound fact in linear algebra: every matrix has special directions that it leaves invariant.

### 5.2 Geometric intuition

Imagine $A$ as a transformation of space. Most vectors get both stretched and rotated when you apply $A$. But eigenvectors are special: they only get stretched (or shrunk, or flipped). The eigenvalue tells you the stretch factor.

- $\lambda = 2$: the eigenvector doubles in length.
- $\lambda = -1$: the eigenvector flips direction.
- $\lambda = 0.5$: the eigenvector shrinks to half.
- $\lambda = 0$: the eigenvector is annihilated — it lies in the nullspace.

### 5.3 Finding eigenvalues

Rearrange $Av = \lambda v$ to $(A - \lambda I)v = 0$. For a non-zero solution, $A - \lambda I$ must be singular:

$$
\det(A - \lambda I) = 0.
$$

This is the **characteristic polynomial** — a degree-$n$ polynomial in $\lambda$. Its roots are the eigenvalues. For $n \ge 5$, there is no closed-form formula (Abel-Ruffini); we use numerical methods (QR algorithm).

**Example — $2 \times 2$ matrix:**

$$
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}, \quad
\det(A - \lambda I) = \begin{vmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{vmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3.
$$

Roots: $\lambda_1 = 3$, $\lambda_2 = 1$. Eigenvectors: for $\lambda=3$, solve $(A-3I)v = 0 \implies v_1 = (1,1)$. For $\lambda=1$, $v_2 = (1,-1)$.

### 5.4 Key properties

| Property | Meaning |
|---|---|
| $\sum_i \lambda_i = \text{tr}(A)$ | The sum of eigenvalues equals the trace (sum of diagonal entries). |
| $\prod_i \lambda_i = \det(A)$ | The product of eigenvalues equals the determinant. |
| $\lambda_i(A^k) = \lambda_i(A)^k$ | Eigenvalues power with the matrix. |
| $\lambda_i(A^{-1}) = 1/\lambda_i(A)$ | Eigenvalues invert with the matrix (if $A$ is invertible). |
| $A$ is invertible $\iff$ all $\lambda_i \neq 0$ | A zero eigenvalue means a collapsed dimension. |
| Symmetric $A$ ($A = A^T$) | All eigenvalues are real; eigenvectors form an orthonormal basis. |

### 5.5 The spectral theorem (symmetric matrices)

If $A$ is symmetric ($A = A^T$), then:

1. All eigenvalues are **real**.
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**.
3. $A$ can be diagonalised by an orthogonal matrix:

$$
A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i \, q_i q_i^T,
$$

where $Q$ has orthonormal eigenvectors as columns and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$. This is the **spectral decomposition** — it decomposes $A$ into a sum of rank-1 matrices weighted by eigenvalues.

This is the single most useful fact in applied linear algebra. It means:
- $A$ acts by scaling each eigen-direction independently.
- The action is fully described by the eigen-pairs $(\lambda_i, q_i)$.
- $A^k = Q \Lambda^k Q^T$ — just power the eigenvalues.
- For positive definite $A$ (all $\lambda_i > 0$), $A^{1/2} = Q \Lambda^{1/2} Q^T$.

### 5.6 Positive definiteness

A symmetric matrix $A$ is:

- **Positive definite** if $x^T A x > 0$ for all $x \neq 0$, equivalently all $\lambda_i > 0$.
- **Positive semi-definite** if $x^T A x \ge 0$, equivalently all $\lambda_i \ge 0$.

Covariance matrices are always positive semi-definite. Hessians at a minimum are positive definite. Positive definiteness guarantees that the quadratic form $x^T A x$ is a convex bowl — gradient descent will find the unique minimum.

### 5.7 Eigen decomposition vs singular value decomposition

The eigendecomposition $A = Q\Lambda Q^{-1}$ exists only for diagonalisable square matrices. The SVD exists for **every** matrix, square or rectangular. For symmetric positive semi-definite matrices, eigenvalues = singular values. For general matrices, they are different concepts (see Section 7).

---

## 6. Matrix decompositions: the toolchest

Every decomposition answers a specific question. Here is the map:

| Decomposition | Formula | Requirements | Answers the question |
|---|---|---|---|
| LU | $A = LU$ | Square, invertible | How do I solve $Ax = b$ efficiently? |
| Cholesky | $A = LL^T$ | Symmetric, positive definite | Is my matrix PD? How do I sample from $\mathcal{N}(0, \Sigma)$? |
| QR | $A = QR$ | Any $m \times n$ | What is an orthonormal basis for the column space? |
| Eigendecomposition | $A = Q\Lambda Q^{-1}$ | Diagonalisable square | What are the invariant directions and their stretch factors? |
| SVD | $A = U\Sigma V^T$ | **Any** $m \times n$ matrix | What is the best low-rank approximation? What is the fundamental geometric action? |

### 6.1 LU decomposition

$$
A = LU, \quad L \text{ is lower-triangular (1s on diagonal)},\; U \text{ is upper-triangular}.
$$

LU decomposes $A$ into the product of a lower-triangular and an upper-triangular matrix. It is essentially Gaussian elimination recorded as matrix factors.

**Why it matters:** Solving $Ax = b$ becomes two easy triangular solves:

$$
\text{Solve } Ly = b \text{ (forward substitution)},\quad \text{Solve } Ux = y \text{ (backward substitution)}.
$$

Each takes $O(n^2)$ — much faster than inverting $A$ ($O(n^3)$) and numerically more stable. When you need to solve $Ax = b$ for many different $b$ (same $A$), compute $LU$ once, then do two triangular solves for each $b$.

In practice, LU is computed with **partial pivoting**: $PA = LU$, where $P$ is a permutation matrix that swaps rows to avoid small pivots. Without pivoting, LU can be numerically unstable.

### 6.2 Cholesky decomposition

For a symmetric positive definite matrix $A$:

$$
A = LL^T, \quad L \text{ is lower-triangular with positive diagonal entries}.
$$

This is roughly twice as fast as LU ($\frac{1}{3}n^3$ vs $\frac{2}{3}n^3$ flops) and extremely stable — no pivoting needed. Cholesky is also the definitive test for positive definiteness: attempt Cholesky; if it fails (negative pivot), $A$ is not positive definite.

**Key applications:**
- **Sampling from a multivariate Gaussian:** if $z \sim \mathcal{N}(0, I)$, then $\mu + Lz \sim \mathcal{N}(\mu, \Sigma)$, where $\Sigma = LL^T$.
- **Computing $\log\det\Sigma$:** $\log\det\Sigma = 2\sum_i \log L_{ii}$.
- **Solving normal equations:** $(X^T X)\beta = X^T y$. Compute Cholesky of $X^T X$, then forward/backward solve.

### 6.3 QR decomposition

For any $m \times n$ matrix $A$ ($m \ge n$):

$$
A = QR, \quad Q \text{ is } m \times m \text{ orthogonal } (Q^T Q = I),\; R \text{ is } m \times n \text{ upper-triangular}.
$$

The **thin QR** ($Q$ is $m \times n$, $R$ is $n \times n$) is often sufficient and cheaper.

QR produces an orthonormal basis for the column space of $A$. It is computed via Gram-Schmidt (theoretically clean but numerically unstable), Householder reflections (stable, the standard), or Givens rotations (good for sparse/structured matrices).

**Key applications:**
- **Least squares:** to solve $\min_x \|Ax - b\|^2$, compute $QR = A$, then $Rx = Q^T b$ (backward solve). This avoids forming the ill-conditioned $A^T A$.
- **Finding an orthonormal basis:** the columns of $Q$ are an orthonormal basis for $\mathcal{R}(A)$.
- **Eigenvalue algorithms:** the QR algorithm for finding eigenvalues iteratively applies QR decomposition.

### 6.4 Spectral decomposition (eigendecomposition)

For a diagonalisable square matrix:

$$
A = V \Lambda V^{-1},
$$

where columns of $V$ are eigenvectors, $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$. For symmetric $A$, $V = Q$ is orthogonal, so $Q^{-1} = Q^T$.

**Intuition:** change basis to the eigenvector basis ($V^{-1}$), scale each coordinate independently ($\Lambda$), change back ($V$). This is the "right" coordinate system for understanding $A$.

**Key applications:**
- **PCA:** eigendecomposition of the covariance matrix $\Sigma$ — the eigenvectors are principal components; eigenvalues measure variance explained.
- **Matrix powers:** $A^k = V \Lambda^k V^{-1}$. For Markov chains, $P^n = V \Lambda^n V^{-1}$ gives $n$-step transition probabilities.
- **Matrix exponential:** $e^{tA} = V e^{t\Lambda} V^{-1}$. Used for solving linear ODEs and continuous-time Markov chains.
- **Spectral clustering:** the second-smallest eigenvector of the graph Laplacian partitions the graph.

### 6.5 Singular Value Decomposition (SVD)

The SVD is the crown jewel of linear algebra. **Every** $m \times n$ matrix $A$ decomposes as:

$$
A = U \Sigma V^T,
$$

where:
- $U$ is $m \times m$ orthogonal — its columns are **left singular vectors** (orthonormal basis for $\mathbb{R}^m$).
- $\Sigma$ is $m \times n$ diagonal — $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$ are the **singular values**.
- $V$ is $n \times n$ orthogonal — its columns are **right singular vectors** (orthonormal basis for $\mathbb{R}^n$).

**Geometric interpretation (the best mental model):**

> Every linear transformation $A$ can be decomposed into three steps:
> 1. **Rotate** (or reflect) the input space: $V^T$.
> 2. **Scale** each coordinate independently by $\sigma_i$: $\Sigma$.
> 3. **Rotate** (or reflect) the output space: $U$.
>
> That is it. Every matrix does exactly this — rotate, scale, rotate.

### 6.6 SVD: the sum of rank-1 matrices

Equivalently, the SVD can be written as:

$$
A = \sum_{i=1}^{r} \sigma_i \, u_i v_i^T.
$$

This decomposes $A$ into a sum of rank-1 matrices, ordered by importance. Each term adds one "layer" of structure.

### 6.7 The Eckart-Young theorem and low-rank approximation

The **best rank-$k$ approximation** to $A$ (in both Frobenius and spectral norms) is:

$$
A_k = \sum_{i=1}^{k} \sigma_i \, u_i v_i^T.
$$

The error is $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$.

This is the mathematical foundation for:
- **PCA:** low-rank approximation of the data matrix.
- **Latent semantic analysis (LSA):** topic modelling via document-term matrix SVD.
- **Image compression:** keep only the top $k$ singular values/vectors.
- **Recommender systems (matrix factorisation / collaborative filtering):** approximate the user-item matrix.
- **LoRA (Low-Rank Adaptation for LLMs):** fine-tune by learning low-rank weight updates.

### 6.6 SVD vs eigendecomposition

| | Eigendecomposition | SVD |
|---|---|---|
| Applies to | Square, diagonalisable | **Any** $m \times n$ matrix |
| Factors | $A = V \Lambda V^{-1}$ | $A = U \Sigma V^T$ |
| Decomposition basis | One basis (eigenvectors) | Two bases (left + right singular vectors) |
| Relationship | For symmetric $A$: $\lambda_i = \sigma_i$, $V = U$ | $\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(AA^T)}$ |
| Numerical stability | Can be unstable for non-symmetric, near-defective matrices | Always stable and well-conditioned |

**Rule of thumb:** the SVD is the numerically reliable way to compute everything — rank, nullspace, condition number, least-squares solutions, low-rank approximations. When in doubt, use the SVD.

### 6.8 The condition number

The **condition number** of a matrix is:

$$
\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
$$

It measures how sensitive $Ax = b$ is to small perturbations. If $\kappa(A) = 10^k$, you can lose up to $k$ digits of accuracy when solving $Ax = b$. Large condition numbers = ill-conditioned = nearly singular in a practical sense.

In ML, feature matrices with highly correlated features have large condition numbers — this is (one reason) why we normalise and why we use regularisation (adding $\lambda I$ to $X^T X$ increases all eigenvalues by $\lambda$, reducing the condition number).

---

## 7. Applications in ML and quant finance

### 7.1 Principal Component Analysis (PCA)

Given data matrix $X \in \mathbb{R}^{N \times d}$ (centered), the covariance matrix is $\Sigma = \frac{1}{N-1} X^T X$. Eigendecompose:

$$
\Sigma = V \Lambda V^T.
$$

- $V$'s columns = principal components (directions of maximum variance).
- $\Lambda_{ii}$ = variance along the $i$-th component.
- The $k$-dimensional projection preserving maximum variance: $X_k = X V_k$, where $V_k$ is the first $k$ columns of $V$.
- The fraction of variance explained by the first $k$ components: $\sum_{i=1}^{k} \lambda_i / \sum_{i=1}^{d} \lambda_i$.

PCA via SVD: $X = U \Sigma V^T$ (no need to compute $X^T X$). Right singular vectors $V$ = principal components. Singular values squared = eigenvalues of $\Sigma$ (up to scaling).

### 7.2 Linear regression: the normal equations

Ordinary least squares: $\min_\beta \|y - X\beta\|^2$. The solution:

$$
\hat{\beta} = (X^T X)^{-1} X^T y.
$$

**Never compute $(X^T X)^{-1}$ explicitly.** It is $O(d^3)$ and numerically unstable. Instead:
- **Cholesky:** $X^T X = LL^T$, solve $L L^T \hat{\beta} = X^T y$ via forward/backward substitution.
- **QR:** $X = QR$, solve $R\hat{\beta} = Q^T y$.
- **SVD:** $\hat{\beta} = V\Sigma^{+} U^T y$ — works even when $X$ is rank-deficient (pseudoinverse).

### 7.3 Portfolio optimisation

Minimum-variance portfolio with target return $\mu_p$:

$$
\min_w \; w^T \Sigma w \quad \text{s.t.} \quad w^T \mathbf{1} = 1,\; w^T \mu = \mu_p.
$$

Solution: $w^* = \Sigma^{-1}$ applied to the constraints. eigendecompose $\Sigma$ to understand the risk structure — the dominant eigenvector is the direction of maximum portfolio variance. Cholesky: sample from $\mathcal{N}(0, \Sigma)$ for Monte Carlo simulation.

### 7.4 PageRank

PageRank is the stationary distribution of a Markov chain defined by the web graph. Let $P$ be the transition matrix. The PageRank vector $\pi$ satisfies:

$$
\pi = \pi P_{\text{Google}}, \quad P_{\text{Google}} = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T.
$$

$\pi$ is the dominant left eigenvector of $P_{\text{Google}}$ (eigenvalue = 1). Computed via the power method: $\pi^{(k+1)} = \pi^{(k)} P_{\text{Google}}$, which converges to $\pi$ because all other eigenvalues are $< 1$.

### 7.5 Neural networks: every forward pass

A linear (fully-connected) layer: $y = Wx + b$. The weight matrix $W$ transforms the input representation. The SVD of $W$ reveals its effective rank — how much information it preserves. Low effective rank → the layer is a bottleneck; you might reduce its width.

BatchNorm: $y = \gamma \frac{x - \mu}{\sigma} + \beta$. The parameters $\gamma$ and $\beta$ are per-feature scale and shift — a diagonal affine transformation applied after standardisation.

---

## 8. Common traps

| Trap | What happens | Fix |
|---|---|---|
| Computing inverses | $A^{-1}$ is $O(n^3)$, numerically unstable, and almost never needed. | Use `solve(A, b)` (LU) instead of `inv(A) @ b`. |
| Forming $X^T X$ for least squares | $X^T X$ has condition number $\kappa(X)^2$ — squared ill-conditioning. | Use QR or SVD directly. |
| Checking $\det(A) \approx 0$ for singularity | The determinant does not tell you *how* singular the matrix is. | Check $\sigma_{\min}$ or $\kappa(A)$. |
| Confusing eigenvalues and singular values | They are different for non-symmetric, non-square matrices. | $\sigma_i = \sqrt{\lambda_i(A^T A)}$. |
| Assuming eigendecomposition exists | Non-diagonalisable (defective) matrices don't have one. | SVD always exists. Use it. |
| Ignoring the condition number | $Ax = b$ silently gives garbage if $\kappa(A)$ is large. | Check $\kappa(A)$; regularise; use stable algorithms. |
| Cholesky on indefinite matrices | Fails or produces NaN. | Test positive definiteness first; use LDL or LU as fallback. |
| Forgetting centering before PCA | The first PC becomes the mean direction, not the variance direction. | Always center (and usually scale) before PCA. |
| Using `== 0` on floating-point determinant/eigenvalues | Floating-point arithmetic means values are never exactly zero. | Use a tolerance: check $\sigma_i < \epsilon \cdot \sigma_1$. |

---

## 9. Interview practice

### 9.1 "What is an eigenvalue, intuitively?"

**Answer.** An eigenvalue $\lambda$ is the factor by which a special direction (the eigenvector $v$) is stretched when you apply the matrix $A$. $Av = \lambda v$ means $A$ does not rotate $v$ — it just scales it. If you think of $A$ as a transformation of space, the eigenvectors are the directions that survive the transformation unchanged in direction.

### 9.2 "What is the SVD and why is it useful?"

**Answer.** The SVD decomposes any matrix $A$ into $U \Sigma V^T$: a rotation ($V^T$), a scaling ($\Sigma$), and another rotation ($U$). It is useful because (1) it exists for every matrix, unlike eigendecomposition; (2) it gives the best low-rank approximation via Eckart-Young — keep the top $k$ singular values; (3) it is numerically stable; (4) it reveals the fundamental structure: rank, range, nullspace, condition number, pseudoinverse. In ML, it underlies PCA, collaborative filtering, and LoRA fine-tuning.

### 9.3 "When would you use Cholesky vs LU vs QR?"

**Answer.**
- **Cholesky:** when $A$ is symmetric positive definite. Twice as fast as LU. Use for covariance matrices, normal equations (if well-conditioned), sampling from multivariate Gaussians.
- **LU:** when $A$ is square but not symmetric/P.D., and you need to solve $Ax = b$ for multiple $b$.
- **QR:** when you need an orthonormal basis (columns of $Q$), or for least squares — avoids forming $X^T X$. Also used in the QR eigenvalue algorithm.

### 9.4 "What does the determinant tell you about a matrix?"

**Answer.** Geometrically, it is the signed volume scaling factor of the transformation. $\det(A) = 0$ means the transformation collapses at least one dimension — the matrix is singular. $\det(A) > 0$ preserves orientation; $\det(A) < 0$ reverses it. Algebraically, it is the product of the eigenvalues. For large matrices, the condition number $\kappa(A)$ is more practically useful than the raw determinant.

### 9.5 "Why does $X^T X$ appear everywhere in least squares?"

**Answer.** Minimising $\|y - X\beta\|^2$ gives the normal equations: $X^T X \beta = X^T y$. Geometrically: $X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$. The residual $y - X\hat{\beta}$ is orthogonal to every column of $X$, so $X^T(y - X\hat{\beta}) = 0$, hence $X^T X \hat{\beta} = X^T y$. $X^T X$ is the Gram matrix of the features — its $(i,j)$ entry is the dot product of feature $i$ and feature $j$.

### 9.6 "Explain PCA in terms of eigendecomposition."

**Answer.** PCA finds the directions of maximum variance in the data. These directions are the eigenvectors of the covariance matrix $\Sigma = \frac{1}{N-1}X^T X$ (with data centered). The eigenvalue $\lambda_i$ is the variance along the $i$-th eigenvector. To reduce from $d$ dimensions to $k$, project onto the top $k$ eigenvectors (those with the largest eigenvalues). Equivalently, via the SVD of $X$: the right singular vectors $V$ are the principal components.

### 9.7 "Why not just invert the matrix?"

**Answer.** Three reasons: (1) **Cost:** inversion is $O(n^3)$ with a larger constant than solving $Ax = b$; you never need the full inverse — you need $A^{-1}b$, which is a solve, not an inversion. (2) **Stability:** computing $A^{-1}$ is less numerically stable than solving via LU/QR/Cholesky. (3) **Sparsity:** $A^{-1}$ is generally dense even when $A$ is sparse, destroying structure. In ML and optimisation, you should almost never see `inv()` in production code.

### 9.8 "Your Cholesky decomposition failed. What now?"

**Answer.** Cholesky requires the matrix to be symmetric positive definite. A failure (encountering a zero or negative pivot) means it is not. Options: (1) Add a small ridge: $A + \lambda I$ with $\lambda > 0$ makes it positive definite — this is Tikhonov regularisation. (2) Use the LDL decomposition (no positive-definiteness requirement). (3) Use LU with pivoting. Check: did you forget to center the data before computing the covariance? Are there duplicate or constant features?

---

## 10. Cheat sheet

| Concept | Remember |
|---|---|
| Dot product | $v^T w = \|v\|\|w\|\cos\theta$. Measures alignment. |
| Matrix × vector | $Ax$ = linear combination of columns of $A$, weights from $x$. |
| Rank | Dimension of column space = # of independent columns. |
| Nullspace | $\{x : Ax = 0\}$. Dimension = $n - r$. |
| Determinant | Signed volume scaling. $\det(A) = \prod \lambda_i$. $\det = 0 \iff$ singular. |
| Eigenvalue | $Av = \lambda v$. Invariant direction. |
| Eigendecomposition | $A = V\Lambda V^{-1}$. Diagonalisable only. |
| Positive definite | $x^T A x > 0$ for all $x \neq 0$ $\iff$ all $\lambda_i > 0$. |
| LU | $A = LU$. For solving $Ax = b$ efficiently. |
| Cholesky | $A = LL^T$. For PSD matrices. Sampling, log-det, fast solves. |
| QR | $A = QR$. Orthonormal basis; stable least squares. |
| SVD | $A = U\Sigma V^T$. Rotate → scale → rotate. Always exists. |
| Eckart-Young | Best rank-$k$ approx = keep top $k$ singular values. |
| Condition number | $\kappa = \sigma_{\max}/\sigma_{\min}$. Measures sensitivity. |
| PCA | Eigendecomposition of covariance = SVD of data matrix. |

---

## 11. Final checklist

Before using linear algebra in production:

- Do I really need to invert this matrix, or can I solve instead? (`solve(A, b)` not `inv(A) @ b`)
- Is my matrix symmetric positive definite → Cholesky? Square but not symmetric → LU? Any shape → QR or SVD?
- Have I checked the condition number? If $\kappa > 10^8$, regularise or expect trouble.
- For PCA: did I center (and scale) the data first?
- For gradients through linear algebra ops: does my autograd framework support `solve`, `cholesky`, `svd`? (PyTorch and JAX do.)
- Am I using the numerically stable path? (QR for least squares, not $X^T X$; Cholesky for log-det, not `det()`; SVD for rank, not counting non-zero eigenvalues.)

If you can explain what a determinant geometrically means, why the SVD is the universal decomposition, and how to choose the right decomposition for the task at hand, you have the linear algebra fluency that makes the rest of quantitative work feel natural.
