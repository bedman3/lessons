# Chapter 2 — Mechanics: The Moves You Must Be Fluent In

> The hands-on chapter: solving $Ax = b$, inverses, rank, elimination, and determinants — every move with a worked example you can replay by hand.

## 2.1 What $Ax = b$ means

We want an arrow $\mathbf x$ that the machine $A$ sends to $\mathbf b$:

$$
A\mathbf x = \mathbf b
$$

Column view: **$\mathbf b$ must be a linear combination of the columns of $A$.** The system is solvable exactly when $\mathbf b$ lies in the column space. For a square $A$, the solution exists and is unique when the machine does not collapse anything — i.e., when $\det(A) \neq 0$.

**Worked example.** Solve:

$$
\begin{cases} 2x + y = 4 \\ x - 2y = -3 \end{cases}
\qquad
A = \begin{pmatrix} 2 & 1 \\ 1 & -2 \end{pmatrix},\;\; \mathbf b = \begin{pmatrix} 4 \\ -3 \end{pmatrix}
$$

Check $\mathbf b$ against the columns: $2(1) + (2) = 4$ and $1 - 2(2) = -3$, so $(x, y) = (1, 2)$. Verify: $2(1) + 2 = 4$ ✓ and $1 - 4 = -3$ ✓.

**Why the check matters:** guessing is fine for 2×2, but the method that scales is elimination.

## 2.2 Elimination and back-substitution (the method that scales)

Gaussian elimination turns $A$ into an upper-triangular matrix by row operations, then solves from the bottom up.

**Worked example.** Solve $A\mathbf x = \mathbf b$ with $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$, $\mathbf b = \begin{pmatrix} 5 \\ 16 \end{pmatrix}$.

1. **Eliminate below the pivot:** subtract 3 × row 1 from row 2 (so the 6 becomes 0):

$$
\begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix} \mathbf x = \begin{pmatrix} 5 \\ 1 \end{pmatrix}
$$

2. **Back-substitute:** from row 2, $y = 1$. From row 1, $2x + 1 = 5$, so $x = 2$.

3. **Verify:** $2(2) + 1 = 5$ ✓; $6(2) + 4(1) = 16$ ✓.

The row operation "row 2 ← row 2 − 3 × row 1" is itself a matrix multiplication — by the **elimination matrix** $E = \begin{pmatrix} 1 & 0 \\ -3 & 1 \end{pmatrix}$. Keep track of elimination matrices and you have built $A$'s LU decomposition (Chapter 4) for free.

## 2.3 The inverse: the machine's "undo" button

The inverse $A^{-1}$ satisfies $A^{-1}A = AA^{-1} = I$. It undoes what $A$ does. It exists iff $\det(A) \neq 0$ — iff the machine collapses nothing.

**2×2 formula:**

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

**Worked example.** $A = \begin{pmatrix} 2 & 1 \\ 1 & -2 \end{pmatrix}$: $\det = 2(-2) - 1(1) = -5$.

$$
A^{-1} = -\frac{1}{5}\begin{pmatrix} -2 & -1 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} 0.4 & 0.2 \\ 0.2 & -0.4 \end{pmatrix}
$$

Verify: $A A^{-1} = \begin{pmatrix} 2 & 1 \\ 1 & -2 \end{pmatrix}\begin{pmatrix} 0.4 & 0.2 \\ 0.2 & -0.4 \end{pmatrix} = \begin{pmatrix} 0.8+0.2 & 0.4-0.4 \\ 0.4-0.4 & 0.2+0.8 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ ✓

**Never compute $A^{-1}$ just to solve $Ax = b$.** Solve directly by elimination: it is faster, more stable, and you do not need the whole inverse — only $A^{-1}b$. (`solve(A, b)`, never `inv(A) @ b`.)

**Gauss–Jordan shortcut (any size):** write $[A \mid I]$, row-reduce until the left block is $I$; the right block is $A^{-1}$.

## 2.4 Rank: how many dimensions the machine keeps

The **rank** $r$ of $A$ is the number of independent columns (equivalently, independent rows). It is the dimension of the image — how many directions of space survive the machine.

| Rank | Meaning | Solvability of $Ax = b$ |
|---|---|---|
| $r = n$ (full) | Machine collapses nothing; $\det \neq 0$ | Unique solution for every $\mathbf b$ |
| $r < n$ | Some directions are destroyed | Solutions only if $\mathbf b$ is in the column space; infinitely many if it is |

**Worked example.** $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$. Column 2 = 2 × column 1 → $r = 1$. The image is the line through $(1, 2)$. The system $A\mathbf x = (3, 6)$ has infinitely many solutions ($x_1 + 2x_2 = 3$ — any point on that line); the system $A\mathbf x = (3, 5)$ has none.

**The nullspace** $\mathcal{N}(A) = \{\mathbf x : A\mathbf x = 0\}$ collects the arrows the machine kills. Its dimension is $n - r$. The rank–nullity theorem:

$$
\dim \mathcal{N}(A) + \text{rank}(A) = n
$$

## 2.5 Determinants: how to compute without tears

For 2×2: $ad - bc$. For 3×3 and above, **never** use the permutation formula ($n!$ terms). Use row reduction:

1. Swap rows → flip sign.
2. Multiply a row by $c$ → multiply the determinant by $c$.
3. Add a multiple of one row to another → determinant unchanged.
4. Triangle form → determinant = product of the diagonal.

**Worked example.** $A = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$ (symmetric tridiagonal).

Row-reduce (add $-\tfrac12$ row 1 to row 2, then $-\tfrac23$ row 2 to row 3 — record no sign changes):

$$
\det = 2 \times \frac{3}{2} \times \frac{4}{3} = 4
$$

Check via eigenvalues (Chapter 3): this matrix has eigenvalues $2 + \sqrt2, 2, 2 - \sqrt2$, whose product is $(2)(4 - 2) = 4$ ✓. The trace is $2 + 2 + 2 = 6$ and the eigenvalue sum is $6$ ✓ — both sanity checks agree.

## 2.6 Determinant properties worth their weight in gold

| Property | Why it matters |
|---|---|
| $\det(AB) = \det(A)\det(B)$ | Volume change composes. |
| $\det(A^T) = \det(A)$ | Rows and columns are equally informative. |
| $\det(A^{-1}) = 1/\det(A)$ | Undo shrinks the inverse volume. |
| $\det(cA) = c^n \det(A)$ | Stretching *all* coordinates scales volume by $c^n$. |
| $\det = \prod \lambda_i$ | Product of eigenvalues — the fastest check. |
| $\det(A) \approx 0$ | Means nearly-singular — but use the **condition number** $\kappa = \sigma_{\max}/\sigma_{\min}$ to judge *how* singular. |

## 2.7 The moves you must be fluent in (checklist)

- Solve $Ax = b$ by elimination + back-substitution.
- Invert a 2×2 in your head ($\frac{1}{ad-bc}$ pattern); know that you should never invert a big matrix just to solve.
- Read off rank from independent columns; know $\dim\mathcal{N} = n - r$.
- Compute determinants by row reduction, not expansion.
- Explain why $AB \neq BA$ and what $AB$'s columns are.
- Know: solvable ⇔ $\mathbf b$ in column space; unique ⇔ $\det \neq 0$.

## 2.8 Practice

1. Solve $\begin{cases} 3x - y = 7 \\ x + 2y = 7 \end{cases}$ *(Answer: $x = 3$, $y = 2$: check $9 - 2 = 7$ ✓, $3 + 4 = 7$ ✓.)*
2. Invert $\begin{pmatrix} 3 & -1 \\ 1 & 2 \end{pmatrix}$. *(Answer: $\det = 7$, so $A^{-1} = \frac{1}{7}\begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}$.)*
3. What is the rank of $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 1 & 1 & 3 \end{pmatrix}$? *(Answer: row 3 = row 1 + row 2, so $r = 2$.)*
4. Find the nullspace of $\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$. *(Answer: all multiples of $(2, -1)$ — check $1(2) + 2(-1) = 0$ ✓.)*
