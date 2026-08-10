# Chapter 1 — Matrices as Machines: Intuition First

> Before any formula: what does a matrix *do*? This chapter builds the visual intuition that makes every later chapter feel obvious.

## 1.1 Vectors are arrows; matrices are machines

A vector is an arrow from the origin to a point. That is all you need. Two vectors $v = (2, 1)$ and $w = (-1, 3)$ are just two arrows in the plane.

**A matrix is a machine that takes an arrow in and spits an arrow out.** You feed it a vector; it moves the vector — stretches it, spins it, flips it, squashes it — and out comes a new vector.

$$
\text{input arrow } \mathbf v \xrightarrow{\;\; A \;\;} \text{output arrow } A\mathbf v
$$

The entire subject of linear algebra is: *understand the machine by watching what it does to arrows.*

## 1.2 The simplest machine: diagonal matrices

Look at:

$$
A = \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}
$$

Feed it the horizontal arrow $e_1 = (1, 0)$: you get $(2, 0)$ — it stretches horizontal arrows ×2. Feed it $e_2 = (0, 1)$: you get $(0, 0.5)$ — it squashes vertical arrows to half.

Diagonal matrices stretch each axis independently. **That is the whole job description: a diagonal matrix = independent stretching along each axis.**

Most matrices are not diagonal, but the deep secret of linear algebra is that *almost every matrix is a diagonal matrix hiding in disguise* — it stretches along directions that are not the obvious axes. Finding those hidden directions is exactly the eigenvalue problem (Chapter 3).

## 1.3 Multiplying a matrix by a vector: three views

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ and $\mathbf v = (x, y)$:

**View 1 — Dot products (coordinates out):**

$$
A\mathbf v = \begin{pmatrix} a x + b y \\ c x + d y \end{pmatrix}
$$

Each output coordinate is a dot product of a row of $A$ with $\mathbf v$.

**View 2 — Columns (the good one):**

$$
A\mathbf v = x \begin{pmatrix} a \\ c \end{pmatrix} + y \begin{pmatrix} b \\ d \end{pmatrix}
$$

The output is a **linear combination of the columns of $A$**, weighted by the components of $\mathbf v$. Read it like a recipe: take $x$ of column 1, add $y$ of column 2.

**View 3 — The machine (what $A$ does to the grid):**

Apply $A$ to every point of the plane. The unit square (corners $(0,0), (1,0), (0,1), (1,1)$) becomes a parallelogram with corners $(0,0), \text{col}_1, \text{col}_2, \text{col}_1 + \text{col}_2$. **The columns of $A$ are where the unit arrows land.**

This third view is the intuition pump for everything: *the columns of $A$ tell you where the basis arrows land, and everything else follows by linearity.*

## 1.4 The four moves of 2×2 machines

| Machine | Matrix | What it does | Determinant |
|---|---|---|---|
| Stretch | $\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$ | Stretches ×2 along x, ×3 along y | 6 (area ×6) |
| Rotate 90° | $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ | Spins everything 90° | 1 (area preserved) |
| Reflect | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Flips across the x-axis | −1 (orientation flips) |
| Project | $\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$ | Collapses everything onto the line $y = x$ | 0 (area destroyed) |
| Shear | $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ | Slides x-proportionally to height | 1 |

Check the projection by hand: $P^2 = P$ — projecting twice is the same as once:

$$
\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}^2 =
\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}
$$

A machine that "does nothing more the second time" is a projection. A machine whose determinant is 0 collapses the plane to a line — information is destroyed, and there is no way back (no inverse).

## 1.5 The determinant as "volume change"

The **determinant** is the signed area multiplier of the machine:

- $|\det(A)|$ = how many times the machine multiplies areas/volumes.
- Sign = whether orientation is preserved (+) or flipped (−).
- $\det(A) = 0$ ⇔ the machine collapses at least one dimension ⇔ no inverse.

For a 2×2 matrix:

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

**Why $ad - bc$?** The unit square becomes the parallelogram spanned by $(a,c)$ and $(b,d)$. Its area is $|ad - bc|$. The formula *is* the area computation.

**Multiplicativity:** applying machine $A$ then machine $B$ multiplies areas by $\det(A)\det(B)$ — so $\det(AB) = \det(A)\det(B)$. If each step doubles area, two steps quadruple it.

## 1.6 Multiplying two matrices: composing machines

$AB$ means "run $B$ first, then $A$". This is why order matters and why it is not commutative:

$$
AB \neq BA \quad \text{in general}
$$

Rotate-then-stretch is generally not stretch-then-rotate. The columns of $AB$ are $A$ applied to the columns of $B$:

$$
(AB)_j = A(B_j) \quad \text{(column } j \text{ of } AB \text{ = } A \text{ times column } j \text{ of } B)
$$

## 1.7 The big picture of this chapter

1. Vectors are arrows. Matrices are machines that move arrows.
2. The columns of $A$ are where the basis arrows land — everything follows from that.
3. Diagonal matrices stretch axes independently; most matrices just do this along hidden directions (Chapter 3).
4. The determinant measures volume change; zero means collapse; negative means flip.
5. Multiplication composes machines, so order matters.

**Gotcha:** a matrix is not a "grid of numbers" — it is an action. When you meet a new matrix, first ask: *what does it do to space?* The numbers are just the instruction manual.

## 1.8 Practice

1. Where does $\mathbf v = (1, 1)$ land under $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$? *(Answer: column 1 + column 1 again… no — $A\mathbf v = (3, 4)$: $x\cdot\text{col}_1 + y\cdot\text{col}_2 = (2,1) + (1,3) = (3,4)$.)*
2. Compute $\det$ of the rotation matrix $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ and explain the sign. *(Answer: $0\cdot0 - (-1)(1) = 1$; positive because rotation preserves orientation.)*
3. Is $\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$ invertible? *(Answer: $\det = 4 - 4 = 0$; not invertible — column 2 is 2× column 1, the machine collapses the plane to a line.)*
4. For a shear $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$, where does the unit square go? *(Answer: a parallelogram of area 1 — shear preserves area: $\det = 1$.)*
