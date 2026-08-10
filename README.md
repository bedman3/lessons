# Lessons

Readable, math-rendered lesson pages on quantitative topics — viewable locally or on GitHub Pages.

## Site

https://bedman3.github.io/lessons/

Each lesson is a markdown file plus a self-contained HTML viewer that renders it with MathJax (headings, tables, code blocks, and `$$`/`$` math). GitHub renders the markdown natively too.

## Lessons

| Lesson | Topics |
|---|---|
| [Markov Chains for Quant Finance](markov-chains-quant-finance-viewer.html) | Transition matrices, stationary distributions, absorbing chains, credit migration, HMMs |
| [Conditional Probability](conditional-probability-viewer.html) | Bayes, chain rule, conjugate priors, information theory, KL divergence, inequalities |
| [Machine Learning & Deep Learning Foundations](machine-learning-foundations-viewer.html) | Trees, gradient descent, backprop, neural nets, Transformers, LLMs |
| [Linear Algebra for ML & Quant Finance](linear-algebra-viewer.html) | Determinants, eigenvalues, SVD, LU/Cholesky/QR, PCA |
| [Factor Models & Residualization](factor-models-residualization-viewer.html) | OLS, alpha/beta, Fama–MacBeth, Barra, FWL theorem, factor hedging |
| [Linear Algebra — Textbook in Four Chapters](lin-alg/index.html) | Intuition, mechanics, eigenvalues & eigenvectors, decompositions — with worked examples |

## Local viewing

```bash
./serve.sh        # serves http://localhost:8080
```

## Layout

```
index.html              — table of contents
*-lesson.md             — lesson content (GitHub-style math delimiters)
*-viewer.html           — standalone viewer that fetches the .md and renders with MathJax
lin-alg/                — the linear algebra textbook (4 chapters)
```
