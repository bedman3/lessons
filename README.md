# Lessons

Readable, math-rendered lesson pages on quantitative topics — viewable locally or on GitHub Pages.

## Site

https://bedman3.github.io/lessons/

Each lesson is a markdown file plus a self-contained HTML viewer that renders it with MathJax (headings, tables, code blocks, and `$$`/`$` math). GitHub renders the markdown natively too.

## Lessons

| Lesson | Topics |
|---|---|
| [Probability & Statistics Foundations — Six-Chapter Course](prob-stats/index.html) | Discrete vs continuous probability, distributions, dependence, sampling, LLN, CLT, z/t/χ²/F inference, regression |
| [Applied Machine Learning — Six-Chapter Course](applied-ml/index.html) | Learning theory, regularized linear models, model families, evaluation, calibration, validation, leakage, production |
| [Monte Carlo, PDEs & Advanced Derivatives — Seven-Chapter Course](computational-finance/index.html) | Martingale pricing, simulation, variance reduction, Greeks, pricing PDEs, finite differences, American options, volatility calibration |
| [Machine Learning for Quant Research — Seven-Chapter Course](ml-quant/index.html) | Point-in-time data, financial validation, purging, multiple testing, signal/portfolio evaluation, non-stationarity, production |
| [Deep Learning Architectures in Practice — Seven-Chapter Course](deep-learning/index.html) | Representations and optimization, CNNs, recurrent models, Transformers and LLMs, embeddings, generative models, transfer, scaling, diagnosis |
| [Markov Chains for Quant Finance](markov-chains-quant-finance-viewer.html) | Transition matrices, stationary distributions, absorbing chains, credit migration, HMMs |
| [Conditional Probability](conditional-probability-viewer.html) | Bayes, chain rule, conjugate priors, information theory, KL divergence, inequalities |
| [Probability Toolkit](probability-toolkit-viewer.html) | Distributions, MGFs, expectation tricks, order statistics, CLT, heavy tails |
| [Statistics & Statistical Inference](statistics-inference-viewer.html) | Estimators, MLE, confidence intervals, hypothesis testing, power, Bayes |
| [Statistical Tests](statistical-tests-viewer.html) | z vs t, binomial mean/variance, Welch, chi-square, F tests, correlation |
| [Calculus & Optimization](calculus-optimization-viewer.html) | Derivatives, gradients, Taylor, convexity, Lagrange multipliers, Newton |
| [Stochastic Processes](stochastic-processes-viewer.html) | Random walks, gambler's ruin, martingales, Brownian motion, Ito's lemma |
| [Options Pricing](options-pricing-viewer.html) | Put-call parity, binomial model, Black-Scholes, Greeks, implied vol |
| [Machine Learning & Deep Learning Foundations](machine-learning-foundations-viewer.html) | Trees, gradient descent, backprop, neural nets, Transformers, LLMs |
| [Linear Algebra for ML & Quant Finance](linear-algebra-viewer.html) | Determinants, eigenvalues, SVD, LU/Cholesky/QR, PCA |
| [Factor Models & Residualization](factor-models-residualization-viewer.html) | OLS, alpha/beta, Fama–MacBeth, Barra, FWL theorem, factor hedging |
| [Linear Algebra — Textbook in Four Chapters](lin-alg/index.html) | Intuition, mechanics, eigenvalues & eigenvectors, decompositions — with worked examples |

## Recommended paths

- **Core mathematical prerequisites:** Probability & Statistics Foundations → Linear Algebra textbook → Calculus & Optimization → Stochastic Processes.
- **Quantitative finance:** core prerequisites → Options Pricing → Monte Carlo, PDEs & Advanced Derivatives → Factor Models & Residualization → Machine Learning for Quant Research → Markov Chains for Quant Finance.
- **Machine learning:** Probability & Statistics Foundations → Linear Algebra for ML & Quant Finance → Applied Machine Learning → Machine Learning & Deep Learning Foundations → Deep Learning Architectures in Practice.

The longer course pages teach concepts in sequence. The standalone lessons are concise references and alternate explanations.

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
prob-stats/             — probability and statistics foundations (6 chapters)
applied-ml/             — applied machine learning (6 chapters)
computational-finance/  — Monte Carlo, PDEs, and advanced derivatives (7 chapters)
ml-quant/               — machine learning for quant research (7 chapters)
deep-learning/          — deep learning architectures in practice (7 chapters)
assets/                 — shared course viewer styles and renderer
scripts/                — static validation for course content and links
```
