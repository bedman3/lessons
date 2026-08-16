# Advanced Curriculum Expansion Design

**Date:** 2026-08-16  
**Status:** Approved in chat; awaiting written-spec review  
**Repository:** `bedman3/lessons`

## 1. Purpose

Expand the lessons site from a collection of strong topic references into a connected prerequisite curriculum for quantitative finance, applied machine learning, deep learning, and later interview or professional work.

The new material must remain concept-first. It should build mathematical intuition and technical fluency before introducing interview-style pressure or memorized response patterns. Topics should nevertheless be selected and taught because they recur in quantitative research, derivatives, machine-learning engineering, and related interviews.

## 2. Audience and Assumptions

The primary reader is progressing from upper-undergraduate mathematics toward practitioner or early-researcher competence.

Readers may have seen calculus, linear algebra, probability, and statistics before, but the curriculum must not assume those ideas are currently fluent. The new probability and statistics course provides the shared foundation. Advanced courses may include short refresh boxes, but should link back to the canonical explanation rather than duplicating it.

The curriculum assumes basic programming literacy. Code supports the theory; it does not replace mathematical explanation.

## 3. Scope and Delivery Boundaries

This program contains five independent mini-textbooks:

1. Probability & Statistics Foundations
2. Applied Machine Learning
3. Monte Carlo, PDEs & Advanced Derivatives
4. Machine Learning for Quant Research
5. Deep Learning Architectures in Practice

Each mini-textbook is an independently publishable subproject. It receives its own detailed implementation plan, content pass, verification pass, commit, and push. This prevents partially written courses from appearing on the public landing page.

After all five courses are published, a final integration pass will audit cross-course prerequisites, navigation, terminology, and reading paths.

### Non-goals

- A quant interview question bank or puzzle book
- Exhaustive graduate-level proofs for every theorem
- A full programming or Python course
- Live notebooks, server-side execution, accounts, or progress tracking
- Rewriting the existing lessons without a concrete integration need
- Treating formulas as memorization targets without motivation or assumptions

## 4. Curriculum Architecture

### 4.1 Probability & Statistics Foundations

This course is the common prerequisite and the first delivery milestone.

1. **Probability language and random variables**
   - Experiments, outcomes, events, sample spaces, and probability axioms
   - Discrete, continuous, and mixed random variables
   - PMFs, PDFs, and CDFs
   - Why density is not probability and why `P(X = x) = 0` can hold for possible continuous outcomes
2. **Distributions and moments**
   - Bernoulli, binomial, geometric, Poisson, uniform, exponential, Gaussian, and Student's t
   - Expectation, variance, covariance, correlation, quantiles, MGFs, and tail behaviour
   - Transformations, sums, and mixtures
3. **Joint, conditional, and dependent variables**
   - Joint and marginal distributions
   - Conditional probability and conditional expectation
   - Bayes' theorem, independence, conditional independence, and zero correlation
   - Law of total probability, tower property, and total variance
4. **Samples, estimators, LLN, and CLT**
   - Population, sample, statistic, estimator, estimate, and sampling distribution
   - Bias, consistency, efficiency, and mean-squared error
   - Law of large numbers versus central limit theorem
   - Standard deviation versus standard error
5. **z, t, chi-square, and F inference**
   - z-scores versus z-statistics
   - Why Student's t appears when variance is estimated
   - Degrees of freedom
   - Confidence intervals, test statistics, p-values, effect size, power, and multiple testing
   - Binomial inference and normal approximations
6. **Regression and modelling bridge**
   - Least squares and probabilistic interpretations
   - Residuals, uncertainty, diagnostics, and assumptions
   - Connections to maximum likelihood, ML loss functions, Monte Carlo error, time series, and financial models

### 4.2 Applied Machine Learning

1. Learning problems, generalization, bias-variance trade-offs, and approximation versus estimation error
2. Linear regression, logistic regression, GLMs, regularization, and optimization geometry
3. Trees, bagging, random forests, boosting, nearest neighbours, SVMs, and kernels
4. Metrics, decision thresholds, class imbalance, ranking, probability calibration, and uncertainty
5. Data splitting, cross-validation, leakage, preprocessing, feature engineering, and selection
6. Interpretability, error analysis, experiment design, distribution shift, monitoring, and production trade-offs

The course emphasizes model assumptions and selection logic rather than presenting an algorithm catalogue.

### 4.3 Monte Carlo, PDEs & Advanced Derivatives

1. No-arbitrage, equivalent measures, numeraires, martingales, and risk-neutral pricing intuition
2. Monte Carlo estimators, convergence, discretization, random-number generation, and confidence intervals
3. Variance reduction, quasi-Monte Carlo, path simulation, Greeks, and adjoint/pathwise/likelihood-ratio intuition
4. Derivation and interpretation of the Black-Scholes PDE and the Feynman-Kac connection
5. Explicit, implicit, and Crank-Nicolson finite differences; grids, boundaries, stability, and convergence
6. American exercise, free boundaries, trees, least-squares Monte Carlo, and path-dependent payoffs
7. Implied-volatility surfaces, local volatility, stochastic volatility, calibration, hedging error, and model risk

The existing stochastic-process and options-pricing lessons remain prerequisites and concise references.

### 4.4 Machine Learning for Quant Research

1. Economic hypotheses, prediction targets, labels, horizons, universes, and the research lifecycle
2. Market and alternative data, corporate actions, timestamps, missingness, point-in-time correctness, and survivorship bias
3. Time-series versus cross-sectional learning; rolling, expanding, and walk-forward validation
4. Look-ahead leakage, overlapping labels, purging, embargo, nested selection, and multiple comparisons
5. Signal metrics, rank information coefficients, portfolios, neutralization, turnover, costs, capacity, and realistic backtests
6. Non-stationarity, regimes, drift, ensembles, uncertainty, robustness, and stress testing
7. Research-to-production parity, monitoring, attribution, decay, retraining, and governance

This course connects the existing factor-model material to end-to-end empirical research without promising that predictive performance implies tradable performance.

### 4.5 Deep Learning Architectures in Practice

1. Representation learning, computational graphs, optimization, initialization, normalization, regularization, and gradient flow
2. CNNs, locality, translation equivariance, receptive fields, pooling, residual networks, and vision applications
3. RNNs, truncated backpropagation, vanishing gradients, LSTMs, GRUs, and sequence modelling
4. Attention and Transformers from projections through multi-head attention, masking, position information, and encoder/decoder variants
5. Autoencoders, metric and contrastive learning, embeddings, self-supervision, and representation evaluation
6. Latent-variable modelling, VAEs, GANs, score models, diffusion, and the trade-offs among generative families
7. Transfer learning, fine-tuning, parameter-efficient adaptation, scaling, data quality, evaluation, inference cost, and failure diagnosis

The existing ML/DL foundations lesson remains the concise overview; this course becomes the deeper architectural path.

## 5. Teaching Contract

Every chapter follows the same pedagogical sequence:

1. **Motivation:** the question or modelling problem the concept answers
2. **Intuition first:** plain-language mental models and small numerical examples
3. **Formal theory:** definitions, notation, assumptions, and theorem statements
4. **Derivation:** the origin of important formulas and what each step means
5. **Worked example:** a complete calculation or model walkthrough
6. **Computational view:** pseudocode or Python when computation clarifies the mathematics
7. **Application:** a connection to quant research, derivatives, ML, or DL
8. **Failure modes:** invalid assumptions, misleading outputs, and numerical or statistical traps
9. **Knowledge checks:** conceptual, calculation, derivation, and applied exercises
10. **Summary and dependencies:** retained ideas, prerequisite links, and the next chapter

Answers or concise solution outlines must accompany exercises. Exercises should test understanding, not merely symbol manipulation.

### 5.1 Required conceptual distinctions

The foundations course must explicitly resolve:

- Discrete mass versus continuous density
- `P(X = x)` versus `f(x)`
- Population distribution versus sampling distribution
- Standard deviation versus standard error
- z-score versus z-statistic
- Normal distribution versus Student's t distribution
- Law of large numbers versus central limit theorem
- Statistical significance versus practical significance
- Independence versus zero correlation
- Conditional association versus causal interpretation

## 6. Site and File Design

Each course uses a directory-based textbook structure modelled on `lin-alg/`:

```text
<course-slug>/
  index.html
  ch1-<topic>-lesson.md
  ch1-<topic>-viewer.html
  ch2-<topic>-lesson.md
  ch2-<topic>-viewer.html
  ...
```

Initial course slugs are:

- `prob-stats/`
- `applied-ml/`
- `computational-finance/`
- `ml-quant/`
- `deep-learning/`

Every viewer must:

- Render its paired Markdown file with MathJax
- Expose course table-of-contents navigation
- Link to the previous and next chapters where applicable
- Link back to the root site
- Remain usable at desktop and narrow viewport widths
- Display a visible error if its Markdown file cannot be loaded

The root `index.html` and `README.md` will organize content into learning paths and show the recommended prerequisite order. A course is linked from the public root only after every chapter in its initial release passes verification.

Existing lessons remain available as quick references. New chapters link to them where they provide useful alternate explanations, and existing files are changed only when necessary for navigation or terminology consistency.

## 7. Content and Technical Quality

### 7.1 Mathematical quality

- Define notation before use.
- State the assumptions behind formulas and the consequences of violating them.
- Distinguish exact results, approximations, heuristics, and modelling conventions.
- Check dimensional consistency and limiting or boundary cases where applicable.
- Avoid silently switching between population, sample, risk-neutral, and real-world quantities.

### 7.2 Computational quality

- Prefer small, reproducible examples.
- Use numerically stable formulations when relevant and explain why.
- Label pseudocode as pseudocode and runnable examples as runnable code.
- Avoid dependencies that are unnecessary for understanding the concept.
- Explain simulation error, discretization error, estimation error, and model error as distinct sources.

### 7.3 Editorial quality

- Use consistent terminology across courses.
- Keep paragraphs readable and headings descriptive.
- Introduce jargon only after the underlying idea.
- Include cross-links where a prerequisite is genuinely needed, not as decorative references.
- Avoid interview-answer phrasing in the main exposition.

## 8. Validation and Acceptance Criteria

Each course must pass the following checks before publication:

1. Every planned chapter exists as a Markdown/viewer pair.
2. Viewer fetch paths resolve when served through the repository's local HTTP server.
3. Internal links, previous/next navigation, course-index links, and root links resolve.
4. Markdown headings have a coherent hierarchy and no unintended duplicates.
5. Display and inline mathematics use supported delimiters and render without obvious source leakage.
6. Code fences are balanced and tables are structurally valid.
7. A representative sample of pages is visually inspected at desktop and narrow widths.
8. The root landing page and README describe the course accurately.
9. The working tree contains no unrelated modifications in the course commit.
10. The committed course is pushed successfully to the configured `origin` remote.

Where the repository lacks automated checks, small validation scripts may be added if they are reusable across all courses.

## 9. Delivery Sequence

1. Probability & Statistics Foundations
2. Applied Machine Learning
3. Monte Carlo, PDEs & Advanced Derivatives
4. Machine Learning for Quant Research
5. Deep Learning Architectures in Practice
6. Cross-course navigation and curriculum audit

The first five milestones are separate implementation projects. Completion of one does not require waiting for all later courses. The final audit checks prerequisite chains, duplicated explanations, conflicting notation, broken links, and the end-to-end reading experience.

## 10. Publishing and Version Control

Changes are committed in course-sized increments after verification. Commit messages identify the completed course or integration pass. After each course commit, the branch is pushed to the existing `origin` remote so the repository's current GitHub Pages workflow can publish it.

No force push, history rewrite, or destructive cleanup is part of this work. If the remote rejects a push because it has diverged, publishing pauses for a read-only comparison and a safe integration decision.

## 11. Success Criteria

The expansion succeeds when:

- A reader can follow a visible path from undergraduate probability and statistics into each advanced course.
- Foundational distinctions are explained with both intuition and formal definitions.
- Each advanced topic connects theory to realistic quantitative or ML work without becoming an interview-cramming guide.
- Exercises and worked examples require the reader to reason about assumptions and failure modes.
- Every published course is navigable, math-rendered, locally verified, committed, and pushed.
- The existing concise lessons remain useful and discoverable alongside the deeper courses.
