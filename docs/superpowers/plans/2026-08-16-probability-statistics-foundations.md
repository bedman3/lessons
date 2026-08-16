# Probability & Statistics Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a six-chapter, concept-first probability and statistics prerequisite course with complete navigation, exercises, and integration into the lessons site.

**Architecture:** Add a self-contained `prob-stats/` textbook modelled on `lin-alg/`. Each chapter has one canonical Markdown source and one matching viewer; a course index and the site root expose the reading path only after all chapters pass validation.

**Tech Stack:** Markdown, static HTML/CSS, browser JavaScript, MathJax 3, local Python HTTP server, shell-based structural validation.

**Spec:** `docs/superpowers/specs/2026-08-16-advanced-curriculum-expansion-design.md`

## Global Constraints

- Target upper-undergraduate through practitioner or early-researcher depth.
- Lead with intuition, then definitions, derivations, worked examples, computation, applications, failure modes, exercises, solutions, and summary.
- Keep existing lessons intact except for navigation and terminology links.
- Every Markdown chapter has a matching viewer with previous/next, course-index, and root navigation.
- A course reaches the public root only after every initial chapter passes verification.
- Do not frame the main exposition as interview preparation.

---

### Task 1: Add reusable course validation

**Files:**
- Create: `scripts/validate_lessons.py`

**Interfaces:**
- Consumes: repository root and course directory passed as command-line paths.
- Produces: process exit code `0` when every `*-lesson.md` has a corresponding `*-viewer.html`, referenced local links exist, Markdown fences are balanced, and expected navigation markers occur; nonzero with file-specific messages otherwise.

- [ ] **Step 1: Define failing fixture checks**

Create temporary invalid examples in the validator's `--self-test` mode: a lesson without a viewer, an unbalanced code fence, and an HTML link to a missing local file. Assert that each detector rejects its fixture.

- [ ] **Step 2: Run the missing validator**

Run: `python3 scripts/validate_lessons.py --self-test`

Expected: failure because `scripts/validate_lessons.py` does not exist.

- [ ] **Step 3: Implement validation**

Use only the Python standard library. Expose `validate_course(root: Path, course: Path) -> list[str]`, `validate_markdown(path: Path) -> list[str]`, and `validate_html_links(root: Path, path: Path) -> list[str]`. Ignore `http:`, `https:`, `mailto:`, fragment-only links, and JavaScript-generated dynamic markup when checking local targets.

- [ ] **Step 4: Verify validator behavior**

Run: `python3 scripts/validate_lessons.py --self-test`

Expected: `Self-test passed` and exit code `0`.

- [ ] **Step 5: Commit validator**

Run:

```bash
git add scripts/validate_lessons.py
git commit -m "Add static lesson validation"
```

### Task 2: Write chapters 1–3

**Files:**
- Create: `prob-stats/ch1-probability-language-lesson.md`
- Create: `prob-stats/ch2-distributions-moments-lesson.md`
- Create: `prob-stats/ch3-joint-conditional-lesson.md`

**Interfaces:**
- Consumes: notation from the existing probability, conditional-probability, and statistics lessons.
- Produces: canonical definitions and notation used by Chapters 4–6 and later courses.

- [ ] **Step 1: Write Chapter 1**

Cover sample spaces, events, axioms, random variables, discrete/continuous/mixed types, PMF/PDF/CDF relationships, transformations, quantiles, and the density-versus-probability distinction. Include a Bernoulli example, uniform-density example, mixed-distribution example, and solutions.

- [ ] **Step 2: Write Chapter 2**

Cover expectation as a weighted average/integral, variance, covariance, correlation, moments, MGFs, quantiles, and Bernoulli/binomial/geometric/Poisson/uniform/exponential/Gaussian/Student's t distributions. Derive binomial mean/variance and exponential memorylessness; distinguish thin and heavy tails.

- [ ] **Step 3: Write Chapter 3**

Cover joint/marginal/conditional distributions, Bayes, independence, conditional independence, zero correlation, conditional expectation, tower property, total variance, covariance decomposition, and transformations. Include a correlated-but-dependent counterexample and a complete Bayesian updating calculation.

- [ ] **Step 4: Run Markdown checks**

Run: `python3 scripts/validate_lessons.py --markdown-only prob-stats`

Expected: all three Markdown files pass heading, fence, and delimiter checks.

- [ ] **Step 5: Commit Chapters 1–3**

Run:

```bash
git add prob-stats/ch1-probability-language-lesson.md prob-stats/ch2-distributions-moments-lesson.md prob-stats/ch3-joint-conditional-lesson.md
git commit -m "Add probability foundations chapters"
```

### Task 3: Write chapters 4–6

**Files:**
- Create: `prob-stats/ch4-sampling-clt-lesson.md`
- Create: `prob-stats/ch5-classical-inference-lesson.md`
- Create: `prob-stats/ch6-regression-bridge-lesson.md`

**Interfaces:**
- Consumes: random-variable, distribution, moment, and conditional-expectation notation from Chapters 1–3.
- Produces: shared prerequisites for all four advanced courses.

- [ ] **Step 1: Write Chapter 4**

Cover population/sample/statistic/estimator/estimate, empirical distributions, bias/variance/MSE/consistency, sampling distributions, standard errors, LLN versus CLT, delta-method intuition, bootstrap intuition, and Monte Carlo error. Derive the sample-mean variance and standardized CLT.

- [ ] **Step 2: Write Chapter 5**

Cover z-score versus z-statistic, t statistics, chi-square and F relationships, degrees of freedom, confidence intervals, null testing, p-values, effect sizes, power, multiple testing, binomial inference, and assumption checks. Include a decision table and complete z, Welch-t, and binomial examples.

- [ ] **Step 3: Write Chapter 6**

Cover population versus fitted regression, OLS geometry and probability model, normal equations, coefficient uncertainty, residual diagnostics, heteroskedasticity/autocorrelation, robust errors, maximum-likelihood links, logistic regression bridge, loss functions, and finance/ML applications.

- [ ] **Step 4: Run Markdown checks**

Run: `python3 scripts/validate_lessons.py --markdown-only prob-stats`

Expected: all six Markdown files pass.

- [ ] **Step 5: Commit Chapters 4–6**

Run:

```bash
git add prob-stats/ch4-sampling-clt-lesson.md prob-stats/ch5-classical-inference-lesson.md prob-stats/ch6-regression-bridge-lesson.md
git commit -m "Add statistical inference foundations chapters"
```

### Task 4: Add course viewers and table of contents

**Files:**
- Create: `prob-stats/index.html`
- Create: `prob-stats/ch1-probability-language-viewer.html`
- Create: `prob-stats/ch2-distributions-moments-viewer.html`
- Create: `prob-stats/ch3-joint-conditional-viewer.html`
- Create: `prob-stats/ch4-sampling-clt-viewer.html`
- Create: `prob-stats/ch5-classical-inference-viewer.html`
- Create: `prob-stats/ch6-regression-bridge-viewer.html`

**Interfaces:**
- Consumes: the six canonical Markdown chapter files.
- Produces: browser-rendered course pages and complete course navigation.

- [ ] **Step 1: Add the course index**

Follow the `lin-alg/index.html` card layout. Describe the prerequisite path, list all six chapters, and expose Read and Markdown links.

- [ ] **Step 2: Add viewers for Chapters 1–3**

Reuse the existing safe Markdown renderer and MathJax configuration. Set correct fetch targets, titles, descriptions, previous/next links, course-index links, root links, and a teal/blue course color theme.

- [ ] **Step 3: Add viewers for Chapters 4–6**

Apply the same viewer contract, ending Chapter 6 navigation at the course table of contents.

- [ ] **Step 4: Validate the course**

Run: `python3 scripts/validate_lessons.py . prob-stats`

Expected: `Validated prob-stats: 6 lessons, 7 HTML pages`.

- [ ] **Step 5: Commit course shell**

Run:

```bash
git add prob-stats/index.html prob-stats/*-viewer.html
git commit -m "Add probability and statistics course navigation"
```

### Task 5: Integrate, render, and publish

**Files:**
- Modify: `index.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed `prob-stats/` course.
- Produces: public discovery path from the site root and repository documentation.

- [ ] **Step 1: Add root curriculum card**

Add Probability & Statistics Foundations before shorter probability references. Describe it as a six-chapter prerequisite course and link to `prob-stats/index.html`.

- [ ] **Step 2: Update README curriculum table**

Add the course and recommended reading order while retaining the existing lesson list.

- [ ] **Step 3: Run structural validation**

Run: `python3 scripts/validate_lessons.py . prob-stats && git diff --check`

Expected: validator success and no whitespace errors.

- [ ] **Step 4: Serve and inspect**

Run `./serve.sh 8080`, request the root, course index, and all six viewers over HTTP, and inspect the course index plus representative early, middle, and final chapters at desktop and narrow widths. Confirm content renders, navigation works, formulas typeset, code blocks remain intact, and missing-fetch errors are visible rather than silent.

- [ ] **Step 5: Commit integration**

Run:

```bash
git add index.html README.md
git commit -m "Publish probability and statistics foundations course"
```

- [ ] **Step 6: Push**

Run: `git push origin master`

Expected: the remote accepts all foundation-course commits.
