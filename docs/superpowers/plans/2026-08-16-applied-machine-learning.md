# Applied Machine Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a six-chapter, theory-led Applied Machine Learning course that connects statistical foundations to reliable modelling practice.

**Architecture:** Add `applied-ml/` using the shared course renderer and the established Markdown/viewer pair. Chapters move from generalization and linear models through algorithm families, evaluation, validation, and production diagnosis.

**Tech Stack:** Markdown, static HTML/CSS, shared ES-module viewer, MathJax 3, Python/Node structural tests.

**Spec:** `docs/superpowers/specs/2026-08-16-advanced-curriculum-expansion-design.md`

## Global Constraints

- Target upper-undergraduate through practitioner or early-researcher depth.
- Explain each algorithm through its inductive bias, objective, assumptions, and failure modes.
- Use `prob-stats/` and `lin-alg/` as linked prerequisites rather than duplicating them.
- Include derivations, numerical examples, practical selection logic, exercises, and solution outlines.
- Keep main exposition prerequisite-focused rather than interview-answer-focused.

---

### Task 1: Learning theory and linear models

**Files:**
- Create: `applied-ml/ch1-learning-generalization-lesson.md`
- Create: `applied-ml/ch2-linear-models-regularization-lesson.md`

- [ ] **Step 1:** Write Chapter 1 covering supervised learning notation, empirical versus population risk, loss functions, hypothesis classes, approximation/estimation/optimization error, bias-variance, capacity, generalization, distribution shift, and baselines.
- [ ] **Step 2:** Write Chapter 2 covering linear and logistic regression, likelihood/loss links, GLMs, regularization geometry, ridge/lasso/elastic net, optimization, collinearity, and coefficient versus prediction interpretation.
- [ ] **Step 3:** Run `python3 scripts/validate_lessons.py --markdown-only applied-ml`; expect both chapters to pass.

### Task 2: Model families and evaluation

**Files:**
- Create: `applied-ml/ch3-model-families-lesson.md`
- Create: `applied-ml/ch4-evaluation-calibration-lesson.md`

- [ ] **Step 1:** Write Chapter 3 covering k-nearest neighbours, trees, bagging, random forests, boosting, maximum-margin classification, kernels, and model-selection trade-offs.
- [ ] **Step 2:** Write Chapter 4 covering confusion matrices, cost-sensitive decisions, ROC/PR curves, ranking, proper scoring rules, calibration, imbalance, uncertainty, threshold choice, and metric failure modes.
- [ ] **Step 3:** Run the Markdown validator; expect four chapters to pass.

### Task 3: Validation and reliable practice

**Files:**
- Create: `applied-ml/ch5-validation-features-lesson.md`
- Create: `applied-ml/ch6-debugging-production-lesson.md`

- [ ] **Step 1:** Write Chapter 5 covering train/validation/test roles, cross-validation, nested selection, leakage, preprocessing pipelines, missingness, feature engineering/selection, temporal/group splits, and shift.
- [ ] **Step 2:** Write Chapter 6 covering interpretability, error slicing, ablations, experiment design, reproducibility, monitoring, drift, retraining, latency/cost, feedback loops, and governance.
- [ ] **Step 3:** Run the Markdown validator; expect all six chapters to pass.

### Task 4: Course shell and integration

**Files:**
- Create: `applied-ml/index.html`
- Create: six matching `applied-ml/*-viewer.html` pages
- Modify: `index.html`
- Modify: `README.md`

- [ ] **Step 1:** Add the course index with six chapter cards and prerequisite guidance.
- [ ] **Step 2:** Add matching viewers using `../assets/course-viewer.css` and `../assets/course-viewer.mjs`, complete previous/next/root navigation, and a blue-violet course theme.
- [ ] **Step 3:** Add the completed course to the public root and recommended paths.
- [ ] **Step 4:** Run course, root-link, Node, unittest, and whitespace validation.
- [ ] **Step 5:** Serve locally; inspect the index plus early, middle, and final chapters at desktop and narrow widths, checking content, MathJax, overflow, navigation, and browser errors.
- [ ] **Step 6:** Commit the course and push `master` to `origin`.
