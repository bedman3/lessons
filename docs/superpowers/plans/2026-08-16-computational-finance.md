# Monte Carlo, PDEs & Advanced Derivatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a seven-chapter computational derivatives course linking no-arbitrage theory to simulation, PDE methods, early exercise, volatility models, calibration, and model risk.

**Architecture:** Add `computational-finance/` using canonical Markdown lessons, matching shared-renderer viewers, and a course index. Existing stochastic-process and options-pricing lessons provide concise prerequisites; this course derives and extends their machinery.

**Tech Stack:** Markdown, MathJax 3, static HTML/CSS, shared ES-module viewer, Python/Node structural tests.

**Spec:** `docs/superpowers/specs/2026-08-16-advanced-curriculum-expansion-design.md`

## Global Constraints

- Target upper-undergraduate through practitioner or early-researcher depth.
- Distinguish model assumptions, numerical error, statistical calibration error, and hedging/model risk.
- Lead with financial intuition, then derive the mathematics and algorithms.
- Include worked calculations, pseudocode, stability/convergence discussion, failure modes, exercises, and solutions.
- Keep exposition prerequisite-focused rather than interview-answer-focused.

---

### Task 1: Pricing foundations and Monte Carlo

**Files:**
- Create: `computational-finance/ch1-risk-neutral-pricing-lesson.md`
- Create: `computational-finance/ch2-monte-carlo-foundations-lesson.md`

- [ ] **Step 1:** Derive state prices, equivalent martingale measures, numeraires, discounted martingales, conditional pricing, and the real-world versus risk-neutral distinction.
- [ ] **Step 2:** Develop estimators, standard errors, confidence intervals, random generation, path discretization, weak versus strong error, convergence diagnostics, and reproducibility.
- [ ] **Step 3:** Validate Markdown and commit the two chapters.

### Task 2: Variance reduction, Greeks, and PDEs

**Files:**
- Create: `computational-finance/ch3-variance-reduction-greeks-lesson.md`
- Create: `computational-finance/ch4-black-scholes-pde-lesson.md`

- [ ] **Step 1:** Cover antithetic/control variates, importance and stratified sampling, quasi-Monte Carlo, pathwise and likelihood-ratio Greeks, common random numbers, and bias–variance trade-offs.
- [ ] **Step 2:** Derive the Black–Scholes PDE through delta hedging, connect terminal/boundary conditions and Feynman–Kac, and explain replication versus expectation viewpoints.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 3: Numerical PDEs and early exercise

**Files:**
- Create: `computational-finance/ch5-finite-differences-lesson.md`
- Create: `computational-finance/ch6-american-path-dependent-lesson.md`

- [ ] **Step 1:** Derive explicit, implicit, and Crank–Nicolson finite differences; explain grids, boundaries, stability, consistency, convergence, oscillations, and verification.
- [ ] **Step 2:** Cover optimal stopping, free boundaries, binomial trees, projected PDE methods, least-squares Monte Carlo, path dependence, and exercise-policy bias.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 4: Volatility models, calibration, and model risk

**Files:**
- Create: `computational-finance/ch7-volatility-calibration-model-risk-lesson.md`

- [ ] **Step 1:** Explain implied-volatility surfaces, static arbitrage, local and stochastic volatility, Heston intuition, calibration objectives, regularization, identifiability, hedging error, and model governance.
- [ ] **Step 2:** Validate all seven chapters and commit.

### Task 5: Course shell, integration, and publication

**Files:**
- Create: `computational-finance/index.html`
- Create: seven matching `computational-finance/*-viewer.html` pages
- Modify: `index.html`
- Modify: `README.md`

- [ ] **Step 1:** Add the course index, prerequisite links, seven chapter cards, and complete previous/next navigation using the shared viewer.
- [ ] **Step 2:** Add the completed course to the public site and recommended paths.
- [ ] **Step 3:** Run course, root-link, Node, unittest, and whitespace validation.
- [ ] **Step 4:** Inspect course index and representative early, middle, and final pages over local HTTP at desktop and narrow widths; verify MathJax, navigation, overflow, and browser logs.
- [ ] **Step 5:** Commit and push the verified course to `origin/master`.
