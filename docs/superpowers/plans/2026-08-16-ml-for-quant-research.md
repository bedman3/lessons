# Machine Learning for Quant Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a seven-chapter course covering the full empirical quant-research lifecycle from an economic hypothesis and point-in-time data to realistic portfolio evaluation and production monitoring.

**Architecture:** Add `ml-quant/` as a canonical Markdown/viewer course. It bridges `applied-ml/` and the existing factor-model lesson, emphasizing temporal information boundaries, tradability, non-stationarity, and evidence rather than algorithm novelty.

**Tech Stack:** Markdown, MathJax 3, static HTML/CSS, shared ES-module viewer, Python/Node structural tests.

**Spec:** `docs/superpowers/specs/2026-08-16-advanced-curriculum-expansion-design.md`

## Global Constraints

- Target upper-undergraduate through practitioner or early-researcher depth.
- Every prediction is tied to a timestamp, horizon, universe, action, and cost model.
- Distinguish statistical prediction, portfolio construction, and realizable P&L.
- Cover point-in-time data, dependence-aware validation, multiple testing, and production parity.
- Include equations, examples, research diagnostics, failure modes, exercises, and solution outlines.

---

### Task 1: Research questions, targets, and data

**Files:**
- Create: `ml-quant/ch1-research-question-targets-lesson.md`
- Create: `ml-quant/ch2-point-in-time-data-lesson.md`

- [ ] **Step 1:** Cover hypothesis formation, estimands, prediction times, labels/horizons, cross-sectional versus time-series targets, universe definition, baselines, and research degrees of freedom.
- [ ] **Step 2:** Cover market/reference/alternative data, timestamps, availability versus event time, corporate actions, delistings, survivorship, revisions, joins, missingness, and point-in-time reconstruction.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 2: Validation and leakage control

**Files:**
- Create: `ml-quant/ch3-financial-validation-lesson.md`
- Create: `ml-quant/ch4-leakage-multiple-testing-lesson.md`

- [ ] **Step 1:** Develop rolling/expanding/walk-forward, cross-sectional splits, groups, overlapping labels, regime coverage, and nested temporal selection.
- [ ] **Step 2:** Develop look-ahead, purging, embargo, selection bias, data snooping, multiple comparisons, deflated performance, and untouched confirmation.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 3: Signals, portfolios, and robustness

**Files:**
- Create: `ml-quant/ch5-signal-portfolio-evaluation-lesson.md`
- Create: `ml-quant/ch6-nonstationarity-robustness-lesson.md`

- [ ] **Step 1:** Connect forecasts to IC/rank IC, calibration, neutralization, sizing, turnover, costs, capacity, attribution, and backtest accounting.
- [ ] **Step 2:** Cover drift, regimes, rolling estimation, shrinkage, ensembles, uncertainty, stress tests, sensitivity surfaces, and degradation evidence.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 4: Research-to-production

**Files:**
- Create: `ml-quant/ch7-research-production-lesson.md`

- [ ] **Step 1:** Cover reproducible research artefacts, offline/online parity, feature timing, monitoring, attribution, retraining, decay, incidents, governance, and kill criteria.
- [ ] **Step 2:** Validate all seven chapters and commit.

### Task 5: Shell, integration, and publication

**Files:**
- Create: `ml-quant/index.html`
- Create: seven matching `ml-quant/*-viewer.html` pages
- Modify: `index.html`
- Modify: `README.md`

- [ ] **Step 1:** Add course index and viewers with prerequisite and complete navigation links.
- [ ] **Step 2:** Integrate the completed course into the root site and recommended paths.
- [ ] **Step 3:** Run structural, root-link, renderer, unittest, and whitespace checks.
- [ ] **Step 4:** Inspect representative pages over HTTP at desktop and narrow widths; verify MathJax, navigation, overflow, and browser logs.
- [ ] **Step 5:** Commit and push to `origin/master`.
