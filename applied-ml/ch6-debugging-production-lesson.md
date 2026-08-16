# Chapter 6 — Debugging, Interpretation & Production

A model is complete only when its behaviour can be understood, reproduced, monitored, and connected to a decision. Production is not merely serving a serialized estimator; it is maintaining the assumptions under which evaluation was meaningful.

## 6.1 Debug in layers

When performance disappoints, inspect layers in order:

1. **Data:** labels, units, timestamps, joins, duplicates, missingness.
2. **Task:** target definition, horizon, loss, and baseline.
3. **Pipeline:** transformations, split boundaries, training/serving parity.
4. **Optimization:** convergence, learning curves, gradients, numerical scale.
5. **Generalization:** variance, shift, subgroup errors, and calibration.

Complex model changes should come after cheap data and task checks.

## 6.2 Error analysis

Aggregate metrics hide structure. Slice errors by time, subgroup, target magnitude, confidence, missingness pattern, and operational route. Examine both systematic clusters and individual high-loss cases.

An error slice generates a hypothesis, not proof. Small slices are noisy, and searching many slices creates multiplicity.

## 6.3 Ablations and controlled comparisons

An ablation removes one component while holding the rest fixed. Useful comparisons include:

- feature group removed;
- model family changed with the same validation;
- preprocessing step removed;
- data source or period excluded;
- latency or memory constraint applied.

Change one variable at a time, preserve seeds and split identities, and report uncertainty. A leaderboard of incomparable experiments cannot identify causes.

## 6.4 Interpretation tools

Global coefficients, permutation importance, partial dependence, accumulated local effects, and local attribution methods answer different questions.

Permutation importance measures performance loss when a feature is disrupted. Correlated alternatives can hide importance. Partial dependence averages predictions over altered feature values, potentially creating unrealistic combinations. Local attributions explain a model output relative to a baseline, not the data-generating cause.

Interpretation should be validated for stability and used with domain knowledge, not treated as causal discovery.

## 6.5 Reproducibility

A reproducible run records:

- immutable data snapshot or version;
- label and feature definitions;
- code revision and environment;
- split identifiers;
- configuration and random seeds;
- fitted artefact and evaluation output.

Seed control does not guarantee determinism across hardware or libraries, but unexplained variability should be measured rather than ignored.

## 6.6 Training–serving parity

Offline features and online features must share definitions, clocks, defaults, and transformations. Common failures include timezone shifts, different category handling, stale caches, unavailable joins, and silent schema coercion.

Shadow deployment compares predictions without acting on them. Canary rollout exposes a small fraction of traffic. Both reduce risk but require clear rollback criteria.

## 6.7 Monitoring

Monitor several layers:

- service health: latency, failures, resource use;
- input health: schema, missingness, ranges, categories;
- prediction health: score distribution, confidence, calibration proxies;
- outcome health: delayed labels, loss, subgroup performance;
- decision health: intervention rate, utility, costs, feedback effects.

Drift alerts should lead to a defined diagnosis and response. Retraining automatically on corrupted data can amplify failure.

## 6.8 Feedback loops

Predictions alter the data later observed. A fraud model blocks cases, so labels are observed selectively. A recommender changes exposure. A trading strategy changes market impact and opportunity.

This creates policy-dependent data. Evaluation may require randomized exploration, causal methods, inverse propensity weighting, or explicit simulation—not merely another random split.

## 6.9 Latency, cost, and model value

A slightly more accurate model may be worse if it is too slow, expensive, unstable, or hard to update. Evaluate an end-to-end objective such as

$$
\text{net value}
=\text{decision benefit}-\text{errors}-\text{compute}-\text{latency}-\text{operational risk}.
$$

Distillation, feature caching, batching, quantization, or a smaller model can improve system value even if offline loss rises slightly.

## 6.10 Governance and uncertainty

Document intended use, excluded uses, data limitations, validation population, known failure modes, and ownership. High-impact decisions require escalation paths and human review that is meaningful rather than ceremonial.

Uncertainty comes from data noise, parameter estimation, distribution shift, and model-form uncertainty. A single probability score rarely captures all four.

## 6.11 End-to-end checklist

Before release:

1. target and action are explicit;
2. baseline and validation reflect deployment;
3. leakage checks cover timestamps and entities;
4. metrics include calibration, uncertainty, and key slices;
5. transformations match online implementation;
6. monitoring has owners and response thresholds;
7. rollback is tested;
8. documentation states limitations.

## 6.12 Failure modes

- Treating explanation tools as causal proof.
- Monitoring input drift without labelled outcome performance.
- Retraining on schedule without diagnosing change.
- Allowing online and offline feature code to diverge.
- Declaring reproducibility from a seed alone.
- Optimizing model accuracy while ignoring action cost and feedback.

## 6.13 Knowledge checks

1. Why should data checks precede model tuning in debugging?
2. What does permutation importance measure, and how can correlation distort it?
3. Distinguish covariate drift from performance drift.
4. Why can deployment create selective labels?
5. Name evidence needed to reproduce a model result.

### Solution outlines

1. Incorrect labels, joins, or timing can dominate every later result and cannot be repaired by algorithms.
2. Loss increase after feature disruption; correlated substitutes can make the increase misleadingly small.
3. Inputs changed versus the model's target loss changed; one need not imply the other.
4. Actions based on predictions determine which outcomes become observable.
5. Data version, code/environment, configuration, splits, seed, and evaluation artefacts.

## 6.14 Course synthesis

Applied ML is a chain of assumptions:

1. define a decision and loss;
2. choose an inductive bias;
3. estimate it from finite data;
4. validate the complete pipeline under realistic dependence;
5. translate scores into actions;
6. monitor the data–model–decision loop after deployment.

Return to the [Applied Machine Learning contents](index.html).
