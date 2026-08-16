# Chapter 3 — Model Families & Inductive Bias

Algorithms differ less by branding than by the structures they assume. Understanding an inductive bias—locality, smoothness, margins, partitions, or additive corrections—makes model choice more reliable than memorizing a leaderboard.

## 3.1 k-nearest neighbours: locality

For regression, k-nearest neighbours predicts

$$
\widehat f(x)=\frac1k\sum_{i\in N_k(x)}y_i.
$$

Small $k$ follows local detail but has high variance; large $k$ smooths more and increases bias. Distance depends on scale, so standardized and semantically meaningful features are essential.

In high dimensions, distances concentrate and neighbourhoods become sparse—the curse of dimensionality. Irrelevant features can overwhelm useful local structure.

## 3.2 Decision trees: adaptive partitioning

A tree recursively divides feature space and predicts a constant within each leaf. Regression splits often minimize squared error; classification splits reduce impurity such as Gini:

$$
G=1-\sum_kp_k^2.
$$

Trees capture nonlinearities and interactions without feature scaling. Their axis-aligned, greedy splits create discontinuous predictions and high sampling variance. Depth, minimum leaf size, pruning, and feature restrictions control complexity.

## 3.3 Bagging and random forests

Bagging fits unstable learners on bootstrap samples and averages predictions. If individual errors have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of an average of $M$ learners is approximately

$$
\rho\sigma^2+\frac{1-\rho}{M}\sigma^2.
$$

More trees reduce the second term, but the correlated component remains. Random forests decorrelate trees by considering random feature subsets at each split.

Out-of-bag observations provide a convenient internal diagnostic, though final evaluation still needs a deployment-representative split.

## 3.4 Boosting: sequential error correction

Gradient boosting builds an additive model

$$
F_M(x)=F_0(x)+\sum_{m=1}^M\eta h_m(x).
$$

Each weak learner approximates the negative gradient of loss with respect to current predictions. For squared loss, this means fitting residuals. Learning rate, number of trees, depth, row/feature subsampling, and regularization jointly determine capacity.

Boosting is powerful on structured tabular data but can exploit leakage and unstable identifiers as efficiently as genuine signal.

## 3.5 Maximum-margin classifiers

For labels $y_i\in\{-1,1\}$, a linear support vector machine seeks a large separating margin. The soft-margin objective is

$$
\min_{w,b}\frac12\|w\|^2+C\sum_i\max(0,1-y_i(w^Tx_i+b)).
$$

The norm controls geometric margin; hinge loss penalizes points inside it. Only support vectors directly determine the boundary.

SVM scores are not probabilities. Calibration is a separate estimation step.

## 3.6 Kernels

A kernel computes an inner product in an implicit feature space:

$$
K(x,x')=\langle\phi(x),\phi(x')\rangle.
$$

The radial basis kernel

$$
K(x,x')=\exp(-\gamma\|x-x'\|^2)
$$

encodes smooth local similarity. Large $\gamma$ creates highly local, flexible boundaries; small $\gamma$ creates broad smooth influence.

Kernel methods can be statistically elegant but scale poorly with sample size because training and prediction depend on many pairwise similarities.

## 3.7 Generative versus discriminative classification

A generative classifier models $P(X\mid Y)$ and $P(Y)$, then applies Bayes. A discriminative model targets $P(Y\mid X)$ or a decision boundary directly.

Naive Bayes can work well with limited data because its strong conditional-independence assumption reduces estimation variance. Logistic regression is less committed to the feature distribution but may need more data. Wrong assumptions can still yield good decisions if the relevant boundary is approximated well.

## 3.8 Algorithm comparison

| Family | Main bias | Strength | Common weakness |
|---|---|---|---|
| Linear/GLM | Additive linear score | Stable, interpretable baseline | Misses nonlinear structure |
| kNN | Local smoothness | Minimal fitting assumptions | Scaling and high dimensions |
| Tree | Axis-aligned partitions | Interactions, mixed scales | High variance, discontinuity |
| Random forest | Averaged decorrelated trees | Robust tabular baseline | Larger models, less extrapolation |
| Boosting | Sequential additive corrections | Strong tabular accuracy | Tuning and leakage sensitivity |
| Linear SVM | Large margin | Strong sparse/high-dimensional classifier | No native probabilities |
| Kernel method | Smooth similarity in feature space | Flexible with moderate data | Quadratic/cubic scaling |

## 3.9 Choosing a model

Start from constraints:

- target and loss;
- sample size versus dimension;
- temporal or grouped dependence;
- sparse, dense, categorical, or spatial inputs;
- latency, memory, update frequency;
- probability calibration and uncertainty needs;
- interpretability and governance.

Then compare a small set of meaningfully different inductive biases under one honest validation design. Tuning many nearly identical variants can overfit the validation process.

## 3.10 Worked reasoning example

For a medium-sized tabular credit dataset with nonlinear interactions and missingness:

1. begin with regularized logistic regression for calibration and interpretability;
2. compare gradient-boosted trees for nonlinear structure;
3. assess discrimination and calibration separately;
4. validate by application time and customer group;
5. inspect stability and subgroup errors.

kNN may struggle with mixed scales and dimensionality; an RBF SVM may be expensive and difficult to explain at this sample size. These are hypotheses to test, not absolute rules.

## 3.11 Failure modes

- Comparing tuned complex models with an untuned weak baseline.
- Treating tree feature importance as causal or stable.
- Scaling all data before kNN or SVM validation.
- Calling SVM margins probabilities.
- Increasing forest size to fix correlated bias.
- Letting boosted trees use post-outcome or identifier leakage.

## 3.12 Knowledge checks

1. Why does random feature selection help a random forest?
2. Explain boosting as functional gradient descent.
3. What happens to kNN as dimension grows with fixed sample size?
4. How do $C$ and margin trade off in a soft-margin SVM?
5. Why can naive Bayes work despite false independence assumptions?

### Solution outlines

1. It decorrelates tree errors, making averaging reduce more variance.
2. Each learner fits a direction that reduces loss in prediction-function space.
3. Local neighbourhoods become sparse and distances less discriminative.
4. Larger $C$ penalizes violations more, allowing a narrower, less regularized margin.
5. The simplified probability estimates may still induce a useful decision boundary with low estimation variance.

## 3.13 What to retain

- Model families encode different assumptions about useful structure.
- Ensembles work by managing bias, variance, and error correlation.
- Kernels and margins provide a geometric view of nonlinear classification.
- Model choice begins with data and decision constraints.
- Honest validation matters more than algorithm reputation.

Next: [Chapter 4 — Evaluation, Calibration & Decisions](ch4-evaluation-calibration-viewer.html).
