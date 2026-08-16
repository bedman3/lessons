# Chapter 4 — Leakage, Purging & Multiple Testing

Financial datasets contain weak signal, strong dependence, and enormous researcher flexibility. A small information leak or a large search can create performance that looks economically plausible and statistically convincing.

Leakage controls the information set. Multiple-testing controls the search over hypotheses. Both are necessary.

## 4.1 Look-ahead leakage

Look-ahead occurs whenever a historical prediction uses information unavailable at its decision time. Examples include:

- revised fundamentals attached to fiscal period end;
- end-of-day index constituents used for an earlier intraday decision;
- full-sample normalization or PCA;
- future volatility used to scale past positions;
- using the final bar to both form a signal and assume an earlier fill;
- labels entering target encodings or feature selection.

Timestamp audits should trace raw event, vendor availability, feature calculation, model evaluation, and execution.

## 4.2 Outcome-interval leakage

Suppose observation $i$ uses features at $t_i$ and a label ending at $T_i$. Training observation $i$ overlaps validation observation $j$ if their information/outcome intervals intersect in a way that reveals part of the validation future.

**Purging** removes training examples whose label intervals overlap the validation block. It is based on actual intervals, not an arbitrary row count.

For fixed horizon $h$, a validation block beginning at $v$ typically requires removing training labels with $T_i\ge v$.

## 4.3 Embargo

An embargo excludes observations for a buffer around or after a validation block. It addresses dependence not captured by explicit label overlap: rolling features, delayed reactions, cross-asset propagation, or persistent positions.

Embargo length should follow the information mechanism and strategy horizon. A large embargo reduces contamination but also training data; it is not a substitute for understanding feature timing.

## 4.4 Leakage through portfolio construction

Even point-in-time predictions can be contaminated later:

- volatility or covariance estimated using future returns;
- neutralization against future classifications;
- selecting assets by future liquidity;
- optimizing thresholds on full-period P&L;
- applying transaction costs calibrated on future execution.

Every fitted component of the portfolio pipeline belongs inside the historical loop.

## 4.5 Multiple hypotheses

If $M$ true nulls are tested independently at level $\alpha$, expected false rejections are $M\alpha$. Research searches are correlated and adaptive, but the principle remains: the best result improves as more alternatives are tried even with no true signal.

The family includes features, horizons, transformations, universes, models, hyperparameters, execution delays, and performance metrics. Counting only the final reported p-value understates the search.

## 4.6 Family-wise error and false discovery rate

Bonferroni rejects only when $p_i\le\alpha/M$, controlling probability of any false rejection under broad dependence assumptions. It can be conservative.

Benjamini–Hochberg orders p-values $p_{(1)}\le\cdots\le p_{(M)}$ and finds the largest $k$ satisfying

$$
p_{(k)}\le\frac{k}{M}q.
$$

It controls expected false-discovery proportion under suitable assumptions. These methods require a meaningful declared family; hidden experiments remain uncorrected.

## 4.7 Selection-adjusted performance

An observed Sharpe ratio

$$
\widehat{SR}=\frac{\bar r}{s_r}
$$

has sampling uncertainty and selection bias. Non-normal returns, serial correlation, short history, and choosing the maximum across trials all matter.

Deflated-Sharpe-style reasoning compares observed performance with the expected maximum produced by the number and dependence of trials, then adjusts for skew, kurtosis, and sample length. The exact formula is less important than recording the effective search and return dependence.

## 4.8 Backtest overfitting

A backtest can be viewed as a flexible function fitted to one historical path. Repeated changes based on the same period reduce its status from test evidence to training evidence.

Controls include:

- a research log of all meaningful trials;
- nested walk-forward selection;
- simple mechanism-driven baselines;
- sensitivity surfaces rather than a single optimum;
- untouched confirmation periods or markets;
- reproducibility by an independent researcher;
- economic constraints defined before viewing results.

One final holdout can also be overused; once inspected and acted upon, it is consumed.

## 4.9 Leakage diagnostics

Useful adversarial checks:

- shift every feature backward and forward; suspicious improvement under forward shifts signals timing problems;
- train using deliberately future-only fields to understand upper bounds and detect accidental similarity;
- randomize labels within appropriate time/group blocks;
- compare full-history preprocessing with fold-local preprocessing;
- inspect performance around data revisions and vendor changes;
- remove the most suspicious feature groups;
- verify that predictions can be reproduced from an as-of snapshot alone.

Random-label tests must preserve relevant dependence or they can be too easy.

## 4.10 Economic plausibility is not proof

A compelling story can be invented after seeing results. Conversely, a real effect can lack a neat story. Mechanism is valuable because it guides priors, stress tests, decay expectations, and capacity analysis, but evidence still requires honest validation.

Separate **ex ante rationale** from **ex post interpretation** in documentation.

## 4.11 Worked boundary example

A feature at day $t$ predicts cumulative return from $t+1$ through $t+20$. Validation begins on day 1,000.

Training examples whose label end is day 1,000 or later share future returns with validation and must be purged. If a 60-day rolling feature and monthly portfolio holdings create additional persistence, add an embargo justified by those mechanisms.

Then perform feature/hyperparameter selection only in earlier inner windows. The outer block remains untouched until a complete configuration is fixed.

## 4.12 Failure modes

- Purging a fixed number of rows without using label intervals.
- Applying embargo while leaving full-sample preprocessing leakage.
- Correcting only the final set of models and ignoring discarded trials.
- Using iid p-values for serially correlated strategy returns.
- Treating the latest holdout as reusable after each modification.
- Believing a convincing economic narrative eliminates selection bias.

## 4.13 Knowledge checks

1. Distinguish look-ahead, overlap leakage, and selection bias.
2. What does purging remove?
3. What extra mechanism does embargo address?
4. Why is the number of reported strategies smaller than the relevant test family?
5. What evidence would increase confidence after a large search?

### Solution outlines

1. Future information in features; shared outcome intervals across splits; choosing winners from many trials.
2. Training samples whose label information overlaps the validation period.
3. Persistent dependence or rolling information beyond explicit label overlap.
4. Unreported features, parameters, horizons, and revisions also influenced selection.
5. Nested selection, recorded trials, untouched future/market confirmation, stable sensitivities, and independent reproduction.

## 4.14 What to retain

- Leakage can enter features, labels, splits, and portfolio construction.
- Purging follows outcome intervals; embargo follows residual information persistence.
- Research search size is part of statistical evidence.
- Holdouts are consumable resources.
- Timing audits and adversarial tests should precede performance interpretation.

Next: [Chapter 5 — Signal & Portfolio Evaluation](ch5-signal-portfolio-evaluation-viewer.html).
