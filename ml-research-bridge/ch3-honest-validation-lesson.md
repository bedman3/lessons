# Chapter 3 — Honest Validation for Time-Ordered Data

A score is only as credible as the simulation that produced it. Market observations are ordered, repeated by asset, and often carry targets that mature after the feature timestamp. The split must preserve those facts. This chapter teaches a compact temporal design rather than a universal cross-validation recipe.

## 3.1 Separate three information roles

Use three chronological regions:

- **Development:** fit candidate models and debug the workflow.
- **Validation:** compare a prespecified, limited set of candidates and select a configuration.
- **Original holdout:** a later block opened once after code, features, model, metric, and cost assumptions are fixed.

The holdout is not a resource to query after each experiment. Once a result changes a decision, the holdout has influenced selection and is no longer untouched. A later block or genuinely external market/time period would then be required for a new final confirmation.

Within development and validation, use rolling or expanding windows that match the intended update cadence. An expanding window accumulates past data; a rolling window uses a fixed history. The choice is a model-design choice and should be set using validation, not optimized on the original holdout.

## 3.2 Split by time and define the label clock

For each observation, record a decision timestamp $t_i$ and the interval whose future outcome defines its label, such as $[t_i+1,t_i+5]$. The label is unavailable until its end. If validation begins at session 71 and the target uses the next five sessions, a training label starting at session 67 still reaches into validation. Exclude training labels whose outcome intervals overlap the validation period. This is **purging**.

A sample-based gap can separate fixed-horizon labels in one equally spaced series. It is not automatically correct for irregular observations or panels. With 500 instruments per date, a five-row gap may remove only part of one date. Split on unique dates and purge by label end. Add an embargo only when another information path—such as slowly updated rolling features—requires it.

`TimeSeriesSplit` in scikit-learn orders samples and supports a `gap` measured in samples. For a panel, this helper does not know which rows share a date or when labels mature, so a custom date-aware split and explicit interval checks may be necessary. A splitter can preserve order while encoding the wrong boundary.

## 3.3 Fit every learned transform inside the window

Feature scaling, imputation, clipping thresholds, PCA, feature selection, target encoding, and any learned neutralization are fitted state. At each split, fit them only on that split’s training rows. Apply the stored transformation to validation. Never compute a full-history mean or percentile and then claim later dates were held out.

A pipeline can keep transforms and model fitting together and reduce accidental leakage. It cannot verify raw timestamps, vendor revision history, overlapping labels, or market-specific logic. Manually inspect several examples at the boundary. “Feature timestamp equals decision timestamp” is not enough if the data arrives after the decision.

## 3.4 Worked split with dates and labels

Assume 120 consecutive synthetic sessions, one row per instrument-date, and a forward five-session target. Decide boundaries on dates: development through session 70, validation sessions 76–95, and a frozen original holdout sessions 101–120. Sessions 71–75 and 96–100 are buffers for illustration. Before each fit, remove development/validation training examples whose label end is at or after the next evaluation start. Fit scaling and feature selection on the remaining fit period only.

During development, declare a small candidate set: zero baseline, ridge with three penalty values, and one shallow tree configuration. Use rolling splits ending before session 70 to diagnose the workflow. Fit each candidate on eligible development rows and select using sessions 76–95 and one primary loss, perhaps mean per-date MAE; retain other metrics as diagnostics. Then freeze the complete pipeline, refit it on all eligible data through session 95, and evaluate sessions 101–120 once. Refit preprocessing on those eligible development and validation rows only after selection is frozen; the original holdout stays excluded. This simple design may still have too few independent periods; it makes information flow auditable.

## 3.5 Measure uncertainty at the level of dependence

If 400 stocks share a market date, their errors can move together because of one macro shock. A standard error that treats all 400 as independent can be far too small. First aggregate loss by date, such as

$$
L_t=\frac{1}{N_t}\sum_{i=1}^{N_t}|y_{i,t}-\hat y_{i,t}|,
$$

then compare the model and baseline on matched dates. Preserve time dependence when estimating uncertainty: resample contiguous date blocks or report rolling block results. Block width should reflect serial dependence and the target horizon, not a convenient setting chosen to make an interval narrow. Few independent blocks mean uncertainty remains large; no resampling method creates more market history.

For ranking, calculate rank correlation within date, then examine its distribution across dates and regimes. For thresholded actions, evaluate the same prespecified decision rule, universe, execution delay, and cost model. Report the number of dates and assets behind each summary, not just a pooled statistic.

## 3.6 Keep costs and selection in the record

Suppose a synthetic change improves gross score by 0.4 basis points per selected event, but the additional assumed spread and impact total 0.7 basis points. The illustrative net change is $0.4-0.7=-0.3$ bp. This is arithmetic about assumptions, not evidence of a strategy or realized P&L. In a real study, a cost model is itself uncertain and should be stress-tested against delay, liquidity, turnover, and capacity.

Research flexibility creates selection bias. Under an idealized independent-null example, testing 20 unrelated ideas at a 5% false-positive rate has probability

$$
1-(1-0.05)^{20}\approx 0.64
$$

of at least one false positive. Real experiments are correlated and adaptive, so 64% is not a universal estimate; it shows why a notebook containing only the winner is incomplete. Log meaningful trials: targets, features, dates, models, parameters, thresholds, cost assumptions, and metrics. Reusing an untouched test result to modify the model makes it another validation observation.

## 3.7 Practice and answer key

**Exercise A.** Dates 1–100 each contain 300 stocks. A label uses the next 10 sessions. Validation begins at date 81. Is `gap=10` on flattened rows an adequate purge?

**Exercise B.** You try 30 feature sets and report the best validation correlation. What should be added to the research record, and what role remains for the original holdout?

**Answer key.** A: No. Ten flattened rows may be only a fraction of a session. Split on date and exclude training examples whose label end overlaps date 81 or later; confirm the exact label convention. B: Record all 30 meaningful trials and selection decisions. Choose among them only inside development/validation. Keep the original later holdout closed until the workflow is fixed, then use it once as a final confirmation; if its result guides a new feature change, it is consumed.

**Checkpoint rubric (0–2 each):** label availability drives split; date units are explicit; transformations fit train-only; uncertainty respects time dependence; holdout and trials are logged. Require at least 8/10 and no zero for label overlap or holdout handling before describing a score as out of sample.

## 3.8 Continue selectively

For a longer treatment of financial split design, see [ML Quant Chapter 3 — Validation for Financial Data](../ml-quant/ch3-financial-validation-lesson.md). For overlap, embargo and multiple-testing controls, see [ML Quant Chapter 4 — Leakage, Purging & Multiple Testing](../ml-quant/ch4-leakage-multiple-testing-lesson.md). Next: [Chapter 4 — Training and Research Judgment](ch4-training-and-research-judgment-lesson.md).

## References

- scikit-learn, [TimeSeriesSplit and sample-based `gap`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- scikit-learn, [Pipelines and common leakage pitfalls](https://scikit-learn.org/stable/getting_started) and [Common pitfalls](https://scikit-learn.org/stable/1.7/common_pitfalls.html).
