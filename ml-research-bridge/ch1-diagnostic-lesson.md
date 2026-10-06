# Chapter 1 — Diagnostic: What Is the Experiment Actually Asking?

This bridge is for an experienced engineer who can build fast data and backtest systems but is new to hands-on model research. The first job is not to choose a neural network. It is to define a prediction task whose inputs, target, and evaluation match a decision that could have been made at the time.

A learning score is evidence about a specified experiment. It is not, by itself, evidence of an investable signal, trading profit, or readiness for a machine-learning role. All numbers in this chapter are synthetic teaching examples.

## 1.1 Write the prediction contract

Before fitting anything, write five items:

1. **Decision time:** when must the prediction exist?
2. **Information set:** which feature values were available by that time?
3. **Target:** what future quantity is being estimated, over what interval?
4. **Unit of observation:** one asset-date, one trade, one order, or another unit?
5. **Evaluation:** what baseline, metric, and cost assumptions answer the research question?

For example, a cross-sectional task might predict each stock’s next-session return using features available at the prior close. If the target is $y_{i,t}=r_{i,t+1}$, then $x_{i,t}$ must be computed from records known no later than the decision cutoff at $t$. A daily bar labeled by its end time may not be usable at that same timestamp if it is published later or corrected afterward. Record event time and availability time separately where the source supports both.

This contract also fixes the meaning of a row. If a date contains 500 stocks, that is 500 cross-sectional observations but only one shared market date. It is not equivalent to 500 independent market environments.

## 1.2 Inspect the target before the model

A target can be misaligned by one row, expressed in basis points in one table and decimal returns in another, or populated using a future-filled join. Check its units, missing rate, timestamp interval, tails, and representative rows by hand. Ask what happens around halts, IPOs, delistings, and missing quotes; exclusions change the population being studied.

Suppose the synthetic target values in basis points are $[-4,0,2,10]$. The mean is 2 bp, the median is 1 bp, and the sum is 8 bp. None of these is a trading return after costs. They are useful summaries for checking scale and target construction. If a target with a five-session horizon starts at date $t$ and ends at $t+5$, write down both timestamps: the label is not fully known at the start date.

## 1.3 Establish a baseline that can fail honestly

A model should beat a reference that represents a simple alternative. For zero-centered returns, $\hat y=0$ is a useful first baseline. For a prediction of the conditional mean, a train-period mean is another reference. A cross-sectional rank task may use a baseline computed from information available at prediction time, such as a historical training statistic or existing available score. Do not use the contemporaneous median of future target values as a live baseline. Choose from the decision contract, not after seeing which option makes the model look better.

Use multiple metrics because they answer different questions. Mean absolute error is

$$
\operatorname{MAE}=\frac1n\sum_i |y_i-\hat y_i|,
$$

and root mean squared error is

$$
\operatorname{RMSE}=\sqrt{\frac1n\sum_i (y_i-\hat y_i)^2}.
$$

MAE is easier to interpret in target units; RMSE penalizes large misses more. Correlation or rank correlation measures ordering, not calibration or profit. $R^2=1-\mathrm{SSE}/\mathrm{SST}$ compares squared error with a mean reference and can be negative out of sample.

### Worked example: do not hide the denominator

Let actual values be $[1,2,3]$ and predictions $[0,2,4]$. Absolute errors are $[1,0,1]$, so MAE is $2/3$. Squared errors are $[1,0,1]$, so RMSE is $\sqrt{2/3}\approx0.816$. The actual mean is 2 and total sum of squares is $(1-2)^2+(2-2)^2+(3-2)^2=2$. The model sum of squared errors is also 2, hence $R^2=0$. The model has nonzero correlation with the target, yet it has not beaten the mean predictor on squared error. Each metric tells a different part of the story.

## 1.4 Draw the first information boundary

For a first diagnostic, sort by decision date and divide dates into earlier development, later validation, and a final future block. Do not randomly split rows. Keep every asset from a date on the same side of a date boundary unless the deployment question explicitly requires another design. If labels overlap in time, the split needs a gap or explicit purging; Chapter 3 explains how to calculate it.

The original holdout is reserved for one confirmation after choices are frozen. Looking at its score and then changing a feature, model, threshold, or cost rule uses the holdout as validation. A new untouched future period would then be needed for a fresh final estimate.

## 1.5 Diagnostic checklist

Before running a model, be able to answer:

- Can every feature be reconstructed from data available by the decision timestamp?
- Does each target row point to the correct future interval and unit?
- Are feature and target units explicit and consistent?
- Which observations share a date, entity, or label interval?
- What simple baseline and metric correspond to the question?
- Which dates are development, validation, and frozen confirmation?

If any answer is “we assume,” log it as an unresolved assumption and design a check. A plausible score does not repair an ambiguous target.

## 1.6 Practice and answer key

**Exercise A.** You predict a 10-session return from features computed at the close. Name two timestamps that must be recorded and one reason a ten-row gap may be wrong in a panel.

**Exercise B.** A model’s MAE is 4 bp and the zero baseline’s MAE is 5 bp, but its rank correlation is lower. Is it better? What additional information do you need?

**Answer key.** A: Record the feature decision/availability cutoff and the target’s start/end interval. Ten rows may mean ten assets on one date rather than ten sessions; split by date and purge based on label intervals. B: It is better on absolute pointwise error in the evaluated sample, but worse at ordering; decide which loss matches the purpose, inspect uncertainty and date/regime breakdowns, and use the same universe and time boundaries for both.

**Checkpoint rubric (0–2 each):** timestamp contract explicit; target interval and units correct; baseline justified; split preserves chronology and panel-date boundaries; metrics interpreted accurately. Score 8/10 with no zero on timestamps or split before proceeding. This is a learning gate, not an employment qualification.

## 1.7 If the diagnostic fails

A constant prediction often means the target was centered or model inputs carry little usable variation; inspect distributions and missingness before adding complexity. Implausibly strong results often indicate timestamp leakage, duplicated target fields, incorrect joins, or split contamination. A negative $R^2$ is a result, not a software error. First reproduce a tiny hand calculation, then inspect row-level provenance and units.

The first project may take roughly 50–90 focused hours from diagnostic to a reproducible, reviewed experiment. Completing it demonstrates a project, not professional ML experience or qualification for roles that request years of production model work.

## 1.8 Continue selectively

For model capacity and regularization, see [Applied ML Chapter 2 — Linear Models & Regularization](../applied-ml/ch2-linear-models-regularization-lesson.md). For target and research-question design, see [ML Quant Chapter 1 — Research Questions & Targets](../ml-quant/ch1-research-question-targets-lesson.md). Next: [Chapter 2 — Baselines Before Complexity](ch2-baselines-lesson.md).
