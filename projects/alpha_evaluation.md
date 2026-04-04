## Problem statement

To solve a weighted cross-sectional regression problem

> Given features **X** (trading signals like momentum, mean reversion) and targets **Y** (future returns), fit a linear model across stocks weighted by **W** (e.g. liquidity), then measure predictive power.

The challenge is **scalar**:
- ~5,000 stocks
- ~390 intraday timestamps per day (1-minute bars)
- ~252 trading days per year, multi-year lookback windows
- Multiple features, targets, universe, time intervals, weight schemes

handle this by:
1. **Sufficient statistics** - reduce each day's data to samll matrices (XTX, XTY)
2. **Monthly aggregation** - per-sum daily statistics into monthly batches
3. **Distributed computing** - each day/month/window runs as a slurm job
4. **Rolling windows** - re-fit periodically with configurable lookback and frequency

-- 

## 2. Data Dimensions

### The 3D Panel

All data starts as 3D xarray Datasets with dimensions `(product_id, endtime, date)`:

Every computation is indexed by a SampleGrid = (univ, time_interval, weight, product_group)
A pipeline run creates **Cartesian product** of all grid dimensions. For example:
- 2 universe x 3 intervals x 4 weights x 5 product groups = **X sample grids**

## 3. Phase 1: Daily Sufficient Statistics

**Goal:** For each date x sample_grid, reduce the full 3D panel into compact matrix summaries

### Step 3.1: Load and Filter Data

Load various panels, resample based on the time axis you have for the fitting setup (if 0930-1600 at 1min then generate axis for 390 endtimes), see each panel timestamp and only fill the next available timestamp if not aligned, otherwise make it at the specific timestamp (09:31:05 -> 09:32, 09:35:00 -> 09:35). Fill NaN as 0. Remember to reindex the panel to the specified universe in xarray.

### Step 3.2: Prepare X, Y, W
```
X: features after daily sample at 3.1
eval_Y: eval targets
fitting_Y: fitting targets, clip fitting Y at ± 0.2
W: weight used in fitting
```
Why clip Y? to prevent our fit dominate the regression, the clipped `_c` suffix version is used for. training only, the unclipped version is used for **evaluation** - ensuring metrics reflect real-world performance

Weight brocasting to match X shape


### Step 3.3: Flatten to 2D
Before matrix math, the 3D arrays are flattened to 2D:
```
X: (product_id x endtime x date, n_features) -> e.g., (4000 x 390 x 1, 800)
Y: (product_id x endtime x date, n_targets)
W: (product_id x endtime x date, 1)

do nan_to_num before we do actual calculation
```

### Step 3.4: The Weight Optimization (Critical)
**The problem:** Naively computing X^T W X requires inflating W to an N x N diagonal matrix, which for N = 1M+ is impossible.
**The solution:** Factor through W ^ {1/2}:
```
W_sqrt = sqrt(W)                    # shape: (N, 1) - stays 1D
X_weighted = X * W_sqrt             # shape: (N, p) - element-wise broadcast
tY_weighted = tY * W_sqrt           # shape: (N, q)
eY_weighted = eY * W_sqrt           # shape: (N, r)

XTX = X_weighted.T @ X_weighted     # shape: (p, p)
XTtY = X_weighted.T @ tY_weighted   # shape: (p, q)
eYTeY = eY_weighted.T @ eT_weighted # shape: (r, r)
W_sum = sum(W)                      # scalar
X_diag = diag(XTX)                  # shape: (p,)

```

**Mathematical proof:**
```
X^T W X = X^T (W^{1/2} W^{1/2}) X = (W^{1/2} X)^T (W^{1/2} X)
```

Since W is diagonal, W^{1/2} is also diagonal, and multiplying a diagonal by X is just element-wise scaling. No NxN matrix ever materializes.

```python
W_sqrt = np.sqrt(W_ar).reshape((-1, 1))
X_weighted = X * W_sqrt                     # (N, p) * (N, 1) -> broadcast -> (N, p)
tY_weighted = tY * W_sqrt
eY_weighted = eY * W_sqrt

XTX = X_weighted.T @ W_weighted         # (p, N) @ (N, p) -> (p, p)
XTtY = X_weighted.T @ tY_weighted       # (p, N) @ (N, q) -> (p, q)
eYTeY = eY_weighted.T @ eY_weighted     # (r, N) @ (N, r) -> (r, r)
X_diag = np.diag(XTX)                   # (p,)
W_sum = np.nansum(W_ar)                 # scalar
```

### Why Sufficient Statistics?
These 5 matrices are **sufficient** for Ridge regression - you can recover the exact same fitted coefficients from them as from the full raw data. This is because Ridge regression only needs XTX and XTY

```
beta = (XTX + lambda * I)^{-1} @ XTY
```

And these matrices are **additive across observations**: if you split data into groups A and B, then XYX_total = XTX_A + XTX_B. This enables the monthly aggregation in Phase 2.

## 4. Phase 2: Monthly Aggregatino
**Goal:** Sum daily sufficient statistics into monthly buckets for faster rolling-window construction.

### The Additivity Property

For any partition of observations into subsets (e.g., days within a month)
```
XTX_month = sum_over_days(XTX_day)
XTtY_month = sum_over_days(XTtY_day)
eYTeY_month = sum_over_days(eYTeY_day)
W_sum_month = sum_over_days(W_sum_day)
X_diag_month = sum_over_days(X_diag_day)
```

### Why Monthly?
A 2-year lookback window has ~504 trading days. Loading 504 daily files is slow. With monthly aggregation, you load ~24 monthyl files instead - a ~20x reduction in I/O.

**Trading date partitions** handle edge cases: if a month is incomplete (e.g., the pipeline starts mid-month), those individual days remain as daily cache entries and are loaded separately.

---

## 5. Phase 3: Rolling Window Fitting

**Goal:** For each rolling window x fitting group x fitter, solve for regression coefficients.

### Step 5.1: Rolling Window Construction


A rolling window defines:
- **Training period**: dates used to fit the model
- **Validation period**: dates used to evaluate predictions
```
train_start - TRAINING SET - train_end | 22 business day gap | val_start - VALIDATINO_SET - val_end
```

**The 22-business-day gap** prevents label leakage: if Y is "return over next 1 day", a model trained on data up to day T could have label information from day T+1. The 25-day buffer exceeds any typical prediction horizon.

**Exclude dates**: `fitting_exclude_dates` removes specific dates from training (e.g. COVID crash / black swarn). `evaluation_exclude_dates` removes dates from validation / reporting

### Step 5.2: Accumulate Training Statistics

```
IMPORTANT: Average over number of training days
Get XTX_sum / XTtY across all days in the period and divide by the valid sample size
```

**Why average?** The averaging normalizes the sufficient statistics so that the regularizatino parameter lambda has consistent meaning regardless of training set size. Without averaging, a 2-year window would need a different lambda than a 1-year window.

**Key detail:** Only the clipped Y targets (`_c` suffix) are used for training

### Step 5.3: Fitting Groups - Feature Selection
The system fits separate models per **fitting group**. Each group specifies it's own feaures and targets:

For each group, the code selects the relevant rows/columns from the accumulated XTX and XTtY:

### Step 5.4: Ridge Regression

**Mathematical formulation:**

Given:
- X^T W X (the averaged weighted Gram matrix, shape p x p)
- X^T W Y (the averaged weighted cross-product, shape p x q)
- lambda (regularization strength, default 0.1)

The Ridge solution is:
```
beta = (X^T W X + lambda * I)^{-1} @ X^T W Y
```

**With normalization** (default True)
```
D = diag(1/ sqrt(diag(X^T W X)))            # scaling matrix
M_norm = D @ (X^T W X) @ D                  # normalized so diag(M_norm) = 1
b_norm = D @ (X^T W Y)                      # normalized RHS

beta_norm = solve(M_norm + lambda * I, b_norm)
beta = D @ beta_norm                        # un-normalize

beta = np.linalg.solve(XTWX + lambda * np.eye(XYX.shape[0]), XTWY)
```

**Why normalize?** Without normalization, features with larger variance dominate the regularization. Normalizing ensures lambda penalizes all features equally, regardless of scale.

**Output:** `beta` has shape `(n_features, n_targets)` - one coefficient per feature per target.

### Step 5.5: Alternative Models:
**CONSTANT model**
- No fitting - coefficients are user-specified constants
- Useful for benchmarking or creating equal-weighted portfolios

**FF (Forward Feature Selection)**
- Produces `n_snap` snapshots, each addint the next best feature
- Coefficients at each snapshot are Ridge solutions on the selected subset


### Step 6.1: Prediction
For each validation date, the prediction alpha is:
```
alpha_i = X_i @ beta
```

where X_i is the feature vector for stock i at time t, and beta are the fitted coefficients
we don't need individiual predictions to get IC/SP - it works directly with sufficient statistics
```
alpha^T W alpha = beta^T @ (X^T W X) @ beta         <- "predicted signal variance"
alpha^T W Y     = beta^T @ (X^T W Y)                <- "predicted signal-return covariance"
```

This means IC and SP can be computed from XYX, XTY, and the fitted beta - without ever materializaing the full prediction vector.

### Step 6.2: IC (Information Coefficient)
**Definition:**
```
IC  = corr(alpha, Y, W)
    = sum(alpha * Y * W) / sqrt(sum(alpha^2 * W) * sum(Y^2 * W))
```

**In matrix form** (using sufficient statistics):
```
IC  = (X^T W Y) / sqrt(diag(X^T W X) * diag(Y^T W Y)) * 100
```

For the **term report** (no fitting, evaluating raw features):
- X^T W X is just the diagonal (`X_diag`)
- X^T W Y is the cross-product with eval targets

For the **fitting report** (evaluating predictions):
- Replace X with `alpha = X @ beta`
- `alpha^T W alpha = beta^T @ XTX @ beta` (scalar per prediction)
- `alpha^T W Y = beta^T @ XTY` (scalar per prediction x target)
```python
IC = XWY_vals / (np.sqrt(X_diag.reshape((-1, 1)) * np.diag(YWY_vals).reshape((1, -1)))) * 100
```

**Scale:** IC is reported as percentage (x 100). A daily IC of 2-5% is considered good.

### Step 6.3: SP (Single Predictive Power)
**Definitino:**
```
SP = sum(alpha * Y * W) / sqrt(sum(alpha^2 * W) * sum(W)) * 10000
```

**In matrix form:**
```
SP = (X^T W Y) / sqrt(diag(X^T W X) * W_sum) * 10000
```
```python
SP = (XWY_vals / (np.sqrt(X_diag.reshape((-1, 1)) * W_sum_vals))) * 10000
```

**Difference from IC:** SP normalized by total weight instead of Y variance. This means SP captures both predictive correlation AND the variabiility of Y. SP is more directly related to P&L.

**Scale:** SP is in basis points (x 10000).

### Step 6.4: sp (unnormalized predictor)
```
sp  = sum(alpha * Y * W) / sum(alpha^2 * W)
    = (X^T W Y) / diag(X^T W X)
```

This is the OLS coefficient of regressing Y on the prediction alpha - i.e., how much of Y does the prediction explain per unit of predictino variance.


### Step 6.5: SR (Sharpe Ratio)
```
SR  = mean(daily_metric) / std(daily_metric) * sqrt(252)
```

Annualized by sqrt(252) trading days per year

### Step 6.6: Daily vs Period Metrics

**Daily metrics:** IC and SP are computed per date using that day's sufficient statistics. This gives a time series: IC_day1, IC_day2, ...

**Period metrics:** For an eval period (e.g., 2023-01-01 to 2023-12-31), sum the sufficient statistics across all days in the period, then compute IC/SP form the summed statistics. This is the "overall" IC/SP.

### Step 6.7: Clipped Performance (Optional)
When a flag is turned on, predictions are winsorized at configurable quantiles (default: 1st and 99th percentile) before computing IC/SP. This measures performance after removing extreme outliers.

## 7. Phase 5: Term Statistics (No Fitting)

**Goal:** Measure raw feature predictive power without any model fitting.

### How It Differs from Fitting Report
In the fitting report, IC/SP measure the predictive power of **alpha = X @ beta** (the linear combination).

In the term report, IC/SP measure the predictive power of **each feature independently**. Since there's no beta, we use the diagonal of XTX:
```
IC_feature_j = XTY_j / sqrt(XTX_jj * YTY_jj) * 100
SP_feature_j = XTY_j / sqrt(XTX_jj * W_sum) * 10000
```

Where `XTX_jj = diag(X^T W X)[j]` and `XTY_j = (X^T W Y)[j, :]`

