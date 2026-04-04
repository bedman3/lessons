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
- 2 universe x 3 intervals x 4 weights x 5 product groups = ** X sample grids**

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
```

A rolling window defines
```