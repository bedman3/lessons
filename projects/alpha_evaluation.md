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
W_sqrt = sqrt(W)                # shape: (N, 1) - stays 1D
```