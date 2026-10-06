"""Small, auditable chronological ML experiment for the course.

All observations are synthetic. Scores are prediction diagnostics, not trading
performance. Run with ``python lab.py --output results.json``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Samples:
    features: np.ndarray
    target: np.ndarray
    decision_time: np.ndarray
    label_end: np.ndarray


@dataclass(frozen=True)
class Split:
    train: Samples
    validation: Samples
    test: Samples


@dataclass(frozen=True)
class ScaledSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def make_synthetic_data(seed: int = 19, n_samples: int = 600, horizon: int = 5) -> Samples:
    """Create a noisy, nonstationary synthetic forecasting problem.

    ``target[i]`` is a generated outcome observed ``horizon`` steps after
    decision ``i``; it is not a sum of simulated returns across that horizon.
    The delayed label end exists to demonstrate availability-aware purging.
    """
    if n_samples <= 20 or horizon < 1 or horizon >= n_samples:
        raise ValueError("require n_samples > 20 and 1 <= horizon < n_samples")
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    regime = (t >= int(.62 * n_samples)).astype(float)
    x = rng.normal(size=(n_samples, 5))
    # Feature scale shifts in the later regime; target relationship also changes.
    x[:, 0] += .7 * regime
    x[:, 1] *= 1 + .8 * regime
    noise = rng.normal(scale=.8 + .25 * regime, size=n_samples)
    y = (0.8 * x[:, 0] - .5 * x[:, 1] + .4 * x[:, 2] * x[:, 3] + noise)
    y += regime * (-.45 * x[:, 0] + .65 * x[:, 1])
    return Samples(x, y, t, t + horizon)


def _take(s: Samples, mask: np.ndarray) -> Samples:
    return Samples(s.features[mask], s.target[mask], s.decision_time[mask], s.label_end[mask])


def chronological_split(data: Samples, train_end: int, validation_end: int, horizon: int = 5) -> Split:
    """Split by decision time; purge labels that reach the next decision boundary."""
    if not 0 < train_end < validation_end < len(data.target):
        raise ValueError("require 0 < train_end < validation_end < sample count")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    tr = (data.decision_time < train_end) & (data.label_end < train_end)
    va = (data.decision_time >= train_end) & (data.decision_time < validation_end) & (data.label_end < validation_end)
    te = data.decision_time >= validation_end
    if not tr.any() or not va.any() or not te.any():
        raise ValueError("split produced an empty partition")
    return Split(_take(data, tr), _take(data, va), _take(data, te))


def scale_with_train_only(split: Split) -> ScaledSplit:
    scaler = StandardScaler().fit(split.train.features)
    return ScaledSplit(scaler.transform(split.train.features), scaler.transform(split.validation.features),
                       scaler.transform(split.test.features), scaler.mean_.copy(), scaler.scale_.copy())


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** .5),
            "r2": float(r2_score(y, pred))}


def run_experiment(seed: int = 19, n_samples: int = 600) -> dict:
    horizon = 5
    train_end, validation_end = int(.55 * n_samples), int(.78 * n_samples)
    data = make_synthetic_data(seed, n_samples, horizon)
    split = chronological_split(data, train_end, validation_end, horizon)
    scaled = scale_with_train_only(split)
    # Small, declared candidate set. Validation MAE chooses one configuration.
    candidates = [
        ("zero", None),
        ("ridge_alpha_1", Ridge(alpha=1.0)),
        ("ridge_alpha_10", Ridge(alpha=10.0)),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7,
                                                                  l2_regularization=2.0,
                                                                  random_state=seed)),
    ]
    validation_results = []
    for name, model in candidates:
        if name == "zero":
            pred = np.zeros_like(split.validation.target)
        else:
            model.fit(scaled.train, split.train.target)
            pred = model.predict(scaled.validation)
        validation_results.append({"name": name, **_metrics(split.validation.target, pred)})
    selected = min(validation_results, key=lambda row: row["mae"])["name"]

    # Freeze choice, refit scaler and model on all labels matured by test start.
    development = _take(data, data.decision_time < validation_end)
    final_train_mask = development.label_end < validation_end
    final_train = _take(development, final_train_mask)
    test = split.test
    final_scaler = StandardScaler().fit(final_train.features)
    xt, xe = final_scaler.transform(final_train.features), final_scaler.transform(test.features)
    start = time.perf_counter()
    if selected == "zero":
        test_pred = np.zeros_like(test.target)
    else:
        final_model = next(model for name, model in candidates if name == selected)
        final_model.fit(xt, final_train.target)
        test_pred = final_model.predict(xe)
    fit_seconds = time.perf_counter() - start
    # A deliberately invalid future target column demonstrates leakage only.
    # It is never included in candidates or selection.
    future_target_feature = test.target.copy()
    leakage_pred = future_target_feature
    return {
        "config": {"seed": seed, "n_samples": n_samples, "horizon": horizon,
                   "train_end_decision": train_end, "validation_end_decision": validation_end,
                   "selection_metric": "validation_mae", "candidate_count": len(candidates)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "split_sizes": {"train": len(split.train.target), "validation": len(split.validation.target), "test": len(test.target)},
        "boundaries": {"train_max_label_end": int(split.train.label_end.max()),
                       "validation_first_decision": int(split.validation.decision_time.min()),
                       "validation_max_label_end": int(split.validation.label_end.max()),
                       "test_first_decision": int(test.decision_time.min()),
                       "selection_scaler_fit_rows": len(split.train.target),
                       "final_fit_rows": len(final_train.target),
                       "final_max_label_end": int(final_train.label_end.max())},
        "validation_candidates": validation_results, "selected_model": selected,
        "test_metrics": _metrics(test.target, test_pred), "test_evaluation_count": 1,
        "final_fit_and_predict_seconds": fit_seconds,
        "leakage_demo": {"description": "invalid feature equal to future test target; control only",
                         "metrics": _metrics(test.target, leakage_pred), "used_for_selection": False},
        "interpretation": "Synthetic prediction exercise only; no alpha, P&L, costs, or investment claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.seed, args.n_samples)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
