from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


ModelFn = Callable[[pd.Series], Tuple[np.ndarray, Optional[np.ndarray]]]
"""
A model_fn takes a training series (pd.Series) and returns:
  (forecasts_array_shape_h, lower_ci_array_or_None)
where forecasts_array has length h (h=1 at index 0, h=2 at index 1, ...).
"""


def walk_forward_backtest(
    series: pd.Series,
    model_fn: ModelFn,
    n_test: int = 2,
    h: int = 2,
    min_train: int = 20,
) -> pd.DataFrame:
    n = len(series)
    records = []

    for o in range(n - n_test, n):
        train = series.iloc[:o]
        if len(train) < min_train:
            continue

        steps_available = n - o
        steps_to_forecast = min(h, steps_available)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecasts, _ = model_fn(train)
        except Exception as exc:
            warnings.warn(f"model_fn failed at origin {o}: {exc}")
            for step in range(1, steps_to_forecast + 1):
                records.append(
                    {
                        "origin_date": series.index[o - 1],
                        "horizon": step,
                        "actual": (
                            series.iloc[o + step - 1] if o + step - 1 < n else np.nan
                        ),
                        "forecast": np.nan,
                        "error": np.nan,
                        "abs_error": np.nan,
                        "pct_error": np.nan,
                    }
                )
            continue

        for step in range(1, steps_to_forecast + 1):
            actual_idx = o + step - 1
            if actual_idx >= n:
                break
            actual_val = series.iloc[actual_idx]
            if step - 1 < len(forecasts):
                try:
                    fc_val = float(forecasts.iloc[step - 1])
                except AttributeError:
                    fc_val = float(forecasts[step - 1])
            else:
                fc_val = np.nan

            error = fc_val - actual_val if not np.isnan(fc_val) else np.nan
            abs_err = abs(error) if not np.isnan(error) else np.nan
            pct_err = (
                (abs_err / actual_val * 100)
                if (not np.isnan(abs_err) and actual_val != 0)
                else np.nan
            )

            records.append(
                {
                    "origin_date": series.index[o - 1],
                    "horizon": step,
                    "actual": actual_val,
                    "forecast": fc_val,
                    "error": error,
                    "abs_error": abs_err,
                    "pct_error": pct_err,
                }
            )

    return pd.DataFrame(records)


def summarise_backtest(
    results: Dict[str, pd.DataFrame],
    horizons: List[int] = (1, 2),
) -> pd.DataFrame:
    rows = []
    for model_name, df in results.items():
        for h in horizons:
            sub = df[df["horizon"] == h].dropna(subset=["actual", "forecast"])
            if sub.empty:
                rows.append(
                    {
                        "model": model_name,
                        "horizon": h,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE": np.nan,
                    }
                )
                continue
            y_true = sub["actual"].values
            y_pred = sub["forecast"].values
            m = compute_metrics(y_true, y_pred)
            rows.append({"model": model_name, "horizon": h, **m})

    summary = pd.DataFrame(rows).set_index(["model", "horizon"])
    return summary
