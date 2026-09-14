from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Sequence, Tuple


def build_lag_features(
    series: pd.Series,
    lags: Sequence[int] = (1, 2, 4),
    rolling_windows: Sequence[int] = (4,),
    include_quarter_dummies: bool = True,
    include_yoy: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame({"target": series})

    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    for w in rolling_windows:
        df[f"roll_mean_{w}"] = series.shift(1).rolling(w).mean()

    if include_yoy:
        df["yoy_pct"] = (series.shift(1) / series.shift(5) - 1) * 100

    if include_quarter_dummies:
        q = pd.PeriodIndex(series.index, freq="Q").quarter
        for qi in [1, 2, 3]:
            df[f"q{qi}"] = (q == qi).astype(int)

    df = df.dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "target"]


def recursive_ml_forecast(
    model,
    feature_df: pd.DataFrame,
    series: pd.Series,
    n_steps: int = 2,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    feat_cols = get_feature_columns(feature_df)
    max_lag = max(int(c.split("_")[1]) for c in feat_cols if c.startswith("lag_"))

    buffer_size = max(40, max_lag + 5)
    history = list(series.values[-buffer_size:])

    roll_cols = [c for c in feat_cols if c.startswith("roll_mean_")]
    roll_w = int(roll_cols[0].split("_")[-1]) if roll_cols else 4

    lag_cols = sorted(
        [c for c in feat_cols if c.startswith("lag_")],
        key=lambda c: int(c.split("_")[1]),
    )
    lags = [int(c.split("_")[1]) for c in lag_cols]

    q_cols = [c for c in feat_cols if c.startswith("q") and c[1:].isdigit()]
    has_yoy = "yoy_pct" in feat_cols

    last_date = series.index[-1]

    forecasts = []
    future_dates = []

    for step in range(1, n_steps + 1):
        row = {}

        for lag in lags:
            idx = -(lag)
            row[f"lag_{lag}"] = history[idx]

        for rw in [roll_w]:
            roll_vals = history[-rw:] if len(history) >= rw else history[:]
            row[f"roll_mean_{rw}"] = (
                float(np.mean(roll_vals)) if len(roll_vals) > 0 else np.nan
            )

        if has_yoy:
            if len(history) >= 5:
                row["yoy_pct"] = (history[-1] / history[-5] - 1) * 100
            else:
                row["yoy_pct"] = 0.0

        if q_cols:
            future_date = last_date + pd.DateOffset(months=3 * step)
            q_num = pd.PeriodIndex([future_date], freq="Q").quarter[0]
            for qi in [1, 2, 3]:
                row[f"q{qi}"] = int(q_num == qi)

        X = pd.DataFrame([row])[feat_cols]
        pred = float(model.predict(X)[0])
        forecasts.append(pred)
        history.append(pred)

        future_date = last_date + pd.DateOffset(months=3 * step)
        future_dates.append(future_date)

    return np.array(forecasts), pd.DatetimeIndex(future_dates)


def yoy_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change(4) * 100
