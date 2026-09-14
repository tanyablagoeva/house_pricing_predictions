#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import build_processed_dataset
from src.evaluate import summarise_backtest, walk_forward_backtest

import pmdarima as pm

warnings.filterwarnings("ignore")
logging.getLogger("pmdarima").setLevel(logging.ERROR)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast"
DEFAULT_MODELS_DIR = ROOT / "models"
N_TEST = 2
H = 2
WINNER = "SARIMA"


def model_naive(train: pd.Series):
    last = float(train.iloc[-1])
    return np.array([last] * H), None


def model_seasonal_naive(train: pd.Series):
    same_q_last_yr = float(train.iloc[-4]) if len(train) >= 4 else float(train.iloc[-1])
    return np.array([same_q_last_yr] * H), None


def model_linear_drift(train: pd.Series):
    x = np.arange(len(train))
    slope, intercept = np.polyfit(x, train.values, 1)
    n = len(train)
    preds = np.array([slope * (n + i) + intercept for i in range(1, H + 1)])
    return preds, None


def model_sarima_fn(train: pd.Series):
    sarima = pm.auto_arima(
        train,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=1,
        D=1,
        seasonal=True,
        m=4,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    fc, conf = sarima.predict(n_periods=H, return_conf_int=True)
    return fc, conf[:, 0]


def retrain_sarima(
    hpi: pd.Series,
    models_dir: Path,
    save_model: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = pm.auto_arima(
        hpi,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=1,
        D=1,
        seasonal=True,
        m=4,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    fc, conf = m.predict(n_periods=H, return_conf_int=True)
    print(f"  SARIMA order: {m.order} seasonal: {m.seasonal_order}")

    if save_model:
        out = models_dir / "sarima_final.joblib"
        joblib.dump(m, out)
        print(f"  saved -> {out}")

    return np.asarray(fc), conf[:, 0], conf[:, 1]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 4),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_and_report(fig: plt.Figure, out_path: Path) -> None:
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_history(hpi: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        hpi.index,
        hpi.values,
        color="#1f77b4",
        linewidth=2,
        label="BG National HPI (2015=100)",
    )
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, label="Base level")
    ax.set_title(
        f"Bulgaria Quarterly HPI -- {hpi.index[0].year} Q1 to "
        f"{hpi.index[-1].year} Q{((hpi.index[-1].month - 1) // 3) + 1}"
    )
    ax.set_ylabel("HPI (2015=100)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    save_and_report(fig, out_path)


def plot_forecast(
    hpi: pd.Series,
    future_dates: pd.DatetimeIndex,
    fc_vals: np.ndarray,
    fc_lower: np.ndarray,
    fc_upper: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    cutoff = hpi.index[-1] - pd.DateOffset(years=5)
    hist = hpi[hpi.index >= cutoff]

    ax.plot(
        hist.index, hist.values, color="#1f77b4", linewidth=2, label="Historical HPI"
    )
    ax.plot(
        future_dates,
        fc_vals,
        "o--",
        color="#d62728",
        linewidth=2,
        markersize=8,
        label="SARIMA forecast",
    )
    ax.fill_between(
        future_dates,
        fc_lower,
        fc_upper,
        color="#d62728",
        alpha=0.15,
        label="95% Prediction Interval",
    )
    ax.axvline(hpi.index[-1], color="gray", linestyle=":", linewidth=1.2)

    for d, v in zip(future_dates, fc_vals):
        ax.annotate(
            f"{v:.1f}",
            (d, v),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#d62728",
        )

    ax.set_title(f"Sofia HPI Forecast -- Next {H} Quarters | Model: SARIMA")
    ax.set_ylabel("HPI (2015=100)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    save_and_report(fig, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forecast Bulgaria/Sofia HPI 2 quarters ahead (SARIMA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for charts, metrics CSV, and forecast JSON.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for the saved .joblib model.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write any files (charts/JSON/model).",
    )
    p.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download Eurostat data instead of using cache.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_plot_style()

    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        args.models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed dataset...")

    df = build_processed_dataset(force_refresh=args.refresh_data)
    hpi = df["hpi"].copy()
    hpi.index.name = "date"

    print(
        f"  shape={df.shape}, range={hpi.index[0].date()} -> {hpi.index[-1].date()}, n={len(hpi)}"
    )

    if not args.no_save:
        plot_history(hpi, args.output / "history.png")

    print("\nRunning baseline floor checks...")
    baselines: dict[str, Callable] = {
        "Naive": model_naive,
        "Seasonal Naive": model_seasonal_naive,
        "Linear Drift": model_linear_drift,
    }
    results: dict[str, pd.DataFrame] = {}

    for name, fn in baselines.items():
        results[name] = walk_forward_backtest(hpi, fn, n_test=N_TEST, h=H)
        print(f"  {name}: {len(results[name])} backtest rows")

    print("\nRunning SARIMA backtest...")
    results["SARIMA"] = walk_forward_backtest(hpi, model_sarima_fn, n_test=N_TEST, h=H)
    print(f"  SARIMA: {len(results['SARIMA'])} backtest rows")

    summary_table = summarise_backtest(results, horizons=[1, 2])
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(summary_table.round(3).to_string())

    seasonal_naive_h1 = (
        float(summary_table.loc[("Seasonal Naive", 1), "MAE"])
        if ("Seasonal Naive", 1) in summary_table.index
        else float("nan")
    )
    sarima_h1 = (
        float(summary_table.loc[("SARIMA", 1), "MAE"])
        if ("SARIMA", 1) in summary_table.index
        else float("nan")
    )

    print(f"\nSARIMA h=1 MAE:         {sarima_h1:.3f}")
    print(f"Seasonal Naive h=1 MAE: {seasonal_naive_h1:.3f}")
    if not np.isnan(sarima_h1) and not np.isnan(seasonal_naive_h1):
        print(f"Beats Seasonal Naive:   {sarima_h1 < seasonal_naive_h1}")

    print(
        f"\nRetraining SARIMA on {len(hpi)} observations and forecasting {H} quarters ahead..."
    )
    fc_vals, fc_lower, fc_upper = retrain_sarima(
        hpi, args.models_dir, save_model=not args.no_save
    )
    future_dates = pd.DatetimeIndex(
        [hpi.index[-1] + pd.DateOffset(months=3 * i) for i in range(1, H + 1)]
    )

    print("\nForecast:")
    for d, v, lo, hi in zip(future_dates, fc_vals, fc_lower, fc_upper):
        q = ((d.month - 1) // 3) + 1
        print(f"  {d.date()} (Q{q}-{d.year}): {v:.2f}  [95% PI: {lo:.2f}, {hi:.2f}]")

    if not args.no_save:
        print("\nSaving artifacts...")
        plot_forecast(
            hpi, future_dates, fc_vals, fc_lower, fc_upper, args.output / "forecast.png"
        )

        summary_table.round(4).to_csv(args.output / "backtest_metrics.csv")
        print(f"  saved -> {args.output / 'backtest_metrics.csv'}")

        forecast_record: dict[str, Any] = {
            "model": WINNER,
            "sarima_h1_mae": sarima_h1,
            "seasonal_naive_h1_mae": seasonal_naive_h1,
            "beats_seasonal_naive": (
                bool(sarima_h1 < seasonal_naive_h1)
                if not np.isnan(sarima_h1) and not np.isnan(seasonal_naive_h1)
                else None
            ),
            "last_observed_date": str(hpi.index[-1].date()),
            "last_observed_hpi": float(hpi.iloc[-1]),
            "horizon_quarters": H,
            "forecast": [
                {
                    "date": str(d.date()),
                    "quarter": f"Q{((d.month - 1) // 3) + 1}-{d.year}",
                    "hpi": float(v),
                    "lower_95": float(lo),
                    "upper_95": float(hi),
                }
                for d, v, lo, hi in zip(future_dates, fc_vals, fc_lower, fc_upper)
            ],
        }
        (args.output / "forecast.json").write_text(
            json.dumps(forecast_record, indent=2)
        )
        print(f"  saved -> {args.output / 'forecast.json'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
