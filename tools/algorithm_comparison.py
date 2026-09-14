from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import (
    CATEGORICAL,
    NUMERIC,
    SEED,
    TARGET,
    build_preprocessor,
    load_dataset,
    time_split,
)

OUT_DIR = ROOT / "outputs" / "algorithm_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def model_registry() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
        "Lasso Regression": Lasso(alpha=1.0, random_state=SEED, max_iter=10_000),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1, max_depth=18
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED
        ),
        "MLP (backprop)": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=SEED,
            early_stopping=True,
            n_iter_no_change=15,
        ),
        "KNN (k=5, distance-weighted)": KNeighborsRegressor(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
    }


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def evaluate_full(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    X_train, y_train = train[CATEGORICAL + NUMERIC], train[TARGET].to_numpy()
    X_test, y_test = test[CATEGORICAL + NUMERIC], test[TARGET].to_numpy()

    rows = []
    for name, estimator in model_registry().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        y_pred = pipe.predict(X_test)
        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape_val = mape(y_test, y_pred)
        r2_val = r2_score(y_test, y_pred)

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        tmp_path = (
            OUT_DIR
            / f"_tmp_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        )
        joblib.dump(pipe, tmp_path)
        size_kb = tmp_path.stat().st_size / 1024.0
        tmp_path.unlink()

        rows.append(
            {
                "model": name,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "MAPE": mape_val,
                "R2": r2_val,
                "train_time_s": train_time,
                "CV_MAE_mean": cv_mae,
                "CV_MAE_std": cv_std,
                "model_size_kb": size_kb,
            }
        )

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def evaluate_baselines(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    pred_overall = np.full_like(y_test, y_train.mean(), dtype=float)

    district_mean = train.groupby("district")[TARGET].mean()
    fallback = y_train.mean()
    pred_district = test["district"].map(district_mean).fillna(fallback).to_numpy()

    last_known = train.sort_values("snapshot_date").groupby("district")[TARGET].last()
    pred_last = test["district"].map(last_known).fillna(fallback).to_numpy()

    rows = []
    for name, pred in [
        ("Baseline: Overall mean", pred_overall),
        ("Baseline: District mean", pred_district),
        ("Baseline: District last value", pred_last),
    ]:
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "MAPE": mape(y_test, pred),
                "R2": r2_score(y_test, pred),
                "train_time_s": 0.0,
                "CV_MAE_mean": np.nan,
                "CV_MAE_std": np.nan,
                "model_size_kb": 0.0,
            }
        )
    return pd.DataFrame(rows)


def learning_curve(
    train: pd.DataFrame, test: pd.DataFrame, sample_sizes: list[int]
) -> pd.DataFrame:
    train_sorted = train.sort_values("snapshot_date")
    X_test = test[CATEGORICAL + NUMERIC]
    y_test = test[TARGET].to_numpy()

    rows = []
    for n in sample_sizes:
        sub = train_sorted.tail(n) if n < len(train_sorted) else train_sorted
        X_sub = sub[CATEGORICAL + NUMERIC]
        y_sub = sub[TARGET].to_numpy()
        for name, estimator in model_registry().items():
            pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
            try:
                pipe.fit(X_sub, y_sub)
                mae_val = mean_absolute_error(y_test, pipe.predict(X_test))
            except Exception as exc:
                print(f"  ! {name} @ N={n} failed: {exc}")
                mae_val = np.nan
            rows.append({"N": len(sub), "model": name, "MAE": mae_val})
            print(f"  N={len(sub):>6d}  {name:<22s}  MAE={mae_val:8.2f}")
    return pd.DataFrame(rows)


def plot_comparison(metrics: pd.DataFrame, out_path: Path) -> None:
    model_only = metrics[~metrics["model"].str.startswith("Baseline")].copy()
    model_only = model_only.sort_values("MAE")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].barh(model_only["model"], model_only["MAE"], color="steelblue")
    axes[0].set_xlabel("Test MAE (EUR/m^2)")
    axes[0].set_title("Test MAE (lower is better)")
    axes[0].invert_yaxis()

    axes[1].barh(model_only["model"], model_only["R2"], color="seagreen")
    axes[1].set_xlabel("Test R^2")
    axes[1].set_title("Test R^2 (higher is better)")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)

    axes[2].barh(model_only["model"], model_only["train_time_s"], color="indianred")
    axes[2].set_xlabel("Train time (s)")
    axes[2].set_title("Training time")
    axes[2].invert_yaxis()
    axes[2].set_xscale("log")

    fig.suptitle("8-Algorithm Comparison — Sofia district-week EUR/m^2", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(lc: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, group in lc.groupby("model"):
        group_sorted = group.sort_values("N")
        ax1.plot(group_sorted["N"], group_sorted["MAE"], marker="o", label=name)
        ax2.plot(group_sorted["N"], group_sorted["MAE"], marker="o", label=name)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Training sample size (log scale)")
        ax.set_ylabel("Test MAE (EUR/m^2)")
        ax.grid(True, alpha=0.3)

    ax1.set_yscale("log")
    ax1.set_title("Log scale (all algorithms)")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.set_ylim(150, 700)
    ax2.set_title("Linear scale, zoomed (150-700 EUR/m^2)")

    fig.suptitle(
        "Learning curves — does the algorithm ranking hold at small N?", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Loading data...")
    df = load_dataset()
    train, test = time_split(df, test_months=12)
    print(f"  total rows: {len(df):,}")
    print(
        f"  train: {len(train):,} rows ({train.snapshot_date.min().date()} -> {train.snapshot_date.max().date()})"
    )
    print(
        f"  test:  {len(test):,} rows ({test.snapshot_date.min().date()} -> {test.snapshot_date.max().date()})"
    )
    print(
        f"  districts (train): {train.district.nunique()}, (test): {test.district.nunique()}"
    )

    print("\nEvaluating baselines...")
    baselines = evaluate_baselines(train, test)
    print(baselines[["model", "MAE", "RMSE", "MAPE", "R2"]].to_string(index=False))

    print("\nEvaluating 8 algorithms on full training set...")
    model_results = evaluate_full(train, test)
    print(
        model_results[
            ["model", "MAE", "RMSE", "MAPE", "R2", "CV_MAE_mean", "CV_MAE_std"]
        ].to_string(index=False)
    )

    combined = (
        pd.concat([model_results, baselines], ignore_index=True)
        .sort_values("MAE")
        .reset_index(drop=True)
    )
    combined.to_csv(OUT_DIR / "comparison_metrics.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'comparison_metrics.csv'}")

    plot_comparison(combined, OUT_DIR / "comparison_plot.png")
    print(f"Wrote {OUT_DIR / 'comparison_plot.png'}")

    print("\nRunning learning curve (this is the slow part)...")
    sample_sizes = [200, 500, 1000, 2500, 5000, 10000, 25000, len(train)]
    lc = learning_curve(train, test, sample_sizes)
    lc.to_csv(OUT_DIR / "learning_curves.csv", index=False)
    plot_learning_curves(lc, OUT_DIR / "learning_curves.png")
    print(f"Wrote {OUT_DIR / 'learning_curves.csv'}")
    print(f"Wrote {OUT_DIR / 'learning_curves.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
