from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

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

TUNING_JSON = ROOT / "outputs" / "tuning" / "full_run.json"
MLP_MODEL_PATH = ROOT / "models" / "per_district_final.joblib"
SARIMA_HIERARCHICAL_PATH = (
    ROOT / "outputs" / "hierarchical_forecast" / "district_forecasts.csv"
)
OUTPUT_PATH = ROOT / "outputs" / "predictions" / "per_district_forecasts.csv"

QUARTER_DATES = {
    "Q2-2026": pd.Timestamp("2026-05-15"),
    "Q3-2026": pd.Timestamp("2026-08-15"),
    "Q4-2026": pd.Timestamp("2026-11-15"),
}


def quarter_col(label: str) -> str:
    return f"forecast_{label.lower().replace('-', '_')}"


def load_tuned_params() -> dict[str, dict]:
    payload = json.loads(TUNING_JSON.read_text())
    out: dict[str, dict] = {}
    for entry in payload["results"]:
        out[entry["model"]] = entry["best_params"]
    return out


def build_estimator(model_name: str, params: dict) -> object:
    if model_name == "LightGBM":
        return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, **params
        )
    if model_name == "Gradient Boosting (sklearn)":
        return GradientBoostingRegressor(random_state=SEED, **params)
    if model_name == "KNN":
        return KNeighborsRegressor(n_jobs=-1, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def fit_one(
    model_name: str, params: dict, X_train: pd.DataFrame, y_train: np.ndarray
) -> Pipeline:
    pipe = Pipeline(
        [("prep", build_preprocessor()), ("model", build_estimator(model_name, params))]
    )
    pipe.fit(X_train, y_train)
    return pipe


def build_prediction_grid(districts: list[str]) -> pd.DataFrame:
    rows = []
    for district in districts:
        for quarter_label, ts in QUARTER_DATES.items():
            iso_week = int(pd.Timestamp(ts).isocalendar().week)
            rows.append(
                {
                    "district": district,
                    "quarter": quarter_label,
                    "year": ts.year,
                    "month": ts.month,
                    "week_of_year": iso_week,
                    "quarter_num": (ts.month - 1) // 3 + 1,
                }
            )
    grid = pd.DataFrame(rows)
    grid = grid.rename(columns={"quarter_num": "quarter_feat"})
    return grid


def main() -> int:
    print("Loading data...")
    df = load_dataset()
    train, _ = time_split(df, test_months=12)
    X_train, y_train = train[CATEGORICAL + NUMERIC], train[TARGET].to_numpy()
    print(f"  train rows: {len(train):,}")

    print("Loading SARIMA hierarchical forecasts...")
    sarima_df = pd.read_csv(SARIMA_HIERARCHICAL_PATH)
    districts = sarima_df["district"].tolist()
    print(f"  districts with SARIMA baseline: {len(districts)}")

    print("Loading tuned hyperparameters...")
    tuned = load_tuned_params()
    print(f"  models tuned: {list(tuned.keys())}")

    grid = build_prediction_grid(districts)
    X_pred = pd.DataFrame(
        {
            "district": grid["district"],
            "year": grid["year"],
            "month": grid["month"],
            "week_of_year": grid["week_of_year"],
            "quarter": grid["quarter_feat"],
        }
    )

    print("Loading tuned MLP from joblib...")
    mlp_pipe = joblib.load(MLP_MODEL_PATH)
    grid["MLP"] = mlp_pipe.predict(X_pred)

    for display_name, column in [
        ("LightGBM", "LightGBM"),
        ("XGBoost", "XGBoost"),
        ("Gradient Boosting (sklearn)", "GBM"),
        ("KNN", "KNN"),
    ]:
        print(f"Refitting {display_name} on train set...")
        pipe = fit_one(display_name, tuned[display_name], X_train, y_train)
        grid[column] = pipe.predict(X_pred)

    print("Joining SARIMA hierarchical forecasts...")
    sarima_value_cols = [quarter_col(q) for q in QUARTER_DATES.keys()]
    sarima_long = sarima_df.melt(
        id_vars=["district"],
        value_vars=sarima_value_cols,
        var_name="quarter",
        value_name="SARIMA_hierarchical",
    )
    sarima_long["quarter"] = sarima_long["quarter"].map(
        {quarter_col(q): q for q in QUARTER_DATES.keys()}
    )
    out = grid.merge(sarima_long, on=["district", "quarter"], how="left")

    columns = [
        "district",
        "quarter",
        "MLP",
        "LightGBM",
        "XGBoost",
        "GBM",
        "KNN",
        "SARIMA_hierarchical",
    ]
    out = out[columns].round(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"  rows: {len(out)} ({len(districts)} districts x {len(QUARTER_DATES)} quarters)"
    )
    print("\nSample (first 5 rows):")
    print(out.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
