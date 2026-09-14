from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "imot_avg_prices.csv"

CATEGORICAL = ["district"]
NUMERIC = ["year", "month", "week_of_year", "quarter"]
TARGET = "eur_per_sqm_overall"
SEED = 42


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
    df = df.dropna(subset=[TARGET]).copy()
    df["month"] = df["snapshot_date"].dt.month
    df["week_of_year"] = df["snapshot_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["snapshot_date"].dt.quarter
    return df.sort_values("snapshot_date").reset_index(drop=True)


def time_split(
    df: pd.DataFrame, test_months: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["snapshot_date"].max() - pd.DateOffset(months=test_months)
    train = df[df["snapshot_date"] <= cutoff].copy()
    test = df[df["snapshot_date"] > cutoff].copy()
    return train, test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
