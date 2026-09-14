from __future__ import annotations

import pathlib

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_HPI_PATH = PROCESSED_DIR / "sofia_hpi.csv"
RAW_EUROSTAT_PATH = RAW_DIR / "eurostat_prc_hpi_q.csv"


def _eurostat_period_to_timestamp(period_str: str) -> pd.Timestamp:
    year, q = period_str.split("-Q")
    month = (int(q) - 1) * 3 + 1
    return pd.Timestamp(year=int(year), month=month, day=1)


def load_eurostat_hpi(
    geo: str = "BG",
    purchase: str = "TOTAL",
    unit: str = "I15_Q",
    force_refresh: bool = False,
) -> pd.Series:
    if not force_refresh and RAW_EUROSTAT_PATH.exists():
        raw_df = pd.read_csv(RAW_EUROSTAT_PATH, index_col=0)
    else:
        try:
            import eurostat

            raw_df = eurostat.get_data_df("prc_hpi_q")
            raw_df.to_csv(RAW_EUROSTAT_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch Eurostat prc_hpi_q. "
                f"Check internet connection. Original error: {exc}"
            ) from exc

    geo_col = [c for c in raw_df.columns if "geo" in c.lower() or "TIME_PERIOD" in c][0]

    mask = (
        (raw_df[geo_col].astype(str) == geo)
        & (raw_df["purchase"].astype(str) == purchase)
        & (raw_df["unit"].astype(str) == unit)
    )
    row = raw_df.loc[mask]
    if row.empty:
        raise ValueError(
            f"No data found for geo={geo}, purchase={purchase}, unit={unit}. "
            "Available combos: "
            + str(
                raw_df.loc[raw_df[geo_col].astype(str) == geo, ["purchase", "unit"]]
                .drop_duplicates()
                .to_dict()
            )
        )

    time_cols = [c for c in raw_df.columns if "Q" in str(c) and str(c)[0].isdigit()]
    vals = row[time_cols].iloc[0].dropna().astype(float)

    timestamps = [_eurostat_period_to_timestamp(p) for p in vals.index]
    series = pd.Series(
        vals.values,
        index=pd.DatetimeIndex(timestamps, freq="QS-OCT"),
        name="hpi",
        dtype=float,
    )
    series.index.name = "date"
    return series.sort_index()


def build_processed_dataset(force_refresh: bool = False) -> pd.DataFrame:
    if PROCESSED_HPI_PATH.exists() and not force_refresh:
        return pd.read_csv(PROCESSED_HPI_PATH, index_col="date", parse_dates=True)

    hpi = load_eurostat_hpi(force_refresh=force_refresh)
    df = pd.DataFrame({"hpi": hpi})
    df.index = pd.DatetimeIndex(df.index, freq="QS-OCT")
    df.index.name = "date"
    df = df.sort_index()
    df.to_csv(PROCESSED_HPI_PATH, index_label="date")
    return df


def load_processed_dataset() -> pd.DataFrame:
    if not PROCESSED_HPI_PATH.exists():
        return build_processed_dataset()
    return pd.read_csv(PROCESSED_HPI_PATH, index_col="date", parse_dates=True)
