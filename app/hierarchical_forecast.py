from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HPI_CSV = ROOT / "data" / "processed" / "sofia_hpi.csv"
IMOT_CSV = ROOT / "data" / "raw" / "imot_avg_prices.csv"
OUT_DIR = ROOT / "outputs" / "hierarchical_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target horizon. The values are produced by the saved SARIMA model at runtime,
# not hardcoded. See load_hpi_forecasts().
FORECAST_QUARTERS = ["Q2-2026", "Q3-2026", "Q4-2026"]
BASELINE_QUARTER = "2025Q4"


def _quarter_key(label: str) -> str:
    return label.lower().replace("-", "_")


def load_hpi_baseline() -> float:
    df = pd.read_csv(HPI_CSV, parse_dates=["date"])
    q4_row = df[df["date"] == "2025-10-01"]
    if q4_row.empty:
        raise RuntimeError(
            "HPI Q4-2025 (2025-10-01) not found in processed HPI series."
        )
    return float(q4_row["hpi"].iloc[0])


def load_hpi_forecasts() -> dict[str, float]:
    """Forecast the target quarters with the saved SARIMA model.

    The horizon is derived from the last observed quarter in the HPI series, so
    refreshing the data moves the forecasts with it instead of requiring the
    numbers to be edited by hand.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data_loader import forecast_hpi_quarters

    return forecast_hpi_quarters(FORECAST_QUARTERS)["forecasts"]


def load_district_baseline() -> pd.DataFrame:
    df = pd.read_csv(IMOT_CSV, parse_dates=["snapshot_date"])
    df = df.dropna(subset=["eur_per_sqm_overall"])
    df["quarter"] = df["snapshot_date"].dt.to_period("Q").astype(str)
    q4 = df[df["quarter"] == BASELINE_QUARTER]
    baseline = (
        q4.groupby("district")["eur_per_sqm_overall"]
        .mean()
        .round(2)
        .rename("baseline_q4_2025")
        .reset_index()
    )
    baseline["snapshots_in_q4"] = (
        q4.groupby("district").size().reindex(baseline["district"]).values
    )
    return baseline


def build_forecast() -> tuple[pd.DataFrame, dict[str, float]]:
    hpi_baseline = load_hpi_baseline()
    district_df = load_district_baseline()
    hpi_forecasts = load_hpi_forecasts()

    growth = {
        label: forecast / hpi_baseline for label, forecast in hpi_forecasts.items()
    }

    out = district_df.copy()
    for label, factor in growth.items():
        out[f"forecast_{_quarter_key(label)}"] = (
            out["baseline_q4_2025"] * factor
        ).round(2)

    for label, factor in growth.items():
        out[f"growth_{_quarter_key(label)}_pct"] = round((factor - 1) * 100, 3)

    out = out.sort_values("baseline_q4_2025", ascending=False).reset_index(drop=True)

    headline = {
        "HPI_baseline_Q4_2025": hpi_baseline,
        **{f"HPI_forecast_{k}": v for k, v in hpi_forecasts.items()},
        **{f"growth_factor_{k}": round(g, 5) for k, g in growth.items()},
        **{f"growth_pct_{k}": round((g - 1) * 100, 3) for k, g in growth.items()},
        "districts_with_baseline": len(out),
        "national_imot_baseline_q4_2025": round(out["baseline_q4_2025"].mean(), 2),
    }
    return out, headline


def plot_topN(out: pd.DataFrame, headline: dict[str, float], n: int = 20) -> None:
    top = out.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = list(range(len(top)))

    quarters = list(FORECAST_QUARTERS)
    n_bars = len(quarters) + 1
    width = 0.8 / n_bars
    offsets = [(i - (n_bars - 1) / 2) * width for i in range(n_bars)]
    colors = ["gray", "seagreen", "steelblue", "darkorange"]

    ax.barh(
        [p + offsets[0] for p in y_pos],
        top["baseline_q4_2025"],
        height=width,
        label="Baseline Q4-2025 (imot.bg)",
        color=colors[0],
    )
    for i, q in enumerate(quarters, start=1):
        col_name = f"forecast_{_quarter_key(q)}"
        pct = headline[f"growth_pct_{q}"]
        ax.barh(
            [p + offsets[i] for p in y_pos],
            top[col_name],
            height=width,
            label=f"Forecast {q} ({pct:+.2f}%)",
            color=colors[i % len(colors)],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["district"])
    ax.set_xlabel("EUR/m^2")
    ax.set_title(
        f"Top {n} Sofia districts - hierarchical {'/'.join(quarters)} forecast"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "district_forecasts.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out, headline = build_forecast()

    out.to_csv(OUT_DIR / "district_forecasts.csv", index=False)
    pd.DataFrame([headline]).to_csv(OUT_DIR / "headline.csv", index=False)
    plot_topN(out, headline, n=20)

    print("National summary:")
    for k, v in headline.items():
        print(f"  {k:<40s} {v}")

    print("\nTop 10 districts by Q4-2025 baseline:")
    forecast_cols = [f"forecast_{_quarter_key(q)}" for q in FORECAST_QUARTERS]
    cols = ["district", "baseline_q4_2025"] + forecast_cols
    print(out[cols].head(10).to_string(index=False))

    print(f"\nWrote {OUT_DIR / 'district_forecasts.csv'}  ({len(out)} districts)")
    print(f"Wrote {OUT_DIR / 'headline.csv'}")
    print(f"Wrote {OUT_DIR / 'district_forecasts.png'}")


if __name__ == "__main__":
    main()
