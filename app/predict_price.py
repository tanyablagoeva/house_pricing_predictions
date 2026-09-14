from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "imot_avg_prices.csv"
MLP_MODEL_PATH = ROOT / "models" / "per_district_final.joblib"

QUARTER_MID_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}
QUARTER_MONTHS = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}

# Forecast horizon. Mirrors QUARTER_DATES in tools/predict_all.py.
QUARTERS = ["Q2-2026", "Q3-2026", "Q4-2026"]


def _parse_quarter(label: str) -> tuple[int, int]:
    q, year = label.split("-")
    return int(q[1]), int(year)


def quarter_with_months(label: str) -> str:
    q_num, _ = _parse_quarter(label)
    return f"{label} ({QUARTER_MONTHS[q_num]})"


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
    df = df.dropna(subset=["eur_per_sqm_overall"])
    return df[["snapshot_date", "district", "eur_per_sqm_overall"]]


@st.cache_data(show_spinner=False)
def load_sarima_context() -> dict | None:
    """National HPI forecast, computed live from the saved SARIMA model.

    Returns None only if the model or the HPI series is absent, in which case
    the national-context panel is hidden rather than failing the app.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.data_loader import forecast_hpi_quarters

        return forecast_hpi_quarters(list(QUARTERS))
    except (FileNotFoundError, ImportError):
        return None


@st.cache_resource(show_spinner=False)
def load_mlp():
    return joblib.load(MLP_MODEL_PATH)


@st.cache_data(show_spinner=False)
def predict(district: str, quarters: tuple[str, ...]) -> dict[str, float]:
    """Run the saved MLP pipeline for one district across the forecast quarters.

    The pipeline carries its own preprocessing, so the only inputs needed are the
    district name and four date parts derived from the quarter label.
    """
    rows = []
    for q in quarters:
        ts = quarter_to_date(q)
        rows.append(
            {
                "district": district,
                "year": ts.year,
                "month": ts.month,
                "week_of_year": int(ts.isocalendar().week),
                "quarter": (ts.month - 1) // 3 + 1,
            }
        )
    values = load_mlp().predict(pd.DataFrame(rows))
    return {q: float(v) for q, v in zip(quarters, values)}


def quarter_to_date(label: str) -> pd.Timestamp:
    q_num, year = _parse_quarter(label)
    return pd.Timestamp(year=year, month=QUARTER_MID_MONTH[q_num], day=15)


def format_price(value: float) -> str:
    return f"{value:,.0f} EUR/m²"


def format_delta(forecast: float, baseline: float) -> str:
    if baseline <= 0:
        return ""
    pct = (forecast - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs today"


def plot_history_with_forecast(
    history: pd.DataFrame,
    district: str,
    quarter_predictions: list[tuple[str, float]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    series = history[history["district"] == district].sort_values("snapshot_date")
    ax.plot(
        series["snapshot_date"],
        series["eur_per_sqm_overall"],
        color="#1f77b4",
        linewidth=1.2,
        label="Observed (imot.bg weekly avg)",
    )
    forecast_dates = [quarter_to_date(q) for q, _ in quarter_predictions]
    forecast_values = [v for _, v in quarter_predictions]
    quarter_labels = ", ".join(quarter_with_months(q) for q, _ in quarter_predictions)
    ax.scatter(
        forecast_dates,
        forecast_values,
        color="#d62728",
        s=25,
        zorder=5,
        label=f"MLP forecast ({quarter_labels})",
    )
    ax.set_title(f"{district} - EUR/m² history and forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUR per m²")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Sofia House Price Forecast",
        page_icon=None,
        layout="centered",
    )

    st.title("Sofia House Price Forecast")
    st.caption(
        "Neural network forecast (MLP) for the next three quarters, by Sofia "
        "district, computed live from the trained model."
    )

    history = load_history()
    sarima = load_sarima_context()

    districts = sorted(history["district"].unique().tolist())
    quarters = list(QUARTERS)

    with st.form("predict_form"):
        district = st.selectbox(
            "District",
            options=districts,
            index=0,
            help="176 Sofia districts the MLP was trained on.",
        )
        submitted = st.form_submit_button("Predict")

    if not submitted and "last_district" not in st.session_state:
        st.info("Select a district and press **Predict**.")
        return

    if submitted:
        st.session_state["last_district"] = district
    district = st.session_state["last_district"]

    # Headline forecast: run the saved MLP pipeline live for this district.
    mlp_forecast = predict(district, tuple(quarters))

    district_history = history[history["district"] == district]
    today_window_start = pd.Timestamp("2026-04-01")
    today_mask = district_history["snapshot_date"] >= today_window_start
    today_rows = district_history.loc[today_mask]
    if not today_rows.empty:
        baseline_today = today_rows["eur_per_sqm_overall"].mean()
        today_window_end = today_rows["snapshot_date"].max()
        today_window_label = (
            f"{today_window_start.strftime('%b %Y')} to "
            f"{today_window_end.strftime('%b %d, %Y')}"
        )
    else:
        baseline_today = float("nan")
        today_window_label = ""

    st.subheader(f"Forecast - {district}")
    metric_cols = st.columns(len(quarters))
    for col, q in zip(metric_cols, quarters):
        value = mlp_forecast[q]
        col.metric(
            f"{quarter_with_months(q)} (MLP)",
            format_price(value),
            (
                format_delta(value, baseline_today)
                if pd.notna(baseline_today)
                else None
            ),
        )

    if pd.notna(baseline_today):
        st.caption(
            f"Today's average for this district (imot.bg, {today_window_label}): "
            f"**{format_price(baseline_today)}**"
        )

    st.subheader("Historical prices for this district")
    if district_history.empty:
        st.warning("No historical observations available for this district.")
    else:
        quarter_predictions = [(q, mlp_forecast[q]) for q in quarters]
        fig = plot_history_with_forecast(history, district, quarter_predictions)
        st.pyplot(fig)
        plt.close(fig)

    if sarima is not None:
        with st.expander("National context (SARIMA HPI forecast)"):
            st.write(
                f"**Bulgarian HPI {sarima['baseline_quarter']} baseline:** "
                f"{sarima['baseline']:.2f}"
            )
            for q in quarters:
                if q in sarima["forecasts"]:
                    st.write(
                        f"**{quarter_with_months(q)} forecast:** "
                        f"{sarima['forecasts'][q]:.2f} "
                        f"({sarima['growth_pct'][q]:+.2f}%)"
                    )
            st.caption(
                "Computed live from `models/sarima_final.joblib`. These growth "
                "rates describe the national market. The district forecast above is "
                "a separate MLP trained on imot.bg district data."
            )


if __name__ == "__main__":
    main()
