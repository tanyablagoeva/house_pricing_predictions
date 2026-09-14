from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "imot_avg_prices.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "per_district_forecasts.csv"
MLP_MODEL_PATH = ROOT / "models" / "per_district_final.joblib"
HEADLINE_PATH = ROOT / "outputs" / "hierarchical_forecast" / "headline.csv"
TUNING_PATH = ROOT / "outputs" / "tuning" / "full_run.json"

QUARTER_MID_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}
QUARTER_MONTHS = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}


def _parse_quarter(label: str) -> tuple[int, int]:
    q, year = label.split("-")
    return int(q[1]), int(year)


def quarter_with_months(label: str) -> str:
    q_num, _ = _parse_quarter(label)
    return f"{label} ({QUARTER_MONTHS[q_num]})"


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
    df = df.dropna(subset=["eur_per_sqm_overall"])
    return df[["snapshot_date", "district", "eur_per_sqm_overall"]]


@st.cache_data(show_spinner=False)
def load_headline() -> pd.Series | None:
    if not HEADLINE_PATH.exists():
        return None
    return pd.read_csv(HEADLINE_PATH).iloc[0]


@st.cache_data(show_spinner=False)
def load_mlp_metrics() -> dict | None:
    if not TUNING_PATH.exists():
        return None
    payload = json.loads(TUNING_PATH.read_text())
    for entry in payload["results"]:
        if entry["model"].startswith("MLP"):
            return {
                "test_mae": entry["test_mae"],
                "test_rmse": entry["test_rmse"],
                "test_mape": entry["test_mape"],
                "test_r2": entry["test_r2"],
            }
    return None


@st.cache_resource(show_spinner=False)
def load_mlp():
    return joblib.load(MLP_MODEL_PATH)


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
        "Neural network forecast (MLP) for the next two quarters, by Sofia district. "
        "Comparison panel shows what other tuned algorithms predicted for the same cell."
    )

    predictions = load_predictions()
    history = load_history()
    headline = load_headline()
    mlp_metrics = load_mlp_metrics()

    districts = sorted(predictions["district"].unique().tolist())
    quarters = list(predictions["quarter"].drop_duplicates())

    with st.form("predict_form"):
        district = st.selectbox(
            "District",
            options=districts,
            index=0,
            help="151 Sofia districts have a Q4-2025 imot.bg baseline.",
        )
        submitted = st.form_submit_button("Predict")

    if not submitted and "last_district" not in st.session_state:
        st.info("Select a district and press **Predict**.")
        return

    if submitted:
        st.session_state["last_district"] = district
    district = st.session_state["last_district"]

    district_rows = predictions[predictions["district"] == district]
    quarter_rows = {
        q: district_rows[district_rows["quarter"] == q].iloc[0] for q in quarters
    }

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
        row = quarter_rows[q]
        col.metric(
            f"{quarter_with_months(q)} (MLP)",
            format_price(row["MLP"]),
            (
                format_delta(row["MLP"], baseline_today)
                if pd.notna(baseline_today)
                else None
            ),
        )

    if pd.notna(baseline_today):
        st.caption(
            f"Today's average for this district (imot.bg, {today_window_label}): "
            f"**{format_price(baseline_today)}**"
        )
        st.caption(
            "_Disclaimer: training data and this baseline both end 2026-05-14. "
            "Real current prices may sit ~100 EUR/m^2 above or below the figure shown, "
            "depending on more recent listings not included in the snapshot._"
        )

    if mlp_metrics is not None:
        mae = mlp_metrics["test_mae"]
        mape = mlp_metrics["test_mape"]
        r2 = mlp_metrics["test_r2"]
        st.caption(
            f"**MLP confidence:** typical error +/- {mae:.0f} EUR/m^2 "
            f"({mape:.1f}% MAPE, R^2 = {r2:.2f}). "
            "Measured on the held-out last 12 months of imot.bg observations "
            "after Bayesian hyperparameter tuning (250 Optuna trials, 5-fold TimeSeriesSplit CV)."
        )
        if pd.notna(baseline_today) and baseline_today > 0:
            parts = []
            for q in quarters:
                pct = (quarter_rows[q]["MLP"] - baseline_today) / baseline_today * 100
                direction = "increase" if pct >= 0 else "decrease"
                sign = "+" if pct >= 0 else ""
                parts.append(
                    f"a {sign}{pct:.1f}% {direction} in {quarter_with_months(q)}"
                )
            change_text = " and ".join(parts)
            st.caption(
                f"**What the percentage means:** the model predicts {change_text} "
                f"versus what sellers are asking right now ({today_window_label}) in this district."
            )
        st.caption(
            f"**Is it legit?** the {mape:.1f}% MAPE and R^2 = {r2:.2f} above "
            f"come from predicting the last 12 months of real imot.bg data "
            f"that the model never saw during training. A made-up model would "
            f"score MAPE around 30-50% and R^2 near 0 on that test."
        )

    st.subheader("Comparison panel - what other algorithms predicted")
    algorithms = ["MLP", "LightGBM", "XGBoost", "GBM", "KNN", "SARIMA_hierarchical"]
    display_names = {
        "MLP": "MLP (neural network)",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "GBM": "GBM",
        "KNN": "KNN",
        "SARIMA_hierarchical": "SARIMA hierarchical",
    }
    comparison_data = {"Algorithm": [display_names[a] for a in algorithms]}
    for q in quarters:
        col_name = f"{quarter_with_months(q)} (EUR/m²)"
        comparison_data[col_name] = [quarter_rows[q][a] for a in algorithms]
    comparison = pd.DataFrame(comparison_data)
    st.dataframe(
        comparison.style.format(
            {f"{quarter_with_months(q)} (EUR/m²)": "{:,.0f}" for q in quarters}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "MLP is the production headline. The other rows are kept for context - the "
        "model selection rationale lives in `outputs/algorithm_comparison/` and the tuning "
        "results in `outputs/tuning/`."
    )

    st.subheader("Historical prices for this district")
    if district_history.empty:
        st.warning("No historical observations available for this district.")
    else:
        quarter_predictions = [(q, quarter_rows[q]["MLP"]) for q in quarters]
        fig = plot_history_with_forecast(history, district, quarter_predictions)
        st.pyplot(fig)
        plt.close(fig)

    if headline is not None:
        with st.expander("National context (SARIMA HPI forecast)"):
            st.write(
                f"**Bulgarian HPI Q4-2025 baseline:** {headline['HPI_baseline_Q4_2025']:.2f}"
            )
            for q in quarters:
                hpi_col = f"HPI_forecast_{q}"
                pct_col = f"growth_pct_{q}"
                if hpi_col in headline.index and pct_col in headline.index:
                    st.write(
                        f"**{quarter_with_months(q)} forecast:** {headline[hpi_col]:.2f} "
                        f"({headline[pct_col]:+.2f}%)"
                    )
            st.caption(
                "These growth factors are what the SARIMA hierarchical method applies "
                "to each district's Q4-2025 baseline."
            )


if __name__ == "__main__":
    main()
