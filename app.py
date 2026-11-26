"""Streamlit dashboard for synthetic financial time-series stress testing."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import SUPPORTED_ASSETS, download_historical_data
from modules.preprocessing import (
    descriptive_statistics,
    preprocess_historical_data,
)
from modules.risk_analysis import calculate_risk_metrics
from modules.stress_scenarios import apply_stress_scenario
from modules.synthetic_generator import generate_synthetic_data
from modules.validation import (
    compare_return_distributions,
    compare_statistics,
    compare_volatility,
)


ASSET_LABELS = {
    "Stock": "SPY",
    "Bond": "TLT",
    "Forex": "EURUSD=X",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker, start_date, end_date):
    """Cache historical downloads for one hour."""
    return download_historical_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )


def configure_page():
    """Configure the Streamlit page and dashboard styles."""
    st.set_page_config(
        page_title="Synthetic Financial Stress Testing",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #07111f;
        }
        [data-testid="stHeader"] {
            background-color: #07111f;
        }
        [data-testid="stSidebar"] {
            background-color: #0b1929;
            border-right: 1px solid #1c3550;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #eaf2ff;
        }
        h1, h2, h3, p, label, span {
            color: #eaf2ff;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #0f2742, #12365a);
            border: 1px solid #1e4d78;
            border-radius: 10px;
            padding: 16px;
            min-height: 135px;
        }
        [data-testid="stMetricLabel"] {
            color: #9fc5e8;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #1c3550;
            border-radius: 8px;
        }
        .dashboard-caption {
            color: #9fb3c8;
            margin-bottom: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_percentage(value):
    """Format a metric safely as a percentage."""
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.2%}"


def create_line_chart(data, title, columns):
    """Create a professional line chart for a DataFrame."""
    fig = go.Figure()
    colors = ["#56b4e9", "#f0a500"]

    for column, color in zip(columns, colors):
        if column not in data.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                name=column.replace("_", " "),
                line={"color": color, "width": 2},
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.01},
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1c3550")
    fig.update_yaxes(showgrid=True, gridcolor="#1c3550")
    return fig


configure_page()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Configure historical data and synthetic stress settings.")

asset_category = st.sidebar.selectbox(
    "Asset",
    options=list(ASSET_LABELS.keys()),
    help="Select the financial asset to download and analyze.",
)
ticker = ASSET_LABELS[asset_category]

start_date = st.sidebar.date_input(
    "Historical start date",
    value=pd.Timestamp("2023-01-01").date(),
)
end_date = st.sidebar.date_input(
    "Historical end date",
    value=pd.Timestamp("2024-12-31").date(),
)

scenario_name = st.sidebar.selectbox(
    "Market scenario",
    options=["Normal Market", "High Volatility", "Market Crash"],
)

generation_days = st.sidebar.slider(
    "Synthetic generation days",
    min_value=30,
    max_value=1000,
    value=365,
    step=1,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=100000,
    value=42,
    step=1,
)

risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate",
    min_value=-1.0,
    max_value=1.0,
    value=0.02,
    step=0.005,
    format="%.3f",
)

generate_button = st.sidebar.button(
    "Generate Analysis",
    type="primary",
    use_container_width=True,
)

st.title("Synthetic Financial Time-Series Generator")
st.markdown(
    '<p class="dashboard-caption">'
    "Load historical market data, generate synthetic returns, apply stress "
    "scenarios, and compare statistical and financial risk characteristics."
    "</p>",
    unsafe_allow_html=True,
)

if generate_button:
    if start_date >= end_date:
        st.error("The historical start date must be earlier than the end date.")
    else:
        with st.spinner(f"Loading {ticker} historical data..."):
            try:
                historical = load_historical_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )
                processed = preprocess_historical_data(historical)

                if processed.empty:
                    raise RuntimeError(
                        "No valid historical observations were available "
                        "after preprocessing."
                    )

                synthetic = generate_synthetic_data(
                    preprocessed_data=processed,
                    number_of_days=generation_days,
                    random_seed=int(random_seed),
                )

                if synthetic.empty:
                    raise RuntimeError(
                        "Synthetic data could not be generated from the "
                        "selected historical period."
                    )

                stressed = apply_stress_scenario(
                    synthetic_data=synthetic,
                    scenario_name=scenario_name,
                    volatility_multiplier=2.0,
                    crash_days=5,
                    crash_shock=-0.10,
                    random_seed=int(random_seed),
                )

                st.session_state["dashboard_results"] = {
                    "historical": historical,
                    "processed": processed,
                    "synthetic": synthetic,
                    "stressed": stressed,
                    "ticker": ticker,
                    "scenario": scenario_name,
                }
            except ValueError as exc:
                st.error(f"Invalid input: {exc}")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(
                    "An unexpected error occurred while loading or "
                    f"processing the data: {exc}"
                )

if "dashboard_results" not in st.session_state:
    st.info(
        "Choose an asset and analysis settings from the sidebar, "
        "then select **Generate Analysis**."
    )
else:
    results = st.session_state["dashboard_results"]
    historical = results["historical"]
    processed = results["processed"]
    synthetic = results["synthetic"]
    stressed = results["stressed"]
    ticker = results["ticker"]
    scenario_name = results["scenario"]

    historical_returns = processed["Daily_Return"].dropna()
    synthetic_returns = stressed["Synthetic_Return"].dropna()

    historical_risk = calculate_risk_metrics(
        daily_returns=historical_returns,
        price_series=processed["Close"],
        risk_free_rate=float(risk_free_rate),
    )
    synthetic_risk = calculate_risk_metrics(
        daily_returns=synthetic_returns,
        price_series=stressed["Synthetic_Price"],
        risk_free_rate=float(risk_free_rate),
    )

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    kpi_col1.metric(
        "Historical Volatility",
        format_percentage(historical_risk["Annualized_Volatility"]),
    )
    kpi_col2.metric(
        "Synthetic Volatility",
        format_percentage(synthetic_risk["Annualized_Volatility"]),
    )
    kpi_col3.metric(
        "Value at Risk (95%)",
        format_percentage(synthetic_risk["VaR_95"]),
    )
    kpi_col4.metric(
        "Maximum Drawdown",
        format_percentage(synthetic_risk["Maximum_Drawdown"]),
    )

    price_tab, return_tab, volatility_tab, distribution_tab = st.tabs(
        [
            "Price Comparison",
            "Return Analysis",
            "Volatility Analysis",
            "Distribution Analysis",
        ]
    )

    with price_tab:
        price_comparison = pd.DataFrame(
            {
                "Date": pd.to_datetime(processed["Date"]),
                "Historical Price": processed["Close"].to_numpy(),
            }
        )
        synthetic_prices = pd.DataFrame(
            {
                "Date": pd.to_datetime(stressed["Date"]),
                "Synthetic Price": stressed["Synthetic_Price"].to_numpy(),
            }
        )

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=price_comparison["Date"],
                y=price_comparison["Historical Price"],
                mode="lines",
                name=f"Historical {ticker}",
                line={"color": "#56b4e9", "width": 2},
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=synthetic_prices["Date"],
                y=synthetic_prices["Synthetic Price"],
                mode="lines",
                name=f"Synthetic ({scenario_name})",
                line={"color": "#f0a500", "width": 2},
            )
        )
        price_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.08},
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            height=430,
        )
        price_fig.update_xaxes(showgrid=True, gridcolor="#1c3550")
        price_fig.update_yaxes(showgrid=True, gridcolor="#1c3550")
        st.plotly_chart(price_fig, use_container_width=True)

    with return_tab:
        return_comparison = pd.DataFrame(
            {
                "Historical Daily Return": historical_returns.reset_index(
                    drop=True
                ),
                "Synthetic Daily Return": synthetic_returns.reset_index(
                    drop=True
                ),
            }
        )
        return_comparison = return_comparison.dropna(how="all")
        st.plotly_chart(
            create_line_chart(
                return_comparison,
                "Historical vs Synthetic Daily Returns",
                [
                    "Historical Daily Return",
                    "Synthetic Daily Return",
                ],
            ),
            use_container_width=True,
        )

    with volatility_tab:
        rolling_volatility = compare_volatility(
            historical_returns,
            synthetic_returns,
            window=20,
            trading_days=252,
        )
        st.plotly_chart(
            create_line_chart(
                rolling_volatility,
                "20-Day Annualized Rolling Volatility",
                [
                    "Historical_Volatility",
                    "Synthetic_Volatility",
                ],
            ),
            use_container_width=True,
        )

    with distribution_tab:
        historical_distribution, synthetic_distribution = (
            compare_return_distributions(
                historical_returns,
                synthetic_returns,
            )
        )

        distribution_fig = go.Figure()
        if not historical_distribution.empty:
            distribution_fig.add_trace(
                go.Histogram(
                    x=historical_distribution,
                    name="Historical",
                    opacity=0.65,
                    marker={"color": "#56b4e9"},
                    histnorm="probability",
                )
            )
        if not synthetic_distribution.empty:
            distribution_fig.add_trace(
                go.Histogram(
                    x=synthetic_distribution,
                    name="Synthetic",
                    opacity=0.65,
                    marker={"color": "#f0a500"},
                    histnorm="probability",
                )
            )
        distribution_fig.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            title={"text": "Daily Return Distribution", "x": 0.01},
            legend={"orientation": "h", "y": 1.08},
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            height=430,
            bargap=0.05,
        )
        distribution_fig.update_xaxes(showgrid=True, gridcolor="#1c3550")
        distribution_fig.update_yaxes(showgrid=True, gridcolor="#1c3550")
        st.plotly_chart(distribution_fig, use_container_width=True)

    st.subheader("Statistical Validation")
    statistics_table = compare_statistics(
        historical_returns,
        synthetic_returns,
    )
    st.dataframe(
        statistics_table,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Similar summary statistics do not by themselves establish that "
        "synthetic data is realistic."
    )

    st.subheader("Financial Risk Analysis")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    risk_col1.metric("VaR (95%)", format_percentage(synthetic_risk["VaR_95"]))
    risk_col2.metric("CVaR (95%)", format_percentage(synthetic_risk["CVaR_95"]))
    risk_col3.metric(
        "Maximum Drawdown",
        format_percentage(synthetic_risk["Maximum_Drawdown"]),
    )
    risk_col4.metric(
        "Sharpe Ratio",
        f"{synthetic_risk['Annualized_Sharpe_Ratio']:.2f}"
        if np.isfinite(synthetic_risk["Annualized_Sharpe_Ratio"])
        else "N/A",
    )

    st.subheader("Export Synthetic Data")
    csv_data = stressed.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Synthetic Data as CSV",
        data=csv_data,
        file_name=f"{ticker}_synthetic_{scenario_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )