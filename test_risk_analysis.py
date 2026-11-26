"""Simple test script for historical and synthetic risk metrics."""

import math

import pandas as pd

from modules.data_loader import download_historical_data
from modules.preprocessing import preprocess_historical_data
from modules.synthetic_generator import generate_synthetic_data
from modules.stress_scenarios import apply_stress_scenario
from modules.risk_analysis import calculate_risk_metrics


# Download and preprocess January 2024 SPY data.
historical_data = download_historical_data(
    ticker="SPY",
    start_date="2024-01-01",
    end_date="2024-02-01",
)
preprocessed_data = preprocess_historical_data(historical_data)

# Generate synthetic data and apply the Normal Market scenario.
synthetic_data = generate_synthetic_data(
    preprocessed_data=preprocessed_data,
    number_of_days=365,
    random_seed=42,
)
normal_market = apply_stress_scenario(
    synthetic_data,
    scenario_name="Normal Market",
    random_seed=42,
)

# Calculate historical and synthetic risk metrics.
historical_metrics = calculate_risk_metrics(
    daily_returns=preprocessed_data["Daily_Return"],
    price_series=preprocessed_data["Close"],
)
synthetic_metrics = calculate_risk_metrics(
    daily_returns=normal_market["Synthetic_Return"],
    price_series=normal_market["Synthetic_Price"],
)

print("Historical volatility:", historical_metrics["Annualized_Volatility"])
print("Synthetic volatility:", synthetic_metrics["Annualized_Volatility"])
print("Historical VaR:", historical_metrics["VaR_95"])
print("Synthetic VaR:", synthetic_metrics["VaR_95"])
print("Historical CVaR:", historical_metrics["CVaR_95"])
print("Synthetic CVaR:", synthetic_metrics["CVaR_95"])
print("Historical maximum drawdown:", historical_metrics["Maximum_Drawdown"])
print("Synthetic maximum drawdown:", synthetic_metrics["Maximum_Drawdown"])
print("Historical Sharpe ratio:", historical_metrics["Annualized_Sharpe_Ratio"])
print("Synthetic Sharpe ratio:", synthetic_metrics["Annualized_Sharpe_Ratio"])

# Verify that every requested metric is a finite number.
for dataset_name, metrics in (
    ("Historical", historical_metrics),
    ("Synthetic", synthetic_metrics),
):
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, (int, float)):
            raise AssertionError(
                f"{dataset_name} {metric_name} is not numeric."
            )
        if not math.isfinite(float(metric_value)):
            raise AssertionError(
                f"{dataset_name} {metric_name} is not finite."
            )

# VaR and CVaR use a positive-loss convention.
for dataset_name, metrics in (
    ("Historical", historical_metrics),
    ("Synthetic", synthetic_metrics),
):
    if metrics["VaR_95"] < 0:
        raise AssertionError(f"{dataset_name} VaR must be non-negative.")
    if metrics["CVaR_95"] < 0:
        raise AssertionError(f"{dataset_name} CVaR must be non-negative.")

# Empty input should return a metric dictionary with NaN values.
empty_metrics = calculate_risk_metrics([], [])
if not isinstance(empty_metrics, dict):
    raise AssertionError("Empty input did not return a metrics dictionary.")
for metric_name, metric_value in empty_metrics.items():
    if not pd.isna(metric_value):
        raise AssertionError(
            f"Empty input should return NaN for {metric_name}."
        )
print("\nEmpty input handling: passed")

# Zero daily returns and constant prices must not create division errors.
zero_metrics = calculate_risk_metrics(
    daily_returns=pd.Series([0.0, 0.0, 0.0]),
    price_series=pd.Series([100.0, 100.0, 100.0]),
)

for metric_name, metric_value in zero_metrics.items():
    if not math.isfinite(float(metric_value)):
        raise AssertionError(
            f"Zero-volatility case produced an invalid {metric_name}."
        )

print("Zero-volatility handling: passed")
print("\nRisk metric verification: passed")
