"""Simple test script for statistical validation of synthetic returns."""

import pandas as pd

from modules.data_loader import download_historical_data
from modules.preprocessing import preprocess_historical_data
from modules.synthetic_generator import generate_synthetic_data
from modules.stress_scenarios import apply_stress_scenario
from modules.validation import (
    compare_return_distributions,
    compare_statistics,
    compare_volatility,
)


# Download, preprocess, and generate Normal Market synthetic data.
historical_data = download_historical_data(
    ticker="SPY",
    start_date="2024-01-01",
    end_date="2024-02-01",
)
preprocessed_data = preprocess_historical_data(historical_data)
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

historical_returns = preprocessed_data["Daily_Return"]
synthetic_returns = normal_market["Synthetic_Return"]

statistics = compare_statistics(historical_returns, synthetic_returns)
print("Statistical comparison table:")
print(statistics.to_string(index=False))

statistics_by_metric = statistics.set_index("Metric")

historical_mean = statistics_by_metric.loc["Mean", "Historical"]
synthetic_mean = statistics_by_metric.loc["Mean", "Synthetic"]
historical_std = statistics_by_metric.loc["Standard Deviation", "Historical"]
synthetic_std = statistics_by_metric.loc["Standard Deviation", "Synthetic"]
historical_skewness = statistics_by_metric.loc["Skewness", "Historical"]
synthetic_skewness = statistics_by_metric.loc["Skewness", "Synthetic"]
historical_kurtosis = statistics_by_metric.loc["Kurtosis", "Historical"]
synthetic_kurtosis = statistics_by_metric.loc["Kurtosis", "Synthetic"]

print("\nHistorical return mean:", historical_mean)
print("Synthetic return mean:", synthetic_mean)
print("Historical standard deviation:", historical_std)
print("Synthetic standard deviation:", synthetic_std)
print("Historical skewness:", historical_skewness)
print("Synthetic skewness:", synthetic_skewness)
print("Historical kurtosis:", historical_kurtosis)
print("Synthetic kurtosis:", synthetic_kurtosis)

# Prepare cleaned return data for plotting or distribution analysis.
historical_distribution, synthetic_distribution = compare_return_distributions(
    historical_returns,
    synthetic_returns,
)

print("\nReturn distribution data:")
print("Historical distribution size:", len(historical_distribution))
print("Synthetic distribution size:", len(synthetic_distribution))
print("Historical distribution first five:")
print(historical_distribution.head())
print("Synthetic distribution first five:")
print(synthetic_distribution.head())

# Calculate 20-day annualized rolling volatility.
rolling_volatility = compare_volatility(
    historical_returns,
    synthetic_returns,
)

print("\nRolling volatility comparison:")
print(rolling_volatility.head())
print(rolling_volatility.tail())

# Test empty input handling.
empty_statistics = compare_statistics([], [])
empty_distribution = compare_return_distributions([], [])
empty_volatility = compare_volatility([], [])

print("\nEmpty input handling:")
print("Empty statistics rows:", len(empty_statistics))
print("Empty historical distribution:", len(empty_distribution[0]))
print("Empty synthetic distribution:", len(empty_distribution[1]))
print("Empty rolling volatility rows:", len(empty_volatility))
