"""Simple test script for market stress scenarios."""

import numpy as np
import pandas as pd

from modules.data_loader import download_historical_data
from modules.preprocessing import preprocess_historical_data
from modules.synthetic_generator import generate_synthetic_data
from modules.stress_scenarios import apply_stress_scenario


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

scenario_names = ["Normal Market", "High Volatility", "Market Crash"]
scenarios = {
    "Normal Market": apply_stress_scenario(
        synthetic_data,
        "Normal Market",
        random_seed=42,
    ),
    "High Volatility": apply_stress_scenario(
        synthetic_data,
        "High Volatility",
        volatility_multiplier=2.0,
        random_seed=42,
    ),
    "Market Crash": apply_stress_scenario(
        synthetic_data,
        "Market Crash",
        volatility_multiplier=2.0,
        crash_days=5,
        crash_shock=-0.10,
        random_seed=42,
    ),
}

for scenario_name in scenario_names:
    scenario_data = scenarios[scenario_name]

    print(f"\n{scenario_name} - First five rows:")
    print(scenario_data.head())

    print(f"{scenario_name} - Mean return:")
    print(scenario_data["Synthetic_Return"].mean())

    print(f"{scenario_name} - Standard deviation:")
    print(scenario_data["Synthetic_Return"].std())

    print(f"{scenario_name} - Final synthetic price:")
    print(scenario_data["Synthetic_Price"].iloc[-1])

    if len(scenario_data) != 365:
        raise AssertionError(
            f"{scenario_name} returned {len(scenario_data)} rows; expected 365."
        )

    if scenario_data["Synthetic_Price"].isna().any():
        raise AssertionError(f"{scenario_name} contains missing prices.")

    if not (scenario_data["Synthetic_Price"] > 0).all():
        raise AssertionError(f"{scenario_name} contains non-positive prices.")

normal_data = scenarios["Normal Market"]
if not np.array_equal(
    normal_data["Synthetic_Return"].to_numpy(),
    synthetic_data["Synthetic_Return"].to_numpy(),
):
    raise AssertionError("Normal Market did not preserve original returns.")

high_volatility_data = scenarios["High Volatility"]
market_crash_data = scenarios["Market Crash"]

original_std = synthetic_data["Synthetic_Return"].std()
high_std = high_volatility_data["Synthetic_Return"].std()

if high_std <= original_std:
    raise AssertionError(
        "High Volatility standard deviation was not increased."
    )

if not (market_crash_data["Synthetic_Return"] < -0.05).any():
    raise AssertionError("Market Crash does not contain a sufficiently negative shock.")

# Generate each scenario again with the same seed and confirm identical results.
repeat_normal = apply_stress_scenario(
    synthetic_data,
    "Normal Market",
    random_seed=42,
)
repeat_high = apply_stress_scenario(
    synthetic_data,
    "High Volatility",
    volatility_multiplier=2.0,
    random_seed=42,
)
repeat_crash = apply_stress_scenario(
    synthetic_data,
    "Market Crash",
    volatility_multiplier=2.0,
    crash_days=5,
    crash_shock=-0.10,
    random_seed=42,
)

pd.testing.assert_frame_equal(normal_data, repeat_normal)
pd.testing.assert_frame_equal(high_volatility_data, repeat_high)
pd.testing.assert_frame_equal(market_crash_data, repeat_crash)

print("\nVerification passed:")
print("- All scenarios contain exactly 365 rows.")
print("- All synthetic prices are positive.")
print("- Normal Market preserves original synthetic returns.")
print("- High Volatility increases the return standard deviation.")
print("- Market Crash contains negative shocks.")
print("- Every scenario is reproducible with the same random seed.")
