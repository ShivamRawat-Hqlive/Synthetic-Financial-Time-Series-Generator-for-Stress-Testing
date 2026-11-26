"""Simple test script for synthetic financial data generation."""

import numpy as np
import pandas as pd

from modules.data_loader import download_historical_data
from modules.preprocessing import (
    descriptive_statistics,
    preprocess_historical_data,
)
from modules.synthetic_generator import generate_synthetic_data


# Download and preprocess SPY data for January 2024.
historical_data = download_historical_data(
    ticker="SPY",
    start_date="2024-01-01",
    end_date="2024-02-01",
)
preprocessed_data = preprocess_historical_data(historical_data)

# Generate 365 days using a fixed seed for reproducible results.
synthetic_data = generate_synthetic_data(
    preprocessed_data=preprocessed_data,
    number_of_days=365,
    random_seed=42,
)

print("First five rows:")
print(synthetic_data.head())

print("\nLast five rows:")
print(synthetic_data.tail())

synthetic_return_mean = synthetic_data["Synthetic_Return"].mean()
synthetic_return_std = synthetic_data["Synthetic_Return"].std()
print("\nSynthetic return mean:")
print(synthetic_return_mean)
print("\nSynthetic return standard deviation:")
print(synthetic_return_std)

# Verify that all synthetic prices are positive.
if synthetic_data["Synthetic_Price"].isna().any():
    raise AssertionError("Synthetic prices contain missing values.")
if not (synthetic_data["Synthetic_Price"] > 0).all():
    raise AssertionError("All synthetic prices must be positive.")
print("\nVerification passed: all synthetic prices are positive.")

# Verify the requested output size.
if len(synthetic_data) != 365:
    raise AssertionError(
        f"Expected 365 rows, but generated {len(synthetic_data)}."
    )
print("Verification passed: exactly 365 rows were generated.")

# Verify that the same seed creates identical Synthetic_Return and
# Synthetic_Price values.
repeat_data = generate_synthetic_data(
    preprocessed_data=preprocessed_data,
    number_of_days=365,
    random_seed=42,
)

pd.testing.assert_frame_equal(synthetic_data, repeat_data)
print("Verification passed: the same random seed produced identical results.")
