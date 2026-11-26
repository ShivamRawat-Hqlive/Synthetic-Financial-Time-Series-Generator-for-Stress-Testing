"""Simple test script for historical data preprocessing."""

import numpy as np
import pandas as pd

from modules.data_loader import download_historical_data
from modules.preprocessing import (
    descriptive_statistics,
    preprocess_historical_data,
)


# Download SPY data for January 2024.
# The end date is exclusive, so use February 1 to include January 31.
historical_data = download_historical_data(
    ticker="SPY",
    start_date="2024-01-01",
    end_date="2024-02-01",
)

# Clean the downloaded data and calculate returns and volatility.
processed_data = preprocess_historical_data(historical_data)

print("First five rows:")
print(processed_data.head())

print("\nDataFrame columns:")
print(processed_data.columns.tolist())

print("\nDescriptive statistics:")
print(descriptive_statistics(processed_data))

print("\nMissing value counts:")
print(processed_data.isna().sum())

# Verify Daily_Return against Pandas pct_change().
expected_returns = processed_data["Close"].pct_change()
pd.testing.assert_series_equal(
    processed_data["Daily_Return"],
    expected_returns,
    check_names=False,
)

# Verify Rolling_Volatility against a 20-day rolling sample standard deviation.
expected_volatility = (
    processed_data["Daily_Return"]
    .rolling(window=20, min_periods=20)
    .std()
    * np.sqrt(252)
)
pd.testing.assert_series_equal(
    processed_data["Rolling_Volatility"],
    expected_volatility,
    check_names=False,
)

print("\nVerification passed: Daily_Return and Rolling_Volatility are correct.")
