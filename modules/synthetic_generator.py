"""Generate simple synthetic financial time-series data."""

import numpy as np
import pandas as pd


def generate_synthetic_data(
    preprocessed_data: pd.DataFrame,
    number_of_days: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily returns and prices.

    The generator starts with the last historical close price and draws
    normally distributed daily returns using the mean and standard deviation
    of the historical daily returns.

    Args:
        preprocessed_data: Historical DataFrame containing Date, Close, and
            Daily_Return columns.
        number_of_days: Number of future calendar days to generate.
        random_seed: Seed used to make the generated values reproducible.

    Returns:
        A DataFrame containing Date, Synthetic_Return, and Synthetic_Price.

    Raises:
        ValueError: If number_of_days is not a positive integer.
    """
    if isinstance(number_of_days, bool) or not isinstance(number_of_days, (int, np.integer)):
        raise ValueError("number_of_days must be a positive integer.")

    if number_of_days <= 0:
        raise ValueError("number_of_days must be a positive integer.")

    if not isinstance(preprocessed_data, pd.DataFrame):
        raise ValueError("Preprocessed data must be a Pandas DataFrame.")

    if preprocessed_data.empty:
        return pd.DataFrame(
            columns=["Date", "Synthetic_Return", "Synthetic_Price"]
        )

    missing_columns = {"Date", "Close", "Daily_Return"} - set(
        preprocessed_data.columns
    )
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Preprocessed data is missing required columns: {missing}")

    clean_data = preprocessed_data.loc[
        :, ["Date", "Close", "Daily_Return"]
    ].copy()
    clean_data["Date"] = pd.to_datetime(clean_data["Date"], errors="coerce")
    clean_data["Close"] = pd.to_numeric(clean_data["Close"], errors="coerce")
    clean_data["Daily_Return"] = pd.to_numeric(
        clean_data["Daily_Return"], errors="coerce"
    )
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
    clean_data = clean_data.dropna(subset=["Date", "Close"])
    clean_data = clean_data.sort_values("Date").reset_index(drop=True)

    valid_data = clean_data.loc[clean_data["Close"] > 0].copy()
    if valid_data.empty:
        return pd.DataFrame(
            columns=["Date", "Synthetic_Return", "Synthetic_Price"]
        )

    # Use the last valid positive close as the starting synthetic price.
    last_historical_close = float(valid_data["Close"].iloc[-1])
    last_historical_date = valid_data["Date"].iloc[-1]

    # Keep only usable historical returns for the normal distribution.
    valid_returns = clean_data["Daily_Return"].dropna()
    valid_returns = valid_returns[
        np.isfinite(valid_returns)
    ]

    if valid_returns.empty:
        # With no historical returns, there is no volatility or trend to sample.
        historical_mean = 0.0
        historical_std = 0.0
    else:
        historical_mean = float(valid_returns.mean())
        historical_std = float(valid_returns.std(ddof=1))

        # A one-row series has no standard deviation, so use a zero volatility.
        if not np.isfinite(historical_std) or historical_std < 0:
            historical_std = 0.0

    # NumPy's normal distribution is seeded for reproducible results.
    rng = np.random.default_rng(random_seed)
    synthetic_returns = rng.normal(
        loc=historical_mean,
        scale=historical_std,
        size=number_of_days,
    )

    # Start with the last historical close, then compound one day at a time.
    synthetic_prices = np.full(number_of_days, np.nan, dtype=float)
    previous_price = last_historical_close

    for index, synthetic_return in enumerate(synthetic_returns):
        previous_price = previous_price * (1.0 + synthetic_return)
        # Prevent invalid negative or infinite prices from propagating.
        if not np.isfinite(previous_price) or previous_price <= 0:
            synthetic_prices[index] = np.nan
            break
        synthetic_prices[index] = previous_price

    # Generate future calendar dates immediately after the last historical date.
    future_dates = pd.date_range(
        start=last_historical_date + pd.Timedelta(days=1),
        periods=number_of_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Date": future_dates,
            "Synthetic_Return": synthetic_returns,
            "Synthetic_Price": synthetic_prices,
        }
    )