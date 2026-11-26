"""Clean historical financial data and calculate basic return analysis."""

import numpy as np
import pandas as pd


def preprocess_historical_data(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Clean a historical price DataFrame and add return calculations.

    The input must contain Date and Close columns. The output contains:
    - Date
    - Close
    - Daily_Return
    - Rolling_Volatility

    With fewer than 20 observations, rolling volatility values are calculated
    only when enough returns are available and remain NaN where insufficient.
    """
    if not isinstance(historical_data, pd.DataFrame):
        raise ValueError("Historical data must be a Pandas DataFrame.")

    if historical_data.empty:
        return pd.DataFrame(
            columns=["Date", "Close", "Daily_Return", "Rolling_Volatility"]
        )

    missing_columns = {"Date", "Close"} - set(historical_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Historical data is missing required columns: {missing}")

    clean_data = historical_data.loc[:, ["Date", "Close"]].copy()

    # Convert dates and prices; invalid values become missing values.
    clean_data["Date"] = pd.to_datetime(clean_data["Date"], errors="coerce")
    clean_data["Close"] = pd.to_numeric(clean_data["Close"], errors="coerce")

    # Infinite prices are not valid closing prices.
    clean_data["Close"] = clean_data["Close"].replace([np.inf, -np.inf], np.nan)

    # Remove rows with missing or invalid dates/prices.
    clean_data = clean_data.dropna(subset=["Date", "Close"])
    clean_data = clean_data.loc[clean_data["Close"] > 0]

    # Keep the latest row when the source contains duplicate dates.
    clean_data = (
        clean_data.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if clean_data.empty:
        return pd.DataFrame(
            columns=["Date", "Close", "Daily_Return", "Rolling_Volatility"]
        )

    # pct_change() compares each close price with the previous valid close.
    clean_data["Daily_Return"] = clean_data["Close"].pct_change()
    clean_data["Daily_Return"] = clean_data["Daily_Return"].replace(
        [np.inf, -np.inf], np.nan
    )

    # A 20-day window uses at least 20 returns. Multiply daily volatility
    # by sqrt(252) to express it on an approximate annual basis.
    rolling_volatility = clean_data["Daily_Return"].rolling(
        window=20,
        min_periods=20,
    ).std()
    clean_data["Rolling_Volatility"] = rolling_volatility * np.sqrt(252)

    return clean_data.loc[
        :, ["Date", "Close", "Daily_Return", "Rolling_Volatility"]
    ]


def descriptive_statistics(processed_data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for Close and Daily_Return columns.

    Insufficient or empty data is handled gracefully by returning an empty
    DataFrame rather than raising a calculation error.
    """
    if not isinstance(processed_data, pd.DataFrame):
        raise ValueError("Processed data must be a Pandas DataFrame.")

    if processed_data.empty:
        return pd.DataFrame()

    required_columns = {"Close", "Daily_Return"}
    if not required_columns.issubset(processed_data.columns):
        missing = ", ".join(sorted(required_columns - set(processed_data.columns)))
        raise ValueError(
            f"Processed data is missing required columns: {missing}"
        )

    statistics_data = processed_data.loc[:, ["Close", "Daily_Return"]].copy()
    statistics_data = statistics_data.replace([np.inf, -np.inf], np.nan)

    if statistics_data.empty or statistics_data.dropna(how="all").empty:
        return pd.DataFrame()

    return statistics_data.describe()