"""Download historical financial data using Yahoo Finance."""

from typing import Optional

import pandas as pd
import yfinance as yf


SUPPORTED_ASSETS = {
    "SPY": "Stock",
    "TLT": "Bond",
    "EURUSD=X": "Forex",
}


def download_historical_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download a supported asset and return a clean DataFrame.

    The returned DataFrame contains only two columns:
    - Date
    - Close

    Args:
        ticker: One of SPY, TLT, or EURUSD=X.
        start_date: Start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format.

    Returns:
        A Pandas DataFrame with Date and Close columns.

    Raises:
        ValueError: If the ticker or date format is invalid.
        RuntimeError: If Yahoo Finance cannot download the data.
    """
    if ticker not in SUPPORTED_ASSETS:
        supported = ", ".join(SUPPORTED_ASSETS)
        raise ValueError(
            f"Unsupported ticker '{ticker}'. Choose one of: {supported}"
        )

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be valid values in YYYY-MM-DD format.") from exc

    if start >= end:
        raise ValueError("start_date must be earlier than end_date.")

    try:
        data = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance failed while downloading '{ticker}'. "
            "Check your internet connection or try again later."
        ) from exc

    if data is None or data.empty:
        raise RuntimeError(
            f"No data was returned for '{ticker}'. "
            "The ticker or date range may be unavailable."
        )

    # Recent yfinance versions can return a MultiIndex for downloaded columns.
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            close_prices = data[("Close", ticker)].copy()
        elif "Close" in data.columns.get_level_values(0):
            close_prices = data.loc[:, "Close"].iloc[:, 0].copy()
        else:
            raise RuntimeError(
                f"Yahoo Finance did not return a Close column for '{ticker}'."
            )
    elif "Close" in data.columns:
        close_prices = data["Close"].copy()
    else:
        raise RuntimeError(
            f"Yahoo Finance did not return a Close column for '{ticker}'."
        )

    clean_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(data.index).to_numpy(),
            "Close": pd.to_numeric(close_prices.to_numpy(), errors="coerce"),
        }
    )

    # Forward-fill small gaps, then remove any rows that still have missing Close values.
    clean_data["Close"] = clean_data["Close"].ffill()
    clean_data = clean_data.dropna(subset=["Date", "Close"])
    clean_data = clean_data.sort_values("Date").drop_duplicates(
        subset=["Date"], keep="last"
    )

    if clean_data.empty:
        raise RuntimeError(
            f"No valid closing prices were available for '{ticker}'."
        )

    return clean_data.reset_index(drop=True)