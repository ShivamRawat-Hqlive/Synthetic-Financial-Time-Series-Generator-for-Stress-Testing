"""Statistical comparison tools for historical and synthetic returns."""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _clean_returns(
    returns: pd.Series | Sequence[float] | np.ndarray,
) -> pd.Series:
    """Convert return input to a finite Pandas Series."""
    if isinstance(returns, pd.Series):
        clean = returns.copy()
    else:
        if returns is None:
            clean = pd.Series(dtype=float)
        else:
            clean = pd.Series(returns, dtype="float64")

    clean = pd.to_numeric(clean, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return clean.dropna().reset_index(drop=True)


def compare_statistics(
    historical_returns: pd.Series | Sequence[float] | np.ndarray,
    synthetic_returns: pd.Series | Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Compare summary statistics for historical and synthetic returns.

    The comparison includes count-independent moment estimates such as mean,
    sample standard deviation, minimum, maximum, skewness, and Fisher
    kurtosis. Kurtosis is not a guarantee of realistic return behavior.
    """
    historical = _clean_returns(historical_returns)
    synthetic = _clean_returns(synthetic_returns)

    def calculate_statistics(data: pd.Series) -> dict[str, float]:
        if data.empty:
            return {
                "Mean": np.nan,
                "Standard Deviation": np.nan,
                "Minimum Return": np.nan,
                "Maximum Return": np.nan,
                "Skewness": np.nan,
                "Kurtosis": np.nan,
            }

        sample_std = float(data.std(ddof=1)) if len(data) > 1 else 0.0

        return {
            "Mean": float(data.mean()),
            "Standard Deviation": sample_std,
            "Minimum Return": float(data.min()),
            "Maximum Return": float(data.max()),
            "Skewness": (
                float(skew(data.to_numpy(), bias=False, nan_policy="omit"))
                if len(data) > 2
                else np.nan
            ),
            "Kurtosis": (
                float(kurtosis(data.to_numpy(), fisher=True, bias=False,
                               nan_policy="omit"))
                if len(data) > 3
                else np.nan
            ),
        }

    historical_statistics = calculate_statistics(historical)
    synthetic_statistics = calculate_statistics(synthetic)

    return pd.DataFrame(
        {
            "Metric": list(historical_statistics.keys()),
            "Historical": list(historical_statistics.values()),
            "Synthetic": list(synthetic_statistics.values()),
        }
    )


def compare_return_distributions(
    historical_returns: pd.Series | Sequence[float] | np.ndarray,
    synthetic_returns: pd.Series | Sequence[float] | np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """Prepare cleaned return Series for distribution plots."""
    historical = _clean_returns(historical_returns)
    synthetic = _clean_returns(synthetic_returns)
    return historical, synthetic


def compare_volatility(
    historical_returns: pd.Series | Sequence[float] | np.ndarray,
    synthetic_returns: pd.Series | Sequence[float] | np.ndarray,
    window: int = 20,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate annualized rolling volatility for both return datasets."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be a positive integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    if isinstance(trading_days, bool) or not isinstance(
        trading_days, (int, np.integer)
    ):
        raise ValueError("trading_days must be a positive integer.")
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    historical = _clean_returns(historical_returns)
    synthetic = _clean_returns(synthetic_returns)

    def rolling_volatility(data: pd.Series) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        # std(ddof=1) uses the sample standard deviation.
        return data.rolling(window=window, min_periods=window).std(
            ddof=1
        ) * np.sqrt(trading_days)

    return pd.DataFrame(
        {
            "Historical_Volatility": rolling_volatility(historical),
            "Synthetic_Volatility": rolling_volatility(synthetic),
        }
    )