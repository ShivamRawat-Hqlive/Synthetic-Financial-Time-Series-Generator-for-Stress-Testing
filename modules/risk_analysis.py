"""Calculate common financial risk metrics for daily returns and prices."""

from typing import Sequence

import numpy as np
import pandas as pd


def calculate_risk_metrics(
    daily_returns: pd.Series | Sequence[float] | np.ndarray,
    price_series: pd.Series | Sequence[float] | np.ndarray,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate annualized risk and downside metrics.

    VaR and CVaR use a loss convention: negative daily returns are losses,
    so VaR and CVaR are returned as positive loss percentages.

    Args:
        daily_returns: Daily percentage returns.
        price_series: Corresponding prices or synthetic prices.
        risk_free_rate: Annualized risk-free rate used by the Sharpe ratio.

    Returns:
        A dictionary containing volatility, VaR, CVaR, maximum drawdown,
        and annualized Sharpe ratio.
    """
    if daily_returns is None:
        clean_returns = pd.Series(dtype=float)
    elif isinstance(daily_returns, pd.Series):
        clean_returns = daily_returns.copy()
    else:
        clean_returns = pd.Series(daily_returns, dtype="float64")

    clean_returns = pd.to_numeric(clean_returns, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    clean_returns = clean_returns.dropna()

    if price_series is None:
        clean_prices = pd.Series(dtype=float)
    elif isinstance(price_series, pd.Series):
        clean_prices = price_series.copy()
    else:
        clean_prices = pd.Series(price_series, dtype="float64")

    clean_prices = pd.to_numeric(clean_prices, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    clean_prices = clean_prices.dropna()

    try:
        risk_free_rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk_free_rate must be a finite number.") from exc

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be a finite number.")

    if clean_returns.empty:
        return {
            "Annualized_Volatility": np.nan,
            "VaR_95": np.nan,
            "CVaR_95": np.nan,
            "Maximum_Drawdown": np.nan,
            "Annualized_Sharpe_Ratio": np.nan,
        }

    # Daily standard deviation is the starting point for annual volatility.
    daily_std = (
        float(clean_returns.std(ddof=1))
        if len(clean_returns) > 1
        else 0.0
    )
    if not np.isfinite(daily_std) or daily_std < 0:
        daily_std = 0.0
    annualized_volatility = daily_std * np.sqrt(252)

    # Historical VaR uses the 5th percentile of daily returns. Convert the
    # negative return at that percentile into a positive loss percentage.
    fifth_percentile = float(clean_returns.quantile(0.05))
    var_95 = max(-fifth_percentile, 0.0)
    if var_95 == 0:
        var_95 = 0.0

    # CVaR averages returns in the worst 5% tail. Observations at or below
    # the VaR cutoff represent losses exceeding the VaR threshold.
    tail_returns = clean_returns[clean_returns <= fifth_percentile]
    cvar_95 = (
        max(float(-tail_returns.mean()), 0.0)
        if not tail_returns.empty
        else var_95
    )
    if cvar_95 == 0:
        cvar_95 = 0.0

    # Drawdown is negative when price declines: running maximum minus current
    # price, divided by the running maximum. The reported maximum is positive.
    maximum_drawdown = np.nan
    positive_prices = clean_prices[clean_prices > 0]
    if not positive_prices.empty:
        running_maximum = positive_prices.cummax()
        drawdowns = positive_prices / running_maximum - 1.0
        maximum_drawdown = float(-drawdowns.min())
        if not np.isfinite(maximum_drawdown) or maximum_drawdown < 0:
            maximum_drawdown = np.nan
        elif maximum_drawdown == 0:
            maximum_drawdown = 0.0

    # Sharpe uses annualized mean return minus an annualized risk-free rate.
    annualized_mean = float(clean_returns.mean()) * 252
    if annualized_volatility > 0 and np.isfinite(annualized_volatility):
        sharpe_ratio = (
            annualized_mean - risk_free_rate
        ) / annualized_volatility
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = np.nan
    else:
        sharpe_ratio = 0.0

    return {
        "Annualized_Volatility": annualized_volatility,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Maximum_Drawdown": maximum_drawdown,
        "Annualized_Sharpe_Ratio": sharpe_ratio,
    }