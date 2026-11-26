"""Apply market stress scenarios to synthetic financial data."""

import numpy as np
import pandas as pd


def apply_stress_scenario(
    synthetic_data: pd.DataFrame,
    scenario_name: str,
    volatility_multiplier: float = 2.0,
    crash_days: int = 5,
    crash_shock: float = -0.10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Apply a selected market scenario and recalculate synthetic prices.

    Supported scenarios:
    - Normal Market
    - High Volatility
    - Market Crash

    Args:
        synthetic_data: DataFrame with Date, Synthetic_Return, and
            Synthetic_Price columns.
        scenario_name: One of the three supported scenario names.
        volatility_multiplier: Positive factor used to increase volatility.
        crash_days: Number of negative-shock days for a market crash.
        crash_shock: Negative return added to selected crash days.
        random_seed: Seed used for reproducible scenario calculations.

    Returns:
        A DataFrame containing Date, adjusted Synthetic_Return, and
        recalculated Synthetic_Price.
    """
    if not isinstance(synthetic_data, pd.DataFrame):
        raise ValueError("Synthetic data must be a Pandas DataFrame.")

    supported_scenarios = {
        "Normal Market",
        "High Volatility",
        "Market Crash",
    }
    if scenario_name not in supported_scenarios:
        supported = ", ".join(sorted(supported_scenarios))
        raise ValueError(
            f"Invalid scenario '{scenario_name}'. Choose one of: {supported}"
        )

    if synthetic_data.empty:
        return pd.DataFrame(
            columns=["Date", "Synthetic_Return", "Synthetic_Price"]
        )

    required_columns = {"Date", "Synthetic_Return", "Synthetic_Price"}
    missing_columns = required_columns - set(synthetic_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Synthetic data is missing required columns: {missing}"
        )

    clean_data = synthetic_data.loc[
        :, ["Date", "Synthetic_Return", "Synthetic_Price"]
    ].copy()
    clean_data["Date"] = pd.to_datetime(clean_data["Date"], errors="coerce")
    clean_data["Synthetic_Return"] = pd.to_numeric(
        clean_data["Synthetic_Return"], errors="coerce"
    )

    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
    clean_data = clean_data.dropna(subset=["Date", "Synthetic_Return"])

    if clean_data.empty:
        return pd.DataFrame(
            columns=["Date", "Synthetic_Return", "Synthetic_Price"]
        )

    # Preserve the original synthetic returns for the normal market.
    adjusted_returns = clean_data["Synthetic_Return"].to_numpy(dtype=float).copy()

    rng = np.random.default_rng(random_seed)

    if isinstance(volatility_multiplier, bool) or not np.isscalar(
        volatility_multiplier
    ):
        raise ValueError("volatility_multiplier must be a positive number.")
    try:
        volatility_multiplier = float(volatility_multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "volatility_multiplier must be a positive number."
        ) from exc

    if not np.isfinite(volatility_multiplier) or volatility_multiplier <= 0:
        raise ValueError("volatility_multiplier must be a positive number.")

    if scenario_name in {"High Volatility", "Market Crash"}:
        valid_returns = adjusted_returns[np.isfinite(adjusted_returns)]
        if valid_returns.size > 0:
            historical_mean = float(valid_returns.mean())
            historical_std = float(valid_returns.std(ddof=1))
            if not np.isfinite(historical_std) or historical_std < 0:
                historical_std = 0.0
        else:
            historical_mean = 0.0
            historical_std = 0.0

        # Resample around the existing mean with a larger standard deviation.
        adjusted_returns = rng.normal(
            loc=historical_mean,
            scale=historical_std * volatility_multiplier,
            size=len(adjusted_returns),
        )

    if scenario_name == "Market Crash":
        if isinstance(crash_days, bool) or not isinstance(
            crash_days, (int, np.integer)
        ):
            raise ValueError("crash_days must be a non-negative integer.")
        if crash_days < 0:
            raise ValueError("crash_days must be a non-negative integer.")

        if isinstance(crash_shock, bool) or not np.isscalar(crash_shock):
            raise ValueError("crash_shock must be a negative number.")
        try:
            crash_shock = float(crash_shock)
        except (TypeError, ValueError) as exc:
            raise ValueError("crash_shock must be a negative number.") from exc

        if not np.isfinite(crash_shock) or crash_shock >= 0:
            raise ValueError("crash_shock must be a negative number.")

        # Select distinct days so each crash shock is applied once.
        usable_crash_days = min(crash_days, len(adjusted_returns))
        if usable_crash_days > 0:
            selected_days = rng.choice(
                len(adjusted_returns),
                size=usable_crash_days,
                replace=False,
            )

            # A negative shock makes the sampled return more negative.
            adjusted_returns[selected_days] = (
                adjusted_returns[selected_days] + crash_shock
            )

    # Find a positive starting price. This is recalculated instead of copied.
    original_prices = pd.to_numeric(
        clean_data["Synthetic_Price"], errors="coerce"
    )
    valid_original_prices = original_prices[
        np.isfinite(original_prices) & (original_prices > 0)
    ]
    if valid_original_prices.empty:
        raise ValueError("Synthetic data must contain at least one positive price.")

    starting_price = float(valid_original_prices.iloc[0])
    recalculated_prices = np.empty(len(adjusted_returns), dtype=float)
    previous_price = starting_price

    for index, adjusted_return in enumerate(adjusted_returns):
        previous_price = previous_price * (1.0 + adjusted_return)

        # Prevent impossible non-positive or infinite prices.
        if not np.isfinite(previous_price) or previous_price <= 0:
            previous_price = np.finfo(float).eps

        recalculated_prices[index] = previous_price

    return pd.DataFrame(
        {
            "Date": clean_data["Date"].reset_index(drop=True),
            "Synthetic_Return": adjusted_returns,
            "Synthetic_Price": recalculated_prices,
        }
    )