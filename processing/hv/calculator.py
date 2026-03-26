from __future__ import annotations

import math
from typing import Any

import pandas as pd

from processing.iv.utils import first_existing_column, to_float

from .config import DATE_COLUMN_CANDIDATES, PRICE_COLUMN_CANDIDATES


def prepare_price_series(candles: pd.DataFrame) -> tuple[pd.Series, str | None, str | None]:
    if candles.empty:
        return pd.Series(dtype=float), None, 'price_series_empty'

    working = candles.copy()
    date_col = first_existing_column(working, DATE_COLUMN_CANDIDATES)
    if date_col is None:
        return pd.Series(dtype=float), None, 'date_column_missing'

    working['hv_date'] = pd.to_datetime(working[date_col], errors='coerce')
    working = working.dropna(subset=['hv_date']).copy()
    if working.empty:
        return pd.Series(dtype=float), None, 'date_column_missing'

    for price_col in PRICE_COLUMN_CANDIDATES:
        if price_col not in working.columns:
            continue

        numeric = working[price_col].apply(to_float)
        valid_mask = numeric.notna() & (numeric > 0)
        if not valid_mask.any():
            continue

        prices = pd.Series(numeric.loc[valid_mask].astype(float).values, index=working.loc[valid_mask, 'hv_date'])
        prices = prices[~prices.index.duplicated(keep='last')].sort_index()
        if not prices.empty:
            return prices, str(price_col).lower(), None

    return pd.Series(dtype=float), None, 'price_field_missing'


def compute_log_returns(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)

    log_prices = prices.astype(float).apply(math.log)
    returns = log_prices.diff().dropna()
    returns.name = 'log_return'
    return returns


def calculate_hv_from_prices(
    prices: pd.Series,
    window: int,
    annualization_days: int,
) -> tuple[float | None, pd.Series, str | None]:
    returns = compute_log_returns(prices)
    if len(returns) < window:
        return None, returns, 'insufficient_history'

    rolling_std = returns.rolling(window=window).std(ddof=1)
    hv_value = rolling_std.iloc[-1]

    if pd.isna(hv_value):
        return None, returns, 'insufficient_history'
    if not math.isfinite(float(hv_value)) or float(hv_value) <= 0:
        return None, returns, 'hv_not_finite'

    annualized_hv = float(hv_value) * math.sqrt(float(annualization_days))
    if not math.isfinite(annualized_hv) or annualized_hv <= 0:
        return None, returns, 'hv_not_finite'

    return annualized_hv, returns, None
