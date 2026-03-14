from __future__ import annotations

import math
from typing import Any

from .utils import normalize_option_type, to_float

try:
    from py_vollib.black.implied_volatility import implied_volatility_of_undiscounted_option_price
except ImportError:
    implied_volatility_of_undiscounted_option_price = None


def extract_option_price(marketdata: dict[str, Any]) -> tuple[float | None, str | None]:
    bid = to_float(marketdata.get('BID'))
    ask = to_float(marketdata.get('OFFER'))
    last = to_float(marketdata.get('LAST'))
    settle = to_float(marketdata.get('SETTLEPRICE'))

    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0, 'mid'
    if last is not None and last > 0:
        return last, 'last'
    if settle is not None and settle > 0:
        return settle, 'settleprice'
    return None, None


def calculate_option_iv(
    option_price: float,
    futures_price: float,
    strike: float,
    t: float,
    option_type: str,
) -> tuple[float | None, str | None]:
    flag = normalize_option_type(option_type)
    if implied_volatility_of_undiscounted_option_price is None:
        return None, 'py_vollib_missing'
    if flag is None:
        return None, 'invalid_option_type'
    if option_price <= 0:
        return None, 'invalid_option_price'
    if futures_price <= 0:
        return None, 'invalid_futures_price'
    if strike <= 0:
        return None, 'invalid_strike'
    if t <= 0:
        return None, 'invalid_time_to_expiry'

    try:
        iv = implied_volatility_of_undiscounted_option_price(option_price, futures_price, strike, t, flag)
    except Exception as exc:
        return None, f'iv_failed:{type(exc).__name__}'

    if iv is None or not math.isfinite(iv) or iv <= 0:
        return None, 'iv_not_finite'
    return float(iv), None