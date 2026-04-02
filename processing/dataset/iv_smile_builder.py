"""Build per-strike implied volatility smile for each (date, tenor) pair.

Reuses series / futures selection logic from iv_daily_builder but keeps
all available strikes instead of selecting a single ATM one.

Output columns:
    date, tenor, underlying_secid, expiry, days_to_expiry, t,
    futures_price, strike, moneyness, call_price, put_price,
    call_iv, put_iv, mid_iv, delta_call, delta_put, status

Status values:
    ok          - both call_iv and put_iv valid
    call_only   - only call_iv valid
    put_only    - only put_iv valid
    invalid_iv  - neither side valid (row excluded from output)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from processing.iv.calculator import calculate_option_iv
from processing.iv.utils import compute_time_to_expiry

# Reuse shared helpers from iv_daily_builder (no modification to that module)
from .iv_daily_builder import (
    _choose_series,
    _choose_underlying_future,
    _load_contract_candidates,
    _load_futures_raw,
    _prepare_futures,
    _prepare_options,
)


def _bs_forward_delta(
    futures_price: float,
    strike: float,
    t: float,
    sigma: float,
    option_type: str,  # 'call' or 'put'
) -> float | None:
    """Black-Scholes forward delta (no premium adjustment).

    Delta_call = N(d1)
    Delta_put  = N(d1) - 1

    Returns None if inputs are invalid.
    """
    if t <= 0 or sigma <= 0 or futures_price <= 0 or strike <= 0:
        return None
    try:
        d1 = (math.log(futures_price / strike) + 0.5 * sigma ** 2 * t) / (sigma * math.sqrt(t))
    except (ValueError, ZeroDivisionError):
        return None
    if option_type == 'call':
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def _build_smile_rows(
    date_value: pd.Timestamp,
    target_tenor: str,
    options_group: pd.DataFrame,
    futures_price: float,
    underlying_secid: str,
) -> list[dict]:
    """Compute per-strike IV rows for one (date, tenor) group."""
    expiry = options_group['expiry'].dropna().iloc[0]
    days_to_expiry = int((expiry - date_value).days)
    t = compute_time_to_expiry(date_value.to_pydatetime(), expiry.to_pydatetime())

    if t <= 0:
        return []

    strikes = sorted(options_group['strike'].dropna().unique())
    rows = []

    for strike in strikes:
        strike_rows = options_group[options_group['strike'] == strike]

        call_rows = strike_rows[strike_rows['option_type'] == 'call'].copy()
        put_rows = strike_rows[strike_rows['option_type'] == 'put'].copy()

        call_row = (
            call_rows.sort_values(by=['num_trades', 'open_interest'], ascending=[False, False]).iloc[0]
            if not call_rows.empty else None
        )
        put_row = (
            put_rows.sort_values(by=['num_trades', 'open_interest'], ascending=[False, False]).iloc[0]
            if not put_rows.empty else None
        )

        call_price = float(call_row['price_used']) if call_row is not None else None
        put_price = float(put_row['price_used']) if put_row is not None else None

        call_iv, _ = (
            calculate_option_iv(call_price, futures_price, strike, t, 'call')
            if call_price is not None else (None, None)
        )
        put_iv, _ = (
            calculate_option_iv(put_price, futures_price, strike, t, 'put')
            if put_price is not None else (None, None)
        )

        # Determine mid_iv and status
        if call_iv is not None and put_iv is not None:
            mid_iv = (call_iv + put_iv) / 2.0
            status = 'ok'
        elif call_iv is not None:
            mid_iv = call_iv
            status = 'call_only'
        elif put_iv is not None:
            mid_iv = put_iv
            status = 'put_only'
        else:
            # invalid_iv: exclude from output
            continue

        moneyness = math.log(strike / futures_price)
        delta_call = _bs_forward_delta(futures_price, strike, t, mid_iv, 'call')
        delta_put = _bs_forward_delta(futures_price, strike, t, mid_iv, 'put')

        rows.append({
            'date': date_value.strftime('%Y-%m-%d'),
            'tenor': target_tenor,
            'underlying_secid': underlying_secid,
            'expiry': expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': days_to_expiry,
            't': round(t, 6),
            'futures_price': round(futures_price, 2),
            'strike': float(strike),
            'moneyness': round(moneyness, 6),
            'call_price': round(call_price, 4) if call_price is not None else None,
            'put_price': round(put_price, 4) if put_price is not None else None,
            'call_iv': round(call_iv, 6) if call_iv is not None else None,
            'put_iv': round(put_iv, 6) if put_iv is not None else None,
            'mid_iv': round(mid_iv, 6),
            'delta_call': round(delta_call, 4) if delta_call is not None else None,
            'delta_put': round(delta_put, 4) if delta_put is not None else None,
            'status': status,
        })

    return rows


def build_iv_smile(connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Compute per-strike IV smile for all (date, tenor) pairs in date range.

    Parameters
    ----------
    connection:
        SQLite connection (read-only is sufficient).
    start_date, end_date:
        ISO date strings, e.g. '2025-01-01'.

    Returns
    -------
    pd.DataFrame with one row per (date, tenor, strike).
    Empty DataFrame if no data found.
    """
    options_raw = _load_contract_candidates(connection, start_date, end_date)
    futures_raw = _load_futures_raw(connection, start_date, end_date)

    options = _prepare_options(options_raw)
    futures = _prepare_futures(futures_raw)

    if options.empty:
        return pd.DataFrame()

    all_rows: list[dict] = []

    for (date_value, target_tenor), options_group in options.groupby(['date', 'target_tenor']):
        selected_options = _choose_series(options_group, target_tenor)
        if selected_options.empty:
            continue

        futures_group = futures[futures['date'] == date_value].copy()
        underlying_secid, futures_price = _choose_underlying_future(selected_options, futures_group)
        if underlying_secid is None or futures_price is None:
            continue

        rows = _build_smile_rows(
            date_value=date_value,
            target_tenor=target_tenor,
            options_group=selected_options,
            futures_price=futures_price,
            underlying_secid=underlying_secid,
        )
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(['date', 'tenor', 'strike']).reset_index(drop=True)
    return df
