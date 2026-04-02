"""Compute aggregated volatility smile metrics from per-strike smile data.

For each (date, tenor) group from build_iv_smile output, computes:

    atm_vol  - mid_iv at strike closest to ATM (|moneyness| min)
    rr25     - IV(25D call) - IV(25D put)    [negative = put skew, typical for USD/RUB]
    bf25     - [IV(25D call) + IV(25D put)] / 2 - atm_vol
    rr10     - IV(10D call) - IV(10D put)
    skew     - OLS slope of mid_iv vs moneyness (sign: +ve = higher IV for high strikes)
    n_strikes - number of valid strikes used

Interpolation uses Black-Scholes forward delta (delta_call column from smile df).
Put-side entries are mapped to call-equivalent delta: delta_call_equiv = 1 + delta_put
so the full curve runs from ~0 (deep OTM put) to ~1 (deep OTM call).

Quality filter for RR/BF metrics: requires >= 4 valid strikes AND the delta range
must bracket the target delta (i.e. min delta_call <= target <= max delta_call).
If quality filter fails, the metric is NaN (shown as 'n/a' by the caller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_MIN_STRIKES_FOR_METRICS = 4
_TARGET_DELTA_25 = 0.25
_TARGET_DELTA_10 = 0.10


def _interp_iv_at_delta(
    delta_call_arr: np.ndarray,
    mid_iv_arr: np.ndarray,
    target_delta: float,
) -> float | None:
    """Linearly interpolate mid_iv at a given call-equivalent delta.

    delta_call_arr must be sorted ascending. Returns None if the target delta
    is outside the available range.
    """
    if len(delta_call_arr) < 2:
        return None
    if target_delta < delta_call_arr[0] or target_delta > delta_call_arr[-1]:
        return None
    return float(np.interp(target_delta, delta_call_arr, mid_iv_arr))


def _compute_group_metrics(group: pd.DataFrame) -> dict:
    """Compute smile metrics for a single (date, tenor) group."""
    # Require delta_call column (computed in iv_smile_builder)
    group = group.dropna(subset=['delta_call', 'mid_iv']).copy()
    n_strikes = len(group)

    # ATM vol: strike with smallest |moneyness|
    atm_idx = group['moneyness'].abs().idxmin()
    atm_vol = float(group.loc[atm_idx, 'mid_iv'])

    if n_strikes < _MIN_STRIKES_FOR_METRICS:
        return {
            'atm_vol': atm_vol,
            'rr25': float('nan'),
            'bf25': float('nan'),
            'rr10': float('nan'),
            'skew': float('nan'),
            'n_strikes': n_strikes,
        }

    # Build full delta_call curve: call side uses delta_call directly,
    # put side uses call-equivalent = 1 + delta_put (so it maps to (0, 0.5) range).
    # For each strike we pick the best available delta:
    #   - if delta_call >= 0.5 → skip (deep ITM call, not useful for OTM side)
    #   - if delta_call <= 0.5 → use directly (covers ATM and OTM call side)
    # For put-equivalent we need rows where delta_call < 0.5:
    #   call-equiv delta for put side = 1 + delta_put = delta_call (same row, just labeling)
    # In practice delta_call already covers the full (0, 1) range for all strikes.
    # We sort by delta_call and interpolate on the full curve.
    sorted_group = group.sort_values('delta_call').reset_index(drop=True)
    dc = sorted_group['delta_call'].values.astype(float)
    iv = sorted_group['mid_iv'].values.astype(float)

    # Check bracketing for 25D
    can_interp_25 = dc.min() <= _TARGET_DELTA_25 <= dc.max() and (1 - _TARGET_DELTA_25) <= dc.max()
    # Check bracketing for 10D
    can_interp_10 = dc.min() <= _TARGET_DELTA_10 <= dc.max() and (1 - _TARGET_DELTA_10) <= dc.max()

    rr25 = float('nan')
    bf25 = float('nan')
    rr10 = float('nan')

    if can_interp_25:
        iv_25c = _interp_iv_at_delta(dc, iv, _TARGET_DELTA_25)      # 25D call = low delta (OTM call)
        iv_25p = _interp_iv_at_delta(dc, iv, 1 - _TARGET_DELTA_25)  # 25D put equiv = high call delta
        if iv_25c is not None and iv_25p is not None:
            # RR25 convention: positive means OTM call more expensive (call skew)
            # For USD/RUB we expect negative (put skew = OTM puts pricier)
            rr25 = iv_25c - iv_25p
            bf25 = (iv_25c + iv_25p) / 2.0 - atm_vol

    if can_interp_10:
        iv_10c = _interp_iv_at_delta(dc, iv, _TARGET_DELTA_10)
        iv_10p = _interp_iv_at_delta(dc, iv, 1 - _TARGET_DELTA_10)
        if iv_10c is not None and iv_10p is not None:
            rr10 = iv_10c - iv_10p

    # OLS skew: slope of mid_iv vs moneyness (log(K/F))
    moneyness = sorted_group['moneyness'].values.astype(float)
    if len(moneyness) >= 2 and moneyness.std() > 0:
        coeffs = np.polyfit(moneyness, iv, 1)
        skew = float(coeffs[0])
    else:
        skew = float('nan')

    return {
        'atm_vol': atm_vol,
        'rr25': rr25,
        'bf25': bf25,
        'rr10': rr10,
        'skew': skew,
        'n_strikes': n_strikes,
    }


def compute_smile_metrics(smile_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated smile metrics from per-strike smile DataFrame.

    Parameters
    ----------
    smile_df:
        Output of build_iv_smile — one row per (date, tenor, strike).

    Returns
    -------
    pd.DataFrame with one row per (date, tenor):
        date, tenor, atm_vol, rr25, bf25, rr10, skew, n_strikes
    """
    if smile_df.empty:
        return pd.DataFrame(
            columns=['date', 'tenor', 'atm_vol', 'rr25', 'bf25', 'rr10', 'skew', 'n_strikes']
        )

    records = []
    for (date, tenor), group in smile_df.groupby(['date', 'tenor']):
        metrics = _compute_group_metrics(group)
        records.append({'date': date, 'tenor': tenor, **metrics})

    result = pd.DataFrame(records)
    result = result.sort_values(['date', 'tenor']).reset_index(drop=True)
    return result
