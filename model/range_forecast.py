from __future__ import annotations

import math
from typing import Literal

# z-scores for common confidence levels (two-sided log-normal range)
_Z_SCORES: dict[float, float] = {
    0.68: 1.000,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

TRADING_DAYS_PER_YEAR = 252


def compute_spot_range(
    predicted_iv: float,
    spot_price: float,
    confidence: float = 0.90,
    horizon_days: int = 1,
) -> dict:
    """Compute expected spot price range from annualised implied volatility.

    Uses the log-normal (Black-Scholes) assumption:
        σ_period = σ_annual * sqrt(T)
        upper    = spot * exp(+z * σ_period)
        lower    = spot * exp(-z * σ_period)

    Parameters
    ----------
    predicted_iv:
        Annualised implied volatility as a decimal (e.g. 0.15 for 15 %).
    spot_price:
        Current spot price of USD/RUB.
    confidence:
        Confidence level for the range. Supported: 0.68, 0.90, 0.95, 0.99.
    horizon_days:
        Forecast horizon in trading days (default 1 = next trading day).

    Returns
    -------
    dict with keys: lower, upper, move_pct, confidence, z_score,
                    predicted_iv, spot_price, horizon_days.
    """
    if predicted_iv <= 0:
        raise ValueError(f'predicted_iv must be positive, got {predicted_iv}')
    if spot_price <= 0:
        raise ValueError(f'spot_price must be positive, got {spot_price}')
    if confidence not in _Z_SCORES:
        supported = sorted(_Z_SCORES.keys())
        raise ValueError(f'confidence must be one of {supported}, got {confidence}')

    z = _Z_SCORES[confidence]
    t = horizon_days / TRADING_DAYS_PER_YEAR
    sigma_period = predicted_iv * math.sqrt(t)

    upper = spot_price * math.exp(+z * sigma_period)
    lower = spot_price * math.exp(-z * sigma_period)
    move_pct = round(z * sigma_period * 100, 4)

    return {
        'predicted_iv': round(predicted_iv, 6),
        'spot_price': round(spot_price, 4),
        'lower': round(lower, 4),
        'upper': round(upper, 4),
        'move_pct': move_pct,
        'confidence': confidence,
        'z_score': z,
        'horizon_days': horizon_days,
    }
