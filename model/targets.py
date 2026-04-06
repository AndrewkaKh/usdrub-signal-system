"""Target variant definitions for IV forecasting.

Three target variants compared against the baseline raw_delta:

  A. raw_delta          — iv(t+1) - iv(t)
  B. log_return_iv      — log(iv(t+1) / iv(t))
  C. sigma_normalized   — (iv(t+1) - iv(t)) / rolling_std(past_delta, 20)

All rolling statistics are computed strictly causally (past data only).
For each variant, an inverse transform restores predictions to the
raw delta-IV space for unified metric comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Variant identifiers ───────────────────────────────────────────────────────

RAW_DELTA = 'raw_delta'
LOG_RETURN = 'log_return_iv'
SIGMA_NORM = 'sigma_normalized_delta'

ALL_TARGETS: list[str] = [RAW_DELTA, LOG_RETURN, SIGMA_NORM]

# Column names written into the DataFrame by add_target_variants()
TARGET_COL_MAP: dict[str, str] = {
    RAW_DELTA:  'target_raw_delta',
    LOG_RETURN: 'target_log_return_iv',
    SIGMA_NORM: 'target_sigma_normalized',
}

# Human-readable description for each variant
TARGET_LABELS: dict[str, str] = {
    RAW_DELTA:  'Raw Delta IV  (iv[t+1]-iv[t])',
    LOG_RETURN: 'Log-return IV (log(iv[t+1]/iv[t]))',
    SIGMA_NORM: 'Sigma-norm    (delta/rolling_std20)',
}


# ── Target construction ───────────────────────────────────────────────────────

def add_target_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Append all target variant columns to *df* (does not modify in-place).

    Columns added
    -------------
    target_raw_delta          : iv(t+1) - iv(t)
    target_log_return_iv      : log(iv(t+1) / iv(t))
    target_sigma_normalized   : (iv(t+1) - iv(t)) / rolling_std(past_delta, 20)
    iv_rolling_std_20         : rolling std of past daily iv changes (used as
                                inverse-transform denominator at inference time)

    Causality guarantee
    -------------------
    * ``iv_rolling_std_20`` is computed from ``iv(t) - iv(t-1)`` (diff()),
      which is fully known at time *t*, then rolled over the 20 preceding rows.
    * All shift(-1) operations create the *target* (future value) — they are
      never used as features.
    """
    df = df.copy()

    # ── A: raw delta ──────────────────────────────────────────────────────────
    df['target_raw_delta'] = df['iv_1m'].shift(-1) - df['iv_1m']

    # ── B: log-return ─────────────────────────────────────────────────────────
    iv_next = df['iv_1m'].shift(-1).clip(lower=1e-6)
    iv_curr = df['iv_1m'].clip(lower=1e-6)
    df['target_log_return_iv'] = np.log(iv_next / iv_curr)

    # ── C: sigma-normalised ───────────────────────────────────────────────────
    # past_delta = iv(t) - iv(t-1): strictly causal, known at time t
    past_delta = df['iv_1m'].diff()
    rolling_std = past_delta.rolling(20, min_periods=5).std()
    df['iv_rolling_std_20'] = rolling_std  # retained for inference inverse-transform

    future_delta = df['iv_1m'].shift(-1) - df['iv_1m']
    # clip denominator away from zero to avoid inf targets
    df['target_sigma_normalized'] = future_delta / rolling_std.clip(lower=1e-8)

    return df


# ── Inverse transforms ────────────────────────────────────────────────────────

def inverse_transform(
    target_name: str,
    predictions: np.ndarray,
    current_iv: np.ndarray,
    rolling_std: np.ndarray | None = None,
) -> np.ndarray:
    """Convert model predictions back to predicted *delta IV* (absolute units).

    Parameters
    ----------
    target_name  : one of RAW_DELTA, LOG_RETURN, SIGMA_NORM
    predictions  : model output array (in target space)
    current_iv   : array of iv_1m values at prediction time
    rolling_std  : iv_rolling_std_20 values at prediction time
                   (required for SIGMA_NORM only)

    Returns
    -------
    np.ndarray of predicted delta IV (same scale as raw_delta target)
    """
    if target_name == RAW_DELTA:
        return predictions.copy()

    elif target_name == LOG_RETURN:
        # pred_iv = current_iv * exp(log_return)  →  delta = pred_iv - current_iv
        pred_iv = current_iv * np.exp(predictions)
        return pred_iv - current_iv

    elif target_name == SIGMA_NORM:
        if rolling_std is None:
            raise ValueError('rolling_std is required for sigma_normalized inverse transform')
        # pred_delta = predicted_z * rolling_std
        return predictions * rolling_std

    else:
        raise ValueError(f'Unknown target variant: {target_name!r}')


def inverse_transform_to_iv(
    target_name: str,
    predictions: np.ndarray,
    current_iv: np.ndarray,
    rolling_std: np.ndarray | None = None,
) -> np.ndarray:
    """Return predicted next-day IV level (not delta) for each variant."""
    delta = inverse_transform(target_name, predictions, current_iv, rolling_std)
    return current_iv + delta
