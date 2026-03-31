from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import (
    TARGET_COL,
    FEATURE_COLS,
    LAG_COLS,
    ROLLING_COLS,
    CALENDAR_COLS,
)

# Time-based split boundaries
TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-07-31'
TEST_START = '2025-08-01'

EXTERNAL_FEATURES_PATH = Path(__file__).parents[1] / 'data' / 'exports' / 'external_features.csv'
CBR_MEETINGS_PATH = Path(__file__).parents[1] / 'data' / 'exports' / 'cbr_meetings.csv'


def prepare_dataset(csv_path: str) -> pd.DataFrame:
    """Load model_dataset_daily.csv and build all derived features.

    Filtering strategy:
    - Keep rows where iv_status_1m == 'ok' and iv_1m is not NaN (primary filter).
    - 3M columns are kept where available; rows without 3M data get NaN in 3M
      columns — CatBoost handles missing values natively so we don't drop them.
    - Rows with NaN in required 1M lag columns are dropped after lag creation
      (first ~5 rows of each contiguous segment).

    Returns a DataFrame sorted by date with all features and the target column.
    """
    df = pd.read_csv(csv_path, sep=';', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Primary filter: require valid 1M IV
    df = df[df['iv_status_1m'] == 'ok'].copy()
    df = df[df['iv_1m'].notna()].copy()

    # Nullify 3M columns where 3M status is not 'ok'
    mask_3m_bad = df['iv_status_3m'] != 'ok'
    for col in ['iv_3m', 'hv_3m', 'spread_3m', 'ts_3m_1m', 'days_to_expiry_3m']:
        if col in df.columns:
            df.loc[mask_3m_bad, col] = float('nan')

    # --- External market features ---
    if EXTERNAL_FEATURES_PATH.exists():
        ext = pd.read_csv(EXTERNAL_FEATURES_PATH, sep=';', parse_dates=['date'])
        ext = ext.sort_values('date').reset_index(drop=True)
        # Merge on date (left join — keep all MOEX trading days)
        df = df.merge(ext[['date', 'brent', 'dxy', 'vix', 'cbr_rate']], on='date', how='left')
        # Forward-fill any gaps (MOEX holidays where US closed earlier)
        for col in ['brent', 'dxy', 'vix', 'cbr_rate']:
            if col in df.columns:
                df[col] = df[col].ffill()
        # Lag-1 and lag-2 for market prices (use T-1 to avoid lookahead)
        for col in ['brent', 'dxy', 'vix']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
        # Rolling mean of VIX (shift first to avoid lookahead)
        df['vix_roll5_mean'] = df['vix'].shift(1).rolling(5).mean()
        # cbr_rate: announced in advance, no lag needed
    else:
        # External features not available — columns will be missing, CatBoost skips them
        pass

    # --- Lag features ---
    lag_map = {
        'iv_1m': [1, 2, 3, 5],
        'hv_1m': [1, 2, 3],
        'spread_1m': [1, 2, 3],
        'ts_3m_1m': [1, 2, 3],
    }
    for base_col, lags in lag_map.items():
        for lag in lags:
            df[f'{base_col}_lag{lag}'] = df[base_col].shift(lag)

    # --- Rolling features (on iv_1m) ---
    df['iv_1m_roll5_mean'] = df['iv_1m'].shift(1).rolling(5).mean()
    df['iv_1m_roll10_mean'] = df['iv_1m'].shift(1).rolling(10).mean()
    df['iv_1m_roll5_std'] = df['iv_1m'].shift(1).rolling(5).std()
    df['iv_1m_roll21_mean'] = df['iv_1m'].shift(1).rolling(21).mean()

    # --- Calendar features ---
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # --- CBR meeting cycle features ---
    if CBR_MEETINGS_PATH.exists():
        meetings = pd.read_csv(CBR_MEETINGS_PATH, sep=';', parse_dates=['meeting_date'])
        meetings = meetings.sort_values('meeting_date').reset_index(drop=True)
        meeting_dates = meetings['meeting_date'].values  # numpy datetime64 array

        trading_dates = df['date'].values

        days_to_next = []
        days_since   = []
        is_meeting   = []
        last_sent    = []

        for dt in trading_dates:
            future = meetings[meetings['meeting_date'] >= dt]
            past   = meetings[meetings['meeting_date'] <= dt]

            # days_to_next_cbr_meeting: 0 on meeting day
            if not future.empty:
                next_dt = future.iloc[0]['meeting_date']
                days_to_next.append((next_dt - dt).days)
            else:
                days_to_next.append(None)

            # days_since_cbr_meeting: 0 on meeting day
            if not past.empty:
                last_dt = past.iloc[-1]['meeting_date']
                days_since.append((dt - last_dt).days)
            else:
                days_since.append(None)

            # is today a meeting day?
            is_meeting.append(1 if (not future.empty and future.iloc[0]['meeting_date'] == dt) else 0)

            # sentiment of last completed meeting (lag: use past meetings only)
            past_done = meetings[meetings['meeting_date'] < dt]
            last_sent.append(int(past_done.iloc[-1]['sentiment_score']) if not past_done.empty else 0)

        df['days_to_next_cbr_meeting'] = days_to_next
        df['days_since_cbr_meeting']   = days_since
        df['cbr_meeting_day']          = is_meeting
        df['cbr_last_sentiment']       = last_sent

    # --- Delta target: iv(t+1) - iv(t) ---
    # Computed here rather than from CSV so it's always consistent with the
    # filtered & sorted iv_1m series. Last row gets NaN (no next-day observation).
    df['target_delta_iv_1m'] = df['iv_1m'].shift(-1) - df['iv_1m']

    # Drop rows where required 1M lag features are NaN
    required_lag_cols = [c for c in LAG_COLS if c.startswith('iv_1m_lag')]
    df = df.dropna(subset=required_lag_cols).reset_index(drop=True)

    # Keep only columns we actually need
    keep_cols = ['date', 'iv_1m'] + FEATURE_COLS + [TARGET_COL]
    # Some cols may not be present if dataset is minimal — filter safely
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df.columns))
    df = df[keep_cols].copy()

    return df


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date into train / validation / test sets.

    Train:  up to TRAIN_END      (~Feb 2021 – Dec 2024)
    Val:    VAL_START – VAL_END  (Jan 2025 – Jul 2025)
    Test:   TEST_START onwards   (Aug 2025 – Feb 2026)
    """
    train = df[df['date'] <= TRAIN_END].copy()
    val = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()
    test = df[df['date'] >= TEST_START].copy()
    return train, val, test
