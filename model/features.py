from __future__ import annotations

# Target variable: next-day IV change (delta)
# Baseline for this target = predict 0 (no change), not current IV level.
TARGET_COL = 'target_delta_iv_1m'

# Base features present in model_dataset_daily.csv
BASE_1M_COLS = ['iv_1m', 'hv_1m', 'spread_1m', 'days_to_expiry_1m']
BASE_3M_COLS = ['iv_3m', 'hv_3m', 'spread_3m', 'ts_3m_1m', 'days_to_expiry_3m']

# Lag features (computed in data_prep.py)
LAG_COLS = [
    'iv_1m_lag1', 'iv_1m_lag2', 'iv_1m_lag3', 'iv_1m_lag5',
    'hv_1m_lag1', 'hv_1m_lag2', 'hv_1m_lag3',
    'spread_1m_lag1', 'spread_1m_lag2', 'spread_1m_lag3',
    'ts_3m_1m_lag1', 'ts_3m_1m_lag2', 'ts_3m_1m_lag3',
]

# Rolling window features (computed in data_prep.py)
ROLLING_COLS = [
    'iv_1m_roll5_mean',
    'iv_1m_roll10_mean',
    'iv_1m_roll5_std',
    'iv_1m_roll21_mean',
]

# Calendar features (computed in data_prep.py)
CALENDAR_COLS = ['day_of_week', 'month']

# External market features (merged from data/exports/external_features.csv)
# All market lags use T-1 to avoid lookahead (US markets close after MOEX)
EXTERNAL_COLS = [
    'brent_lag1', 'brent_lag2',
    'dxy_lag1', 'dxy_lag2',
    'vix_lag1', 'vix_lag2',
    'vix_roll5_mean',
    'cbr_rate',           # known in advance — no lag needed
]

# CBR meeting cycle features (computed in data_prep.py from cbr_meetings.csv)
# days_to_next_cbr_meeting: countdown to next meeting (0 on meeting day, ~45 max)
# days_since_cbr_meeting:   days elapsed since last meeting (0 on meeting day)
# cbr_meeting_day:          binary flag for meeting day itself
# cbr_last_sentiment:       -1 / 0 / +1 = last decision was cut / hold / hike
CBR_MEETING_COLS = [
    'days_to_next_cbr_meeting',
    'days_since_cbr_meeting',
    'cbr_meeting_day',
    'cbr_last_sentiment',
]

# Full feature set — 3M and external cols may contain NaN (CatBoost handles them natively)
# CBR_MEETING_COLS excluded: tested but degraded walk-forward (5/8 vs 7/8 beats baseline)
FEATURE_COLS: list[str] = (
    BASE_1M_COLS
    + BASE_3M_COLS
    + LAG_COLS
    + ROLLING_COLS
    + CALENDAR_COLS
    + EXTERNAL_COLS
)
