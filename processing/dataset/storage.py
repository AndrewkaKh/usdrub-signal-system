from __future__ import annotations

import sqlite3

import pandas as pd


IV_DAILY_COLUMNS = [
    'date',
    'target_tenor',
    'underlying_secid',
    'expiry',
    'days_to_expiry',
    't',
    'futures_price',
    'strike',
    'call_secid',
    'put_secid',
    'call_price',
    'put_price',
    'call_iv',
    'put_iv',
    'iv',
    'status',
    'message',
    'series_month',
    'series_year',
]

HV_DAILY_COLUMNS = [
    'date',
    'target_tenor',
    'underlying_secid',
    'hv',
    'window',
    'annualization_days',
    'status',
]

MODEL_DATASET_COLUMNS = [
    'date',
    'underlying_1m',
    'underlying_3m',
    'iv_1m',
    'iv_3m',
    'hv_1m',
    'hv_3m',
    'spread_1m',
    'spread_3m',
    'ts_3m_1m',
    'iv_status_1m',
    'iv_status_3m',
    'hv_status_1m',
    'hv_status_3m',
    'days_to_expiry_1m',
    'days_to_expiry_3m',
    'target_iv_1m_next_day',
    'target_delta_iv_1m_next_day',
]


def _normalize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)

    normalized = frame.copy()

    for column in columns:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[columns]
    normalized = normalized.where(pd.notnull(normalized), None)
    return normalized


def save_iv_daily(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = _normalize_frame(frame, IV_DAILY_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO iv_daily (
            date,
            target_tenor,
            underlying_secid,
            expiry,
            days_to_expiry,
            t,
            futures_price,
            strike,
            call_secid,
            put_secid,
            call_price,
            put_price,
            call_iv,
            put_iv,
            iv,
            status,
            message,
            series_month,
            series_year
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_hv_daily(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = _normalize_frame(frame, HV_DAILY_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO hv_daily (
            date,
            target_tenor,
            underlying_secid,
            hv,
            window,
            annualization_days,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_model_dataset_daily(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = _normalize_frame(frame, MODEL_DATASET_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO model_dataset_daily (
            date,
            underlying_1m,
            underlying_3m,
            iv_1m,
            iv_3m,
            hv_1m,
            hv_3m,
            spread_1m,
            spread_3m,
            ts_3m_1m,
            iv_status_1m,
            iv_status_3m,
            hv_status_1m,
            hv_status_3m,
            days_to_expiry_1m,
            days_to_expiry_3m,
            target_iv_1m_next_day,
            target_delta_iv_1m_next_day
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)