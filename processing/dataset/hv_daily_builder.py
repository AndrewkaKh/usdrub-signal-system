from __future__ import annotations

import math

import pandas as pd

from .config import ANNUALIZATION_DAYS, HV_WINDOWS


def _load_iv_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            target_tenor,
            underlying_secid,
            status
        FROM iv_daily
        WHERE date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def _load_futures_raw(connection, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            secid,
            last_price,
            settlement_price
        FROM futures_raw
        WHERE date <= ?
    '''
    return pd.read_sql_query(query, connection, params=[end_date])


def _prepare_futures(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['last_price'] = pd.to_numeric(data['last_price'], errors='coerce')
    data['settlement_price'] = pd.to_numeric(data['settlement_price'], errors='coerce')
    data['price_used'] = data['settlement_price']
    missing_settle = data['price_used'].isna() | (data['price_used'] <= 0)
    data.loc[missing_settle, 'price_used'] = data.loc[missing_settle, 'last_price']
    data = data[data['price_used'].notna() & (data['price_used'] > 0)].copy()
    data = data.sort_values(by=['secid', 'date']).reset_index(drop=True)
    return data


def _build_futures_hv_map(futures: pd.DataFrame) -> pd.DataFrame:
    if futures.empty:
        return pd.DataFrame()

    data = futures.copy()
    data['log_return'] = data.groupby('secid')['price_used'].transform(lambda s: pd.Series(s).apply(math.log).diff())

    for tenor, window in HV_WINDOWS.items():
        data[f'hv_{tenor}'] = (
            data.groupby('secid')['log_return']
            .transform(lambda s: s.rolling(window=window).std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS))
        )

    return data


def build_hv_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    iv_daily = _load_iv_daily(connection, start_date, end_date)
    futures_raw = _load_futures_raw(connection, end_date)

    if iv_daily.empty or futures_raw.empty:
        return pd.DataFrame()

    iv_daily['date'] = pd.to_datetime(iv_daily['date'])
    iv_daily = iv_daily[
        iv_daily['underlying_secid'].notna()
        & iv_daily['status'].isin(['ok', 'partial'])
    ].copy()

    if iv_daily.empty:
        return pd.DataFrame()

    futures = _prepare_futures(futures_raw)
    futures_hv = _build_futures_hv_map(futures)

    result_frames: list[pd.DataFrame] = []

    for tenor, window in HV_WINDOWS.items():
        iv_subset = iv_daily[iv_daily['target_tenor'] == tenor].copy()
        if iv_subset.empty:
            continue

        hv_subset = futures_hv[['date', 'secid', f'hv_{tenor}']].copy()
        hv_subset = hv_subset.rename(columns={'secid': 'underlying_secid', f'hv_{tenor}': 'hv'})

        merged = iv_subset.merge(
            hv_subset,
            on=['date', 'underlying_secid'],
            how='left',
        )

        merged['status'] = merged['hv'].apply(lambda x: 'ok' if pd.notna(x) and x > 0 else 'insufficient_history')
        merged['window'] = window
        merged['annualization_days'] = ANNUALIZATION_DAYS
        merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')

        result_frames.append(
            merged[
                [
                    'date',
                    'target_tenor',
                    'underlying_secid',
                    'hv',
                    'window',
                    'annualization_days',
                    'status',
                ]
            ]
        )

    if not result_frames:
        return pd.DataFrame()

    return pd.concat(result_frames, ignore_index=True).drop_duplicates(
        subset=['date', 'target_tenor', 'underlying_secid']
    ).reset_index(drop=True)