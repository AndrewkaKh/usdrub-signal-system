"""Общая логика выбора серий и контрактов для построения IV.

Используется в:
    - iv_daily_builder.py (построение дневных рядов IV)
    - iv_smile_builder.py (построение улыбки волатильности)
"""
from __future__ import annotations

import re

import pandas as pd

from .config import TARGET_TENOR_DAYS, TENOR_TOLERANCE_DAYS


FULL_FUTURES_CODE_PATTERN = re.compile(r'^(?P<asset>[A-Za-z]+)-(?P<month>\d{1,2})\.(?P<year>\d{2})$')
SHORT_FUTURES_CODE_PATTERN = re.compile(r'^(?P<asset>[A-Za-z]+)(?P<month_code>[HMUZ])(?P<year_digit>\d)$')

FUTURES_MONTH_CODES = {
    'H': 3,
    'M': 6,
    'U': 9,
    'Z': 12,
}


def load_contract_candidates(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            c.date,
            c.target_tenor,
            c.secid,
            c.option_type,
            c.strike,
            c.series_month,
            c.series_year,
            c.series_rank,
            c.strike_rank,
            c.price_used,
            c.last_price,
            c.settlement_price,
            c.open_interest,
            c.num_trades,
            r.expiry,
            r.underlying_secid
        FROM option_contract_candidates c
        LEFT JOIN option_contracts_reference r
            ON c.secid = r.secid
        WHERE c.date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def load_futures_raw(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            secid,
            last_price,
            settlement_price,
            open_interest,
            num_trades
        FROM futures_raw
        WHERE date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def prepare_options(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['expiry'] = pd.to_datetime(data['expiry'], errors='coerce')
    numeric_columns = [
        'strike',
        'series_month',
        'series_year',
        'series_rank',
        'strike_rank',
        'price_used',
        'last_price',
        'settlement_price',
        'open_interest',
        'num_trades',
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data = data[data['price_used'].notna() & (data['price_used'] > 0)].copy()
    return data


def _parse_futures_code(secid: str) -> tuple[int | None, int | None]:
    text = str(secid).strip()

    full_match = FULL_FUTURES_CODE_PATTERN.match(text)
    if full_match:
        return int(full_match.group('month')), 2000 + int(full_match.group('year'))

    short_match = SHORT_FUTURES_CODE_PATTERN.match(text)
    if short_match:
        month = FUTURES_MONTH_CODES.get(short_match.group('month_code'))
        year = 2020 + int(short_match.group('year_digit'))
        return month, year

    return None, None


def prepare_futures(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['last_price'] = pd.to_numeric(data['last_price'], errors='coerce')
    data['settlement_price'] = pd.to_numeric(data['settlement_price'], errors='coerce')
    data['open_interest'] = pd.to_numeric(data['open_interest'], errors='coerce').fillna(0.0)
    data['num_trades'] = pd.to_numeric(data['num_trades'], errors='coerce').fillna(0.0)
    data['futures_price'] = data['settlement_price']
    missing_settle = data['futures_price'].isna() | (data['futures_price'] <= 0)
    data.loc[missing_settle, 'futures_price'] = data.loc[missing_settle, 'last_price']

    parsed = data['secid'].apply(_parse_futures_code).apply(pd.Series)
    parsed.columns = ['future_month', 'future_year']
    data = pd.concat([data.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    data = data[data['futures_price'].notna() & (data['futures_price'] > 0)].copy()
    return data


def choose_series(group: pd.DataFrame, target_tenor: str) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()

    target_days = TARGET_TENOR_DAYS[target_tenor]
    tolerance_days = TENOR_TOLERANCE_DAYS[target_tenor]

    series = (
        group.dropna(subset=['expiry'])
        .groupby(['series_year', 'series_month', 'series_rank', 'expiry'], as_index=False)
        .agg(
            total_open_interest=('open_interest', 'sum'),
            total_num_trades=('num_trades', 'sum'),
            contracts_count=('secid', 'nunique'),
        )
    )

    if series.empty:
        return pd.DataFrame()

    trade_date = group['date'].iloc[0]
    series['days_to_expiry'] = (series['expiry'] - trade_date).dt.days
    series = series[series['days_to_expiry'] > 0].copy()

    if series.empty:
        return pd.DataFrame()

    series['distance_to_target'] = (series['days_to_expiry'] - target_days).abs()
    series = series[series['distance_to_target'] <= tolerance_days].copy()

    if series.empty:
        return pd.DataFrame()

    series = series.sort_values(
        by=[
            'distance_to_target',
            'series_rank',
            'total_open_interest',
            'total_num_trades',
            'contracts_count',
        ],
        ascending=[True, True, False, False, False],
    )

    best = series.iloc[0]
    mask = (
        (group['series_year'] == best['series_year'])
        & (group['series_month'] == best['series_month'])
        & (group['series_rank'] == best['series_rank'])
        & (group['expiry'] == best['expiry'])
    )
    return group[mask].copy()


def choose_underlying_future(options_group: pd.DataFrame, futures_group: pd.DataFrame) -> tuple[str | None, float | None]:
    if futures_group.empty:
        return None, None

    referenced = options_group['underlying_secid'].dropna().astype(str).unique().tolist()
    if referenced:
        matched = futures_group[futures_group['secid'].isin(referenced)].copy()
        if not matched.empty:
            matched = matched.sort_values(by=['open_interest', 'num_trades'], ascending=[False, False])
            row = matched.iloc[0]
            return str(row['secid']), float(row['futures_price'])

    expiry = options_group['expiry'].dropna().iloc[0] if options_group['expiry'].notna().any() else None
    if expiry is not None:
        futures = futures_group.copy()
        futures['months_after_expiry'] = (
            (futures['future_year'] - expiry.year) * 12
            + (futures['future_month'] - expiry.month)
        )
        futures = futures[futures['months_after_expiry'] >= 0].copy()
        if not futures.empty:
            futures = futures.sort_values(
                by=['months_after_expiry', 'open_interest', 'num_trades'],
                ascending=[True, False, False],
            )
            row = futures.iloc[0]
            return str(row['secid']), float(row['futures_price'])

    futures = futures_group.sort_values(by=['open_interest', 'num_trades'], ascending=[False, False])
    row = futures.iloc[0]
    return str(row['secid']), float(row['futures_price'])
