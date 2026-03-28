from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from processing.iv.calculator import calculate_option_iv
from processing.iv.utils import compute_time_to_expiry
from .config import TARGET_TENOR_DAYS, TENOR_TOLERANCE_DAYS


FULL_FUTURES_CODE_PATTERN = re.compile(r'^(?P<asset>[A-Za-z]+)-(?P<month>\d{1,2})\.(?P<year>\d{2})$')
SHORT_FUTURES_CODE_PATTERN = re.compile(r'^(?P<asset>[A-Za-z]+)(?P<month_code>[HMUZ])(?P<year_digit>\d)$')

FUTURES_MONTH_CODES = {
    'H': 3,
    'M': 6,
    'U': 9,
    'Z': 12,
}


def _load_contract_candidates(connection, start_date: str, end_date: str) -> pd.DataFrame:
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


def _load_futures_raw(connection, start_date: str, end_date: str) -> pd.DataFrame:
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


def _prepare_options(frame: pd.DataFrame) -> pd.DataFrame:
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


def _prepare_futures(frame: pd.DataFrame) -> pd.DataFrame:
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


def _choose_series(group: pd.DataFrame, target_tenor: str) -> pd.DataFrame:
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


def _choose_underlying_future(options_group: pd.DataFrame, futures_group: pd.DataFrame) -> tuple[str | None, float | None]:
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


def _choose_strike_pair(options_group: pd.DataFrame, futures_price: float) -> tuple[pd.Series | None, pd.Series | None, float | None]:
    if options_group.empty:
        return None, None, None

    summary = (
        options_group.groupby('strike', as_index=False)
        .agg(
            has_call=('option_type', lambda s: int('call' in set(s))),
            has_put=('option_type', lambda s: int('put' in set(s))),
            total_open_interest=('open_interest', 'sum'),
            total_num_trades=('num_trades', 'sum'),
        )
    )

    summary['both_sides_score'] = summary['has_call'] + summary['has_put']
    summary['distance_to_atm'] = (summary['strike'] - futures_price).abs()

    summary = summary.sort_values(
        by=['both_sides_score', 'distance_to_atm', 'total_open_interest', 'total_num_trades'],
        ascending=[False, True, False, False],
    )

    if summary.empty:
        return None, None, None

    selected_strike = float(summary.iloc[0]['strike'])
    strike_rows = options_group[options_group['strike'] == selected_strike].copy()

    call_rows = strike_rows[strike_rows['option_type'] == 'call'].copy()
    put_rows = strike_rows[strike_rows['option_type'] == 'put'].copy()

    call_row = None
    put_row = None

    if not call_rows.empty:
        call_rows = call_rows.sort_values(by=['num_trades', 'open_interest'], ascending=[False, False])
        call_row = call_rows.iloc[0]

    if not put_rows.empty:
        put_rows = put_rows.sort_values(by=['num_trades', 'open_interest'], ascending=[False, False])
        put_row = put_rows.iloc[0]

    return call_row, put_row, selected_strike


def _build_failure_row(date_value: pd.Timestamp, target_tenor: str, status: str, message: str) -> dict[str, object]:
    return {
        'date': date_value.strftime('%Y-%m-%d'),
        'target_tenor': target_tenor,
        'underlying_secid': None,
        'expiry': None,
        'days_to_expiry': None,
        't': None,
        'futures_price': None,
        'strike': None,
        'call_secid': None,
        'put_secid': None,
        'call_price': None,
        'put_price': None,
        'call_iv': None,
        'put_iv': None,
        'iv': None,
        'status': status,
        'message': message,
        'series_month': None,
        'series_year': None,
    }


def _build_success_row(date_value: pd.Timestamp, target_tenor: str, selected_options: pd.DataFrame, futures_price: float, underlying_secid: str, call_row, put_row, strike: float) -> dict[str, object]:
    expiry = selected_options['expiry'].iloc[0]
    days_to_expiry = int((expiry - date_value).days)
    t = compute_time_to_expiry(date_value.to_pydatetime(), expiry.to_pydatetime())

    call_iv = None
    put_iv = None

    if call_row is not None:
        call_iv, _ = calculate_option_iv(
            float(call_row['price_used']),
            float(futures_price),
            float(strike),
            float(t),
            str(call_row['option_type']),
        )

    if put_row is not None:
        put_iv, _ = calculate_option_iv(
            float(put_row['price_used']),
            float(futures_price),
            float(strike),
            float(t),
            str(put_row['option_type']),
        )

    valid_ivs = [value for value in [call_iv, put_iv] if value is not None]

    if len(valid_ivs) == 2:
        status = 'ok'
        message = None
        iv_value = sum(valid_ivs) / 2.0
    elif len(valid_ivs) == 1:
        status = 'partial'
        message = 'Only one side produced valid IV'
        iv_value = valid_ivs[0]
    else:
        status = 'iv_failed'
        message = 'Neither call nor put produced valid IV'
        iv_value = None

    return {
        'date': date_value.strftime('%Y-%m-%d'),
        'target_tenor': target_tenor,
        'underlying_secid': underlying_secid,
        'expiry': expiry.strftime('%Y-%m-%d'),
        'days_to_expiry': days_to_expiry,
        't': float(t),
        'futures_price': float(futures_price),
        'strike': float(strike),
        'call_secid': None if call_row is None else str(call_row['secid']),
        'put_secid': None if put_row is None else str(put_row['secid']),
        'call_price': None if call_row is None else float(call_row['price_used']),
        'put_price': None if put_row is None else float(put_row['price_used']),
        'call_iv': call_iv,
        'put_iv': put_iv,
        'iv': iv_value,
        'status': status,
        'message': message,
        'series_month': int(selected_options['series_month'].iloc[0]),
        'series_year': int(selected_options['series_year'].iloc[0]),
    }


def build_iv_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    options_raw = _load_contract_candidates(connection, start_date, end_date)
    futures_raw = _load_futures_raw(connection, start_date, end_date)

    options = _prepare_options(options_raw)
    futures = _prepare_futures(futures_raw)

    if options.empty:
        return pd.DataFrame()

    result_rows: list[dict[str, object]] = []

    for (date_value, target_tenor), options_group in options.groupby(['date', 'target_tenor']):
        selected_options = _choose_series(options_group, target_tenor)

        if selected_options.empty:
            result_rows.append(_build_failure_row(date_value, target_tenor, 'series_not_found', 'Could not select series by expiry'))
            continue

        futures_group = futures[futures['date'] == date_value].copy()
        underlying_secid, futures_price = _choose_underlying_future(selected_options, futures_group)

        if underlying_secid is None or futures_price is None:
            result_rows.append(_build_failure_row(date_value, target_tenor, 'underlying_not_found', 'Could not determine underlying futures contract'))
            continue

        call_row, put_row, strike = _choose_strike_pair(selected_options, futures_price)

        if strike is None:
            result_rows.append(_build_failure_row(date_value, target_tenor, 'atm_strike_not_found', 'Could not determine ATM-like strike'))
            continue

        result_rows.append(
            _build_success_row(
                date_value=date_value,
                target_tenor=target_tenor,
                selected_options=selected_options,
                futures_price=futures_price,
                underlying_secid=underlying_secid,
                call_row=call_row,
                put_row=put_row,
                strike=strike,
            )
        )

    return pd.DataFrame(result_rows)