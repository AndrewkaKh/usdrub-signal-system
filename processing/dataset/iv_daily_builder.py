from __future__ import annotations

import pandas as pd

from processing.iv.calculator import calculate_option_iv
from processing.iv.utils import compute_time_to_expiry
from .selection import (
    choose_series,
    choose_underlying_future,
    load_contract_candidates,
    load_futures_raw,
    prepare_futures,
    prepare_options,
)


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
    options_raw = load_contract_candidates(connection, start_date, end_date)
    futures_raw = load_futures_raw(connection, start_date, end_date)

    options = prepare_options(options_raw)
    futures = prepare_futures(futures_raw)

    if options.empty:
        return pd.DataFrame()

    result_rows: list[dict[str, object]] = []

    for (date_value, target_tenor), options_group in options.groupby(['date', 'target_tenor']):
        selected_options = choose_series(options_group, target_tenor)

        if selected_options.empty:
            result_rows.append(_build_failure_row(date_value, target_tenor, 'series_not_found', 'Could not select series by expiry'))
            continue

        futures_group = futures[futures['date'] == date_value].copy()
        underlying_secid, futures_price = choose_underlying_future(selected_options, futures_group)

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
