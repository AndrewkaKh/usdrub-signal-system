from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ..utils import normalize_date


TARGET_TENORS = {
    '1m': 1,
    '3m': 3,
}


def _load_options_raw(connection, start_date: date, end_date: date) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            secid,
            option_type,
            strike,
            series_month,
            series_year,
            last_price,
            settlement_price,
            open_interest,
            num_trades
        FROM options_raw
        WHERE date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date.isoformat(), end_date.isoformat()])


def _prepare_options_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['series_month'] = pd.to_numeric(data['series_month'], errors='coerce')
    data['series_year'] = pd.to_numeric(data['series_year'], errors='coerce')
    data['strike'] = pd.to_numeric(data['strike'], errors='coerce')
    data['last_price'] = pd.to_numeric(data['last_price'], errors='coerce')
    data['settlement_price'] = pd.to_numeric(data['settlement_price'], errors='coerce')
    data['open_interest'] = pd.to_numeric(data['open_interest'], errors='coerce').fillna(0.0)
    data['num_trades'] = pd.to_numeric(data['num_trades'], errors='coerce').fillna(0.0)

    data['price_used'] = data['settlement_price']
    missing_settle = data['price_used'].isna() | (data['price_used'] <= 0)
    data.loc[missing_settle, 'price_used'] = data.loc[missing_settle, 'last_price']

    data['has_price'] = data['price_used'].notna() & (data['price_used'] > 0)
    data['has_liquidity'] = (data['num_trades'] > 0) | (data['open_interest'] > 0)
    data['months_ahead'] = (
        (data['series_year'] - data['date'].dt.year) * 12
        + (data['series_month'] - data['date'].dt.month)
    )

    data = data[
        data['has_price']
        & data['has_liquidity']
        & data['series_month'].notna()
        & data['series_year'].notna()
        & data['strike'].notna()
        & (data['months_ahead'] >= 0)
        & (data['months_ahead'] <= 6)
    ].copy()

    return data


def _build_series_candidates(
    quality_options: pd.DataFrame,
    series_pool_size: int,
) -> pd.DataFrame:
    if quality_options.empty:
        return pd.DataFrame()

    summary = (
        quality_options.groupby(['date', 'series_year', 'series_month', 'months_ahead'], as_index=False)
        .agg(
            contracts_count=('secid', 'nunique'),
            priced_contracts_count=('price_used', 'count'),
            traded_contracts_count=('num_trades', lambda s: int((s > 0).sum())),
            total_open_interest=('open_interest', 'sum'),
            total_num_trades=('num_trades', 'sum'),
        )
    )

    candidate_frames: list[pd.DataFrame] = []

    for target_tenor, target_months in TARGET_TENORS.items():
        block = summary.copy()
        block['target_tenor'] = target_tenor
        block['distance_to_target'] = (block['months_ahead'] - target_months).abs()

        block = block.sort_values(
            by=[
                'date',
                'distance_to_target',
                'total_open_interest',
                'total_num_trades',
                'contracts_count',
            ],
            ascending=[True, True, False, False, False],
        ).copy()

        block['series_rank'] = block.groupby('date').cumcount() + 1
        block = block[block['series_rank'] <= series_pool_size].copy()
        candidate_frames.append(block)

    if not candidate_frames:
        return pd.DataFrame()

    result = pd.concat(candidate_frames, ignore_index=True)
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    return result[
        [
            'date',
            'target_tenor',
            'series_month',
            'series_year',
            'months_ahead',
            'series_rank',
            'contracts_count',
            'priced_contracts_count',
            'traded_contracts_count',
            'total_open_interest',
            'total_num_trades',
        ]
    ].reset_index(drop=True)


def _build_contract_candidates(
    quality_options: pd.DataFrame,
    series_candidates: pd.DataFrame,
    max_strikes_per_series: int,
) -> pd.DataFrame:
    if quality_options.empty or series_candidates.empty:
        return pd.DataFrame()

    data = quality_options.copy()
    data['date_key'] = data['date'].dt.strftime('%Y-%m-%d')

    series_map = series_candidates.rename(columns={'date': 'date_key'})
    merged = data.merge(
        series_map[
            [
                'date_key',
                'target_tenor',
                'series_month',
                'series_year',
                'series_rank',
            ]
        ],
        on=['date_key', 'series_month', 'series_year'],
        how='inner',
    )

    if merged.empty:
        return pd.DataFrame()

    strike_stats = (
        merged.groupby(
            ['date_key', 'target_tenor', 'series_year', 'series_month', 'series_rank', 'strike'],
            as_index=False,
        )
        .agg(
            has_call=('option_type', lambda s: int('call' in set(s))),
            has_put=('option_type', lambda s: int('put' in set(s))),
            strike_total_open_interest=('open_interest', 'sum'),
            strike_total_num_trades=('num_trades', 'sum'),
            strike_contracts_count=('secid', 'nunique'),
        )
    )

    strike_stats['both_sides_score'] = strike_stats['has_call'] + strike_stats['has_put']

    strike_stats = strike_stats.sort_values(
        by=[
            'date_key',
            'target_tenor',
            'series_rank',
            'both_sides_score',
            'strike_total_open_interest',
            'strike_total_num_trades',
            'strike_contracts_count',
            'strike',
        ],
        ascending=[True, True, True, False, False, False, False, True],
    ).copy()

    strike_stats['strike_rank'] = strike_stats.groupby(
        ['date_key', 'target_tenor', 'series_year', 'series_month', 'series_rank']
    ).cumcount() + 1

    strike_stats = strike_stats[strike_stats['strike_rank'] <= max_strikes_per_series].copy()

    result = merged.merge(
        strike_stats[
            [
                'date_key',
                'target_tenor',
                'series_year',
                'series_month',
                'series_rank',
                'strike',
                'strike_rank',
            ]
        ],
        on=['date_key', 'target_tenor', 'series_year', 'series_month', 'series_rank', 'strike'],
        how='inner',
    )

    if result.empty:
        return pd.DataFrame()

    result['date'] = result['date_key']

    return result[
        [
            'date',
            'target_tenor',
            'secid',
            'option_type',
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
    ].drop_duplicates(subset=['date', 'target_tenor', 'secid']).reset_index(drop=True)


def build_candidate_tables(
    connection,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    series_pool_size: int = 2,
    max_strikes_per_series: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    raw_options = _load_options_raw(connection, start_dt, end_dt)
    quality_options = _prepare_options_frame(raw_options)
    series_candidates = _build_series_candidates(quality_options, series_pool_size=series_pool_size)
    contract_candidates = _build_contract_candidates(
        quality_options,
        series_candidates,
        max_strikes_per_series=max_strikes_per_series,
    )
    return series_candidates, contract_candidates