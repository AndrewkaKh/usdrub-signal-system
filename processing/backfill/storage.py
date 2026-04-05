from __future__ import annotations

import sqlite3

import pandas as pd

from ..utils import normalize_frame


FUTURES_COLUMNS = [
    'date',
    'secid',
    'shortname',
    'tradedate',
    'last_price',
    'settlement_price',
    'open_price',
    'high_price',
    'low_price',
    'volume',
    'open_interest',
    'num_trades',
]

OPTIONS_COLUMNS = [
    'date',
    'secid',
    'asset_code',
    'option_type',
    'strike',
    'series_month',
    'series_year',
    'settlement_type',
    'code_format',
    'tradedate',
    'last_price',
    'settlement_price',
    'open_price',
    'high_price',
    'low_price',
    'volume',
    'open_interest',
    'num_trades',
]

OPTION_SERIES_CANDIDATE_COLUMNS = [
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

OPTION_CONTRACT_CANDIDATE_COLUMNS = [
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

OPTION_CONTRACT_REFERENCE_COLUMNS = [
    'secid',
    'asset_code',
    'option_type',
    'strike',
    'series_month',
    'series_year',
    'expiry',
    'underlying_secid',
    'shortname',
    'updated_at',
]


def save_futures_raw(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = normalize_frame(frame, FUTURES_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO futures_raw (
            date,
            secid,
            shortname,
            tradedate,
            last_price,
            settlement_price,
            open_price,
            high_price,
            low_price,
            volume,
            open_interest,
            num_trades
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_options_raw(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = normalize_frame(frame, OPTIONS_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO options_raw (
            date,
            secid,
            asset_code,
            option_type,
            strike,
            series_month,
            series_year,
            settlement_type,
            code_format,
            tradedate,
            last_price,
            settlement_price,
            open_price,
            high_price,
            low_price,
            volume,
            open_interest,
            num_trades
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_option_series_candidates(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = normalize_frame(frame, OPTION_SERIES_CANDIDATE_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO option_series_candidates (
            date,
            target_tenor,
            series_month,
            series_year,
            months_ahead,
            series_rank,
            contracts_count,
            priced_contracts_count,
            traded_contracts_count,
            total_open_interest,
            total_num_trades
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_option_contract_candidates(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = normalize_frame(frame, OPTION_CONTRACT_CANDIDATE_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO option_contract_candidates (
            date,
            target_tenor,
            secid,
            option_type,
            strike,
            series_month,
            series_year,
            series_rank,
            strike_rank,
            price_used,
            last_price,
            settlement_price,
            open_interest,
            num_trades
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)


def save_option_contracts_reference(connection: sqlite3.Connection, frame: pd.DataFrame) -> int:
    normalized = normalize_frame(frame, OPTION_CONTRACT_REFERENCE_COLUMNS)
    if normalized.empty:
        return 0

    rows = [tuple(record) for record in normalized.itertuples(index=False, name=None)]

    connection.executemany(
        '''
        INSERT OR REPLACE INTO option_contracts_reference (
            secid,
            asset_code,
            option_type,
            strike,
            series_month,
            series_year,
            expiry,
            underlying_secid,
            shortname,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    connection.commit()
    return len(rows)