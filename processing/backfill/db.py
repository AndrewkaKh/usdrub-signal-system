from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import SQLITE_DB_PATH


def get_connection(db_path: str = SQLITE_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=WAL;')
    connection.execute('PRAGMA synchronous=NORMAL;')
    connection.execute('PRAGMA temp_store=MEMORY;')
    connection.execute('PRAGMA foreign_keys=ON;')
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        '''
        CREATE TABLE IF NOT EXISTS futures_raw (
            date TEXT NOT NULL,
            secid TEXT NOT NULL,
            shortname TEXT,
            tradedate TEXT,
            last_price REAL,
            settlement_price REAL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            volume REAL,
            open_interest REAL,
            num_trades REAL,
            PRIMARY KEY (date, secid)
        );

        CREATE TABLE IF NOT EXISTS options_raw (
            date TEXT NOT NULL,
            secid TEXT NOT NULL,
            asset_code TEXT,
            option_type TEXT,
            strike REAL,
            series_month INTEGER,
            series_year INTEGER,
            settlement_type TEXT,
            code_format TEXT,
            tradedate TEXT,
            last_price REAL,
            settlement_price REAL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            volume REAL,
            open_interest REAL,
            num_trades REAL,
            PRIMARY KEY (date, secid)
        );

        CREATE TABLE IF NOT EXISTS option_contracts_reference (
            secid TEXT PRIMARY KEY,
            asset_code TEXT,
            option_type TEXT,
            strike REAL,
            series_month INTEGER,
            series_year INTEGER,
            expiry TEXT,
            underlying_secid TEXT,
            shortname TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS option_series_candidates (
            date TEXT NOT NULL,
            target_tenor TEXT NOT NULL,
            series_month INTEGER NOT NULL,
            series_year INTEGER NOT NULL,
            months_ahead INTEGER,
            series_rank INTEGER,
            contracts_count INTEGER,
            priced_contracts_count INTEGER,
            traded_contracts_count INTEGER,
            total_open_interest REAL,
            total_num_trades REAL,
            PRIMARY KEY (date, target_tenor, series_year, series_month)
        );

        CREATE TABLE IF NOT EXISTS option_contract_candidates (
            date TEXT NOT NULL,
            target_tenor TEXT NOT NULL,
            secid TEXT NOT NULL,
            option_type TEXT,
            strike REAL,
            series_month INTEGER,
            series_year INTEGER,
            series_rank INTEGER,
            strike_rank INTEGER,
            price_used REAL,
            last_price REAL,
            settlement_price REAL,
            open_interest REAL,
            num_trades REAL,
            PRIMARY KEY (date, target_tenor, secid)
        );

        CREATE TABLE IF NOT EXISTS iv_daily (
            date TEXT NOT NULL,
            target_tenor TEXT NOT NULL,
            underlying_secid TEXT,
            expiry TEXT,
            days_to_expiry INTEGER,
            t REAL,
            futures_price REAL,
            strike REAL,
            call_secid TEXT,
            put_secid TEXT,
            call_price REAL,
            put_price REAL,
            call_iv REAL,
            put_iv REAL,
            iv REAL,
            status TEXT,
            message TEXT,
            series_month INTEGER,
            series_year INTEGER,
            PRIMARY KEY (date, target_tenor)
        );

        CREATE TABLE IF NOT EXISTS hv_daily (
            date TEXT NOT NULL,
            target_tenor TEXT NOT NULL,
            underlying_secid TEXT NOT NULL,
            hv REAL,
            window INTEGER,
            annualization_days INTEGER,
            status TEXT,
            PRIMARY KEY (date, target_tenor, underlying_secid)
        );

        CREATE TABLE IF NOT EXISTS model_dataset_daily (
            date TEXT PRIMARY KEY,
            underlying_1m TEXT,
            underlying_3m TEXT,
            iv_1m REAL,
            iv_3m REAL,
            hv_1m REAL,
            hv_3m REAL,
            spread_1m REAL,
            spread_3m REAL,
            ts_3m_1m REAL,
            iv_status_1m TEXT,
            iv_status_3m TEXT,
            hv_status_1m TEXT,
            hv_status_3m TEXT,
            days_to_expiry_1m INTEGER,
            days_to_expiry_3m INTEGER,
            target_iv_1m_next_day REAL,
            target_delta_iv_1m_next_day REAL
        );

        CREATE INDEX IF NOT EXISTS idx_futures_raw_date ON futures_raw(date);
        CREATE INDEX IF NOT EXISTS idx_futures_raw_secid ON futures_raw(secid);

        CREATE INDEX IF NOT EXISTS idx_options_raw_date ON options_raw(date);
        CREATE INDEX IF NOT EXISTS idx_options_raw_secid ON options_raw(secid);
        CREATE INDEX IF NOT EXISTS idx_options_raw_series ON options_raw(series_year, series_month);
        CREATE INDEX IF NOT EXISTS idx_options_raw_type ON options_raw(option_type);

        CREATE INDEX IF NOT EXISTS idx_option_contracts_reference_asset ON option_contracts_reference(asset_code);

        CREATE INDEX IF NOT EXISTS idx_option_series_candidates_date ON option_series_candidates(date);
        CREATE INDEX IF NOT EXISTS idx_option_contract_candidates_date ON option_contract_candidates(date);
        CREATE INDEX IF NOT EXISTS idx_option_contract_candidates_secid ON option_contract_candidates(secid);

        CREATE INDEX IF NOT EXISTS idx_iv_daily_date ON iv_daily(date);
        CREATE INDEX IF NOT EXISTS idx_hv_daily_date ON hv_daily(date);
        '''
    )
    connection.commit()