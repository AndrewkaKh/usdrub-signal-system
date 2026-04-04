"""Reusable daily data pipeline: backfill + shortlist + dataset + export.

Both pipeline_runner.py (CLI) and the Telegram bot call this module so that
orchestration logic lives in exactly one place.

Usage (programmatic):
    from processing.daily_pipeline import run_daily_pipeline
    from processing.backfill import get_connection, initialize_database

    conn = get_connection()
    initialize_database(conn)
    result = run_daily_pipeline(conn, '2026-04-09', '2026-04-09')
    conn.close()
"""
from __future__ import annotations

import logging
from datetime import date, datetime

from processing.backfill import (
    build_candidate_tables,
    build_missing_contract_references,
    load_futures_backfill,
    load_options_backfill,
    save_futures_raw,
    save_option_contract_candidates,
    save_option_contracts_reference,
    save_option_series_candidates,
    save_options_raw,
)
from processing.dataset import (
    build_hv_daily,
    build_iv_daily,
    build_model_dataset_daily,
    save_hv_daily,
    save_iv_daily,
    save_model_dataset_daily,
)
from processing.dataset.exporter import export_hv_daily, export_iv_daily, export_model_dataset_daily

logger = logging.getLogger(__name__)

# Default export paths (same as pipeline_runner.py)
DEFAULT_MODEL_CSV = 'data/exports/model_dataset_daily.csv'
DEFAULT_IV_CSV = 'data/exports/iv_daily.csv'
DEFAULT_HV_CSV = 'data/exports/hv_daily.csv'


def _to_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), '%Y-%m-%d').date()


def run_daily_pipeline(
    connection,
    start_date: str | date,
    end_date: str | date,
    series_pool_size: int = 2,
    max_strikes_per_series: int = 10,
    reference_workers: int = 8,
    skip_backfill: bool = False,
    skip_shortlist: bool = False,
    skip_reference: bool = True,
    skip_dataset: bool = False,
    skip_export: bool = False,
    model_csv: str = DEFAULT_MODEL_CSV,
    iv_csv: str = DEFAULT_IV_CSV,
    hv_csv: str = DEFAULT_HV_CSV,
) -> dict:
    """Run the full daily data pipeline for the given date range.

    Parameters
    ----------
    connection:
        Open SQLite connection (caller is responsible for opening/closing).
    start_date, end_date:
        Date range as ISO strings or date objects.
    skip_reference:
        Defaults to True — reference loading is slow and not needed for the
        smile / bot pipeline. Set to False for full historical backfills.

    Returns
    -------
    dict with counts of loaded/saved rows per stage, ready for logging.
    """
    start = _to_date(start_date)
    end = _to_date(end_date)
    start_iso = start.isoformat()
    end_iso = end.isoformat()

    result: dict = {
        'start_date': start_iso,
        'end_date': end_iso,
        'futures_loaded': 0, 'options_loaded': 0,
        'futures_saved': 0, 'options_saved': 0,
        'series_candidates': 0, 'contract_candidates': 0,
        'series_saved': 0, 'contracts_saved': 0,
        'reference_rows': 0, 'reference_saved': 0,
        'iv_rows': 0, 'hv_rows': 0, 'dataset_rows': 0,
        'iv_saved': 0, 'hv_saved': 0, 'dataset_saved': 0,
        'model_csv': None, 'iv_csv': None, 'hv_csv': None,
    }

    # --- Backfill ---
    if not skip_backfill:
        logger.info('Backfill: loading futures + options %s – %s', start_iso, end_iso)
        futures_frame = load_futures_backfill(start, end)
        options_frame = load_options_backfill(start, end)
        result['futures_loaded'] = len(futures_frame)
        result['options_loaded'] = len(options_frame)
        result['futures_saved'] = save_futures_raw(connection, futures_frame)
        result['options_saved'] = save_options_raw(connection, options_frame)
        logger.info(
            'Backfill done: futures=%d options=%d',
            result['futures_loaded'], result['options_loaded'],
        )

    # --- Shortlist ---
    if not skip_shortlist:
        logger.info('Shortlist: building candidate tables')
        series_candidates, contract_candidates = build_candidate_tables(
            connection=connection,
            start_date=start,
            end_date=end,
            series_pool_size=series_pool_size,
            max_strikes_per_series=max_strikes_per_series,
        )
        result['series_candidates'] = len(series_candidates)
        result['contract_candidates'] = len(contract_candidates)
        result['series_saved'] = save_option_series_candidates(connection, series_candidates)
        result['contracts_saved'] = save_option_contract_candidates(connection, contract_candidates)
        logger.info(
            'Shortlist done: series=%d contracts=%d',
            result['series_candidates'], result['contract_candidates'],
        )

    # --- Reference ---
    if not skip_reference:
        logger.info('Reference: loading missing contract references')
        reference_frame = build_missing_contract_references(
            connection=connection,
            start_date=start_iso,
            end_date=end_iso,
            max_workers=reference_workers,
        )
        result['reference_rows'] = len(reference_frame)
        result['reference_saved'] = save_option_contracts_reference(connection, reference_frame)
        logger.info('Reference done: rows=%d', result['reference_rows'])

    # --- Dataset ---
    if not skip_dataset:
        logger.info('Dataset: building IV / HV / model dataset')
        iv_daily = build_iv_daily(connection=connection, start_date=start_iso, end_date=end_iso)
        hv_daily = build_hv_daily(connection=connection, start_date=start_iso, end_date=end_iso)
        model_dataset = build_model_dataset_daily(
            connection=connection, start_date=start_iso, end_date=end_iso,
        )
        result['iv_rows'] = len(iv_daily)
        result['hv_rows'] = len(hv_daily)
        result['dataset_rows'] = len(model_dataset)
        result['iv_saved'] = save_iv_daily(connection, iv_daily)
        result['hv_saved'] = save_hv_daily(connection, hv_daily)
        result['dataset_saved'] = save_model_dataset_daily(connection, model_dataset)
        logger.info(
            'Dataset done: iv=%d hv=%d model=%d',
            result['iv_rows'], result['hv_rows'], result['dataset_rows'],
        )

    # --- Export ---
    if not skip_export:
        logger.info('Export: writing CSVs')
        result['model_csv'] = export_model_dataset_daily(
            connection=connection, output_path=model_csv,
            start_date=start_iso, end_date=end_iso,
        )
        result['iv_csv'] = export_iv_daily(
            connection=connection, output_path=iv_csv,
            start_date=start_iso, end_date=end_iso,
        )
        result['hv_csv'] = export_hv_daily(
            connection=connection, output_path=hv_csv,
            start_date=start_iso, end_date=end_iso,
        )
        logger.info('Export done: %s', result['model_csv'])

    return result
