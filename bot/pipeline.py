"""Data collection for morning and evening bot reports.

Each function returns a payload dict consumed by bot/reports.py.
Every data block is fetched independently — failures are caught and
stored as error strings so the report degrades gracefully.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from bot.config import get_settings as _get_settings


def _cfg():
    return _get_settings()


def _get_dataset_csv() -> str:
    return _cfg().dataset_csv


def _get_db_path() -> str:
    return _cfg().db_path


def _get_smile_csv() -> str:
    return _cfg().smile_csv


def _get_smile_metrics_csv() -> str:
    return _cfg().smile_metrics_csv


def _get_smile_tmp_dir() -> str:
    return _cfg().smile_tmp_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return date.today().isoformat()


def _safe(label: str, fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return result or error string on exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.error('%s failed: %s', label, exc)
        return None


def _external_features() -> tuple[dict, bool]:
    """Read latest external features from CSV; return (row_dict, is_fresh)."""
    from processing.dataset.external_features import get_latest_external_features
    return get_latest_external_features()


def _iv_snapshot():
    from processing.iv.service import calculate_iv_snapshot
    return calculate_iv_snapshot()


def _hv_snapshot(iv_snap):
    from processing.hv.service import calculate_hv_snapshot
    return calculate_hv_snapshot(iv_snap)


def _predict(spot: float, as_of_date: str | None = None) -> dict | None:
    from model.predict import predict_next_day
    return predict_next_day(_get_dataset_csv(), spot_price=spot, confidence=0.90, as_of_date=as_of_date)


def _spot_from_db(as_of_date: str | None = None) -> float | None:
    from model.predict import get_spot_price_from_db
    return get_spot_price_from_db(_get_db_path(), as_of_date=as_of_date)


def _spot_from_snapshot(iv_snap) -> float | None:
    try:
        metrics = iv_snap.get('metrics', {})
        for tenor in ('1m', '3m', '6m'):
            fp = metrics.get(tenor, {}).get('futures_price')
            if fp and fp > 0:
                return round(fp / 1000.0, 4)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Morning payload
# ---------------------------------------------------------------------------

def morning_payload(date_str: str | None = None) -> dict:
    """Collect all data for the morning report.

    Parameters
    ----------
    date_str:
        Report date label (ISO). Defaults to today. Does not affect
        live data sources (IV/HV/spot are always fetched in real time).

    Returns a dict with the following keys (None / error_str on failure):
        timestamp, spot, iv_snap, hv_snap, ext_features, ext_fresh,
        prediction, errors
    """
    payload: dict = {
        'timestamp': datetime.now().strftime('%H:%M МСК'),
        'report_date': date_str or _today_str(),
        'spot': None,
        'iv_snap': None,
        'hv_snap': None,
        'ext_features': None,
        'ext_fresh': False,
        'prediction': None,
        'errors': [],
    }

    # 1. Real-time IV snapshot
    iv_snap = _safe('IV snapshot', _iv_snapshot)
    payload['iv_snap'] = iv_snap

    # 2. HV snapshot (needs IV to know which futures contract)
    if iv_snap and iv_snap.get('status') not in ('network_error',):
        hv_snap = _safe('HV snapshot', _hv_snapshot, iv_snap)
        payload['hv_snap'] = hv_snap

    # 3. Spot price
    spot = _spot_from_snapshot(iv_snap) if iv_snap else None
    if spot is None:
        spot = _safe('spot from DB', _spot_from_db)
    payload['spot'] = spot

    # 4. External features
    ext, is_fresh = _external_features()
    payload['ext_features'] = ext if ext else None
    payload['ext_fresh'] = is_fresh

    # 5. Model prediction
    if spot is not None and Path(_get_dataset_csv()).exists():
        payload['prediction'] = _safe('prediction', _predict, spot)
    else:
        payload['errors'].append('prediction skipped: no dataset CSV or spot price')

    return payload


# ---------------------------------------------------------------------------
# Evening payload
# ---------------------------------------------------------------------------

def evening_payload(date_str: str | None = None) -> tuple[dict, str | None]:
    """Collect all data for the evening report.

    Runs the daily pipeline (backfill + shortlist + dataset + export),
    updates external features, builds the smile, generates a PNG chart.

    Parameters
    ----------
    date_str:
        Target date (ISO). Defaults to today.

    Returns
    -------
    (payload_dict, png_path | None)
    """
    if date_str is None:
        date_str = _today_str()

    payload: dict = {
        'timestamp': datetime.now().strftime('%H:%M МСК'),
        'report_date': date_str,
        'spot': None,
        'iv_end': None,       # IV metrics from end-of-day smile
        'hv_snap': None,
        'smile_metrics': None,
        'ext_features': None,
        'ext_fresh': False,
        'prediction': None,
        'pipeline_result': None,
        'non_trading_day': False,
        'errors': [],
    }
    png_path: str | None = None

    # 1. Run daily pipeline (backfill + shortlist + dataset)
    pipeline_result = _safe('evening pipeline', _run_evening_pipeline, date_str)
    payload['pipeline_result'] = pipeline_result
    if pipeline_result is None:
        payload['errors'].append('daily pipeline failed — data may be missing')
    else:
        payload['non_trading_day'] = _is_non_trading_day(date_str)

    # 2. Update external features
    _safe('update external features', _update_ext, date_str)
    ext, is_fresh = _external_features()
    payload['ext_features'] = ext if ext else None
    payload['ext_fresh'] = is_fresh

    # 3. Build smile + metrics
    smile_df, metrics_df = _safe('smile build', _build_smile, date_str) or (None, None)
    if smile_df is not None and not smile_df.empty:
        payload['smile_metrics'] = _extract_smile_metrics(smile_df, metrics_df, date_str)
        # Also extract end-of-day IV per tenor
        payload['iv_end'] = _extract_iv_end(smile_df, date_str)

        # 4. Generate smile PNG
        png_path = _safe('smile chart', _generate_smile_png, date_str)

    # 5. Spot price — use historical date so retro reports show correct closing price
    spot = _safe('spot from DB', _spot_from_db, date_str)
    if spot is None and smile_df is not None and not smile_df.empty:
        try:
            spot = round(smile_df['futures_price'].iloc[0] / 1000.0, 4)
        except Exception:
            pass
    payload['spot'] = spot

    # 6. Model prediction (for tomorrow)
    # Pass as_of_date so historical reports use data as it was on that date.
    if spot is not None and Path(_get_dataset_csv()).exists():
        payload['prediction'] = _safe('prediction', _predict, spot, date_str)
    else:
        payload['errors'].append('prediction skipped: no dataset CSV or spot price')

    return payload, png_path


# ---------------------------------------------------------------------------
# Private helpers for evening
# ---------------------------------------------------------------------------

def _is_non_trading_day(date_str: str) -> bool:
    """Return True if date_str is a non-trading day on MOEX.

    Uses two signals:
    1. No futures rows in DB for that date (after the pipeline ran).
    2. Current Moscow time is past 19:30 — by then MOEX always publishes
       settlement data for trading days. Before 19:30 we can't be sure:
       0 rows may simply mean the session hasn't settled yet.
    """
    import sqlite3 as _sqlite3
    from datetime import timezone, timedelta

    msk = timezone(timedelta(hours=3))
    now_msk = datetime.now(msk)
    # Before 19:30 MSK settlement data is not yet guaranteed to be published
    if now_msk.hour < 19 or (now_msk.hour == 19 and now_msk.minute < 30):
        return False

    try:
        conn = _sqlite3.connect(_get_db_path())
        count = conn.execute(
            'SELECT COUNT(*) FROM futures_raw WHERE date = ?', [date_str]
        ).fetchone()[0]
        conn.close()
        return count == 0
    except Exception:
        return False


def _run_evening_pipeline(date_str: str) -> dict:
    from processing.backfill import get_connection, initialize_database
    from processing.daily_pipeline import run_daily_pipeline
    from processing.dataset.exporter import export_model_dataset_daily

    conn = get_connection()
    initialize_database(conn)
    try:
        result = run_daily_pipeline(
            connection=conn,
            start_date=date_str,
            end_date=date_str,
            skip_reference=True,
            # Skip per-date export: we export the full history below instead,
            # so today's single-row export never overwrites the full CSV.
            skip_export=True,
        )
        # Export full historical dataset so predict_next_day has all lags.
        # Use the latest date in the table, not date_str — otherwise generating
        # a report for a past date would truncate the CSV, dropping newer rows.
        row = conn.execute('SELECT MAX(date) FROM model_dataset_daily').fetchone()
        export_end = row[0] if row and row[0] else date_str
        export_model_dataset_daily(
            connection=conn,
            output_path=_get_dataset_csv(),
            start_date='2021-01-01',
            end_date=export_end,
        )
        return result
    finally:
        conn.close()


def _update_ext(date_str: str) -> None:
    from processing.dataset.external_features import update_external_features
    update_external_features(end_date=date_str)


def _build_smile(date_str: str) -> tuple:
    import sqlite3 as _sqlite3
    from processing.dataset.iv_smile_builder import build_iv_smile
    from processing.dataset.smile_metrics import compute_smile_metrics

    conn = _sqlite3.connect(_get_db_path())
    try:
        smile_df = build_iv_smile(conn, date_str, date_str)
    finally:
        conn.close()

    if smile_df.empty:
        return smile_df, pd.DataFrame()

    metrics_df = compute_smile_metrics(smile_df)

    # Persist CSVs (same paths as model_runner uses)
    smile_df.to_csv(_get_smile_csv(), index=False, sep=';')
    metrics_df.to_csv(_get_smile_metrics_csv(), index=False, sep=';')

    return smile_df, metrics_df


def _generate_smile_png(date_str: str) -> str | None:
    from data.plot_smile import plot_smile

    tmp_dir = Path(_get_smile_tmp_dir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f'smile_{date_str}.png'
    plot_smile(date_str, tenor_filter=None, out_path=out_path)
    return str(out_path) if out_path.exists() else None


def _extract_smile_metrics(smile_df: pd.DataFrame, metrics_df: pd.DataFrame, date_str: str) -> dict:
    """Extract per-tenor smile metrics as a plain dict."""
    result = {}
    if metrics_df is None or metrics_df.empty:
        return result
    day_metrics = metrics_df[metrics_df['date'] == date_str]
    for _, row in day_metrics.iterrows():
        tenor = row['tenor']
        result[tenor] = {
            'atm_vol': row.get('atm_vol'),
            'rr25': row.get('rr25'),
            'bf25': row.get('bf25'),
            'n_strikes': row.get('n_strikes'),
        }
    return result


def _extract_iv_end(smile_df: pd.DataFrame, date_str: str) -> dict:
    """Extract ATM IV per tenor from end-of-day smile."""
    result = {}
    day_df = smile_df[smile_df['date'] == date_str]
    for tenor, group in day_df.groupby('tenor'):
        atm_idx = group['moneyness'].abs().idxmin()
        result[tenor] = float(group.loc[atm_idx, 'mid_iv'])
    return result
