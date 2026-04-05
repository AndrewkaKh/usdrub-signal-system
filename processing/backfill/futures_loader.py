from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .config import BASE_ASSET_CODE, FUTURES_HISTORY_URL, ISS_PAGE_SIZE_HINT, REQUEST_TIMEOUT_SECONDS
from ..utils import normalize_date


def _coerce_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _extract_history_block(payload: dict[str, Any]) -> pd.DataFrame:
    history = payload.get('history', {})
    columns = history.get('columns', [])
    data = history.get('data', [])
    if not columns or not data:
        return pd.DataFrame()
    return pd.DataFrame(data, columns=columns)


def _fetch_history_page(session: requests.Session, trading_date: date, start: int = 0) -> pd.DataFrame:
    params = {
        'date': trading_date.isoformat(),
        'start': start,
        'iss.meta': 'off',
    }
    response = session.get(FUTURES_HISTORY_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    return _extract_history_block(payload)


def _fetch_full_history_for_date(session: requests.Session, trading_date: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start = 0

    while True:
        page = _fetch_history_page(session, trading_date, start=start)
        if page.empty:
            break

        page = page.dropna(how='all')
        if not page.empty:
            frames.append(page)

        if len(page) < ISS_PAGE_SIZE_HINT:
            break

        start += len(page)

    if not frames:
        return pd.DataFrame()

    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()

    return pd.concat(non_empty_frames, ignore_index=True)


def _standardize_futures_frame(frame: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]

    if 'secid' not in data.columns:
        return pd.DataFrame()

    data['secid'] = data['secid'].astype(str).str.strip()
    data = data[data['secid'].str.startswith(BASE_ASSET_CODE, na=False)].copy()

    if data.empty:
        return pd.DataFrame()

    row_count = len(data)

    result = pd.DataFrame(index=data.index)
    result['date'] = [trading_date.isoformat()] * row_count
    result['secid'] = data['secid'].values
    result['shortname'] = data['shortname'].values if 'shortname' in data.columns else [None] * row_count
    result['tradedate'] = data['tradedate'].values if 'tradedate' in data.columns else [trading_date.isoformat()] * row_count
    result['last_price'] = _coerce_float_series(data['last']).values if 'last' in data.columns else [None] * row_count
    result['settlement_price'] = _coerce_float_series(data['settleprice']).values if 'settleprice' in data.columns else [None] * row_count
    result['open_price'] = _coerce_float_series(data['open']).values if 'open' in data.columns else [None] * row_count
    result['high_price'] = _coerce_float_series(data['high']).values if 'high' in data.columns else [None] * row_count
    result['low_price'] = _coerce_float_series(data['low']).values if 'low' in data.columns else [None] * row_count
    result['volume'] = _coerce_float_series(data['volume']).values if 'volume' in data.columns else [None] * row_count
    result['open_interest'] = _coerce_float_series(data['openposition']).values if 'openposition' in data.columns else [None] * row_count
    result['num_trades'] = _coerce_float_series(data['numtrades']).values if 'numtrades' in data.columns else [None] * row_count

    return result.reset_index(drop=True)


def fetch_futures_for_date(session: requests.Session, trading_date: date | datetime | str) -> pd.DataFrame:
    normalized_date = normalize_date(trading_date)
    raw = _fetch_full_history_for_date(session, normalized_date)
    return _standardize_futures_frame(raw, normalized_date)


def load_futures_backfill(start_date: date | datetime | str, end_date: date | datetime | str) -> pd.DataFrame:
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    frames: list[pd.DataFrame] = []
    current = start_dt

    with requests.Session() as session:
        while current <= end_dt:
            print(f'[futures] loading {current.isoformat()}')
            frame = fetch_futures_for_date(session, current)
            print(f'[futures] rows={len(frame)} for {current.isoformat()}')
            if not frame.empty:
                frames.append(frame)
            current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=['date', 'secid']).reset_index(drop=True)
    return result