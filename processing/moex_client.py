from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

MOEX_OPTIONS_URL = 'https://iss.moex.com/iss/engines/futures/markets/options/securities.json'
MOEX_OPTION_DATA_URL = 'https://iss.moex.com/iss/engines/futures/markets/options/securities/{secid}.json'
MOEX_FUTURES_SECURITY_URL = 'https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}.json'
MOEX_SECURITY_URL = 'https://iss.moex.com/iss/securities/{secid}.json'

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 25
REQUEST_RETRIES = 3
RETRY_SLEEP_SECONDS = 1.5


class MoexRequestError(RuntimeError):
    pass


def _request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise MoexRequestError(f'MOEX request failed after {REQUEST_RETRIES} attempts: {last_error}')


def _frame_from_block(payload: dict[str, Any], block_name: str) -> pd.DataFrame:
    block = payload.get(block_name, {})
    columns = block.get('columns', [])
    data = block.get('data', [])
    return pd.DataFrame(data, columns=columns)


def _first_row_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _merge_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    marketdata = _first_row_dict(_frame_from_block(payload, 'marketdata'))
    securities = _first_row_dict(_frame_from_block(payload, 'securities'))
    merged = {}
    merged.update(securities)
    merged.update(marketdata)
    return merged


def fetch_options_table() -> pd.DataFrame:
    payload = _request_json(
        MOEX_OPTIONS_URL,
        params={'iss.meta': 'off', 'iss.only': 'securities'},
    )
    return _frame_from_block(payload, 'securities')


def fetch_option_marketdata(secid: str) -> dict[str, Any]:
    payload = _request_json(
        MOEX_OPTION_DATA_URL.format(secid=secid),
        params={'iss.meta': 'off', 'iss.only': 'marketdata'},
    )
    frame = _frame_from_block(payload, 'marketdata')
    return _first_row_dict(frame)


def fetch_futures_snapshot(secid: str) -> dict[str, Any]:
    payload = _request_json(
        MOEX_FUTURES_SECURITY_URL.format(secid=secid),
        params={'iss.meta': 'off', 'iss.only': 'marketdata,securities'},
    )
    return _merge_snapshot(payload)


def fetch_security_snapshot(secid: str) -> dict[str, Any]:
    payload = _request_json(
        MOEX_SECURITY_URL.format(secid=secid),
        params={'iss.meta': 'off', 'iss.only': 'marketdata,securities'},
    )
    return _merge_snapshot(payload)


def _extract_price_from_snapshot(snapshot: dict[str, Any]) -> tuple[float | None, str | None]:
    for field in [
        'LAST',
        'MARKETPRICE',
        'LASTTRADEPRICE',
        'SETTLEPRICE',
        'PREVSETTLEPRICE',
        'PREVPRICE',
        'OPEN',
        'LCLOSEPRICE',
        'LEGALCLOSEPRICE',
        'WAPRICE',
    ]:
        value = snapshot.get(field)
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0:
            return price, field.lower()
    return None, None


def fetch_futures_price(secid: str) -> tuple[float | None, str | None]:
    try:
        snapshot = fetch_futures_snapshot(secid)
        price, source = _extract_price_from_snapshot(snapshot)
        if price is not None:
            return price, f'forts:{source}'
    except MoexRequestError:
        pass

    try:
        snapshot = fetch_security_snapshot(secid)
        price, source = _extract_price_from_snapshot(snapshot)
        if price is not None:
            return price, f'generic:{source}'
    except MoexRequestError:
        pass

    return None, None