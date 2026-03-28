from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from .config import REQUEST_TIMEOUT_SECONDS
from .secid_parser import normalize_option_type

ISS_SECURITY_SPEC_URL = 'https://iss.moex.com/iss/securities/{secid}.json'


def _extract_named_map(block: dict[str, Any]) -> dict[str, Any]:
    columns = [str(column).strip().lower() for column in block.get('columns', [])]
    data = block.get('data', [])

    if not columns or not data:
        return {}

    if 'name' not in columns or 'value' not in columns:
        return {}

    name_idx = columns.index('name')
    value_idx = columns.index('value')

    result: dict[str, Any] = {}
    for row in data:
        if len(row) <= max(name_idx, value_idx):
            continue
        key = row[name_idx]
        value = row[value_idx]
        if key is None:
            continue
        result[str(key).strip().upper()] = value

    return result


def _extract_first_row_map(block: dict[str, Any]) -> dict[str, Any]:
    columns = [str(column).strip().upper() for column in block.get('columns', [])]
    data = block.get('data', [])

    if not columns or not data:
        return {}

    first_row = data[0]
    result: dict[str, Any] = {}

    for idx, column in enumerate(columns):
        if idx < len(first_row):
            result[column] = first_row[idx]

    return result


def _first_non_empty(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, '', '0000-00-00'):
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, ''):
        return None

    text = str(value).replace(' ', '').replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _to_iso_date(value: Any) -> str | None:
    if value in (None, '', '0000-00-00'):
        return None

    text = str(value).strip()

    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%Y%m%d'):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def _load_missing_secids(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            c.secid,
            MAX(c.option_type) AS option_type,
            MAX(c.strike) AS strike,
            MAX(c.series_month) AS series_month,
            MAX(c.series_year) AS series_year
        FROM option_contract_candidates c
        LEFT JOIN option_contracts_reference r
            ON c.secid = r.secid
        WHERE c.date BETWEEN ? AND ?
          AND (
              r.secid IS NULL
              OR r.expiry IS NULL
              OR r.expiry = ''
              OR r.underlying_secid IS NULL
              OR r.underlying_secid = ''
          )
        GROUP BY c.secid
        ORDER BY c.secid
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def _fetch_security_metadata(secid: str) -> dict[str, Any]:
    response = requests.get(
        ISS_SECURITY_SPEC_URL.format(secid=secid),
        params={'iss.meta': 'off'},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    description_map = _extract_named_map(payload.get('description', {}))
    securities_map = _extract_first_row_map(payload.get('securities', {}))
    metadata = {**securities_map, **description_map}

    raw_option_type = _first_non_empty(metadata, 'OPTIONTYPE', 'OPTTYPE', 'PUTCALL')
    raw_asset_code = _first_non_empty(metadata, 'ASSETCODE')
    raw_underlying_secid = _first_non_empty(metadata, 'UNDERLYINGASSET', 'UNDERLYINGSECID', 'UNDERLYING')
    raw_strike = _first_non_empty(metadata, 'STRIKE')
    raw_expiry = _first_non_empty(metadata, 'LSTDELDATE', 'LSTTRADE', 'EXPIRATIONDATE', 'EXPIRYDATE', 'MATDATE', 'EXPDATE')
    raw_shortname = _first_non_empty(metadata, 'SHORTNAME', 'SECNAME', 'LATNAME', 'NAME')
    raw_series_month = _first_non_empty(metadata, 'SERIESMONTH')
    raw_series_year = _first_non_empty(metadata, 'SERIESYEAR')

    return {
        'secid': secid,
        'asset_code_meta': raw_asset_code,
        'underlying_secid_meta': raw_underlying_secid,
        'option_type_meta': normalize_option_type(raw_option_type),
        'strike_meta': _to_float(raw_strike),
        'expiry_meta': _to_iso_date(raw_expiry),
        'shortname_meta': raw_shortname,
        'series_month_meta': _to_int(raw_series_month),
        'series_year_meta': _to_int(raw_series_year),
    }


def _safe_fetch_security_metadata(secid: str) -> dict[str, Any]:
    try:
        return _fetch_security_metadata(secid)
    except Exception:
        return {
            'secid': secid,
            'asset_code_meta': None,
            'underlying_secid_meta': None,
            'option_type_meta': None,
            'strike_meta': None,
            'expiry_meta': None,
            'shortname_meta': None,
            'series_month_meta': None,
            'series_year_meta': None,
        }


def _build_reference_rows(missing_secids: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    if missing_secids.empty:
        return pd.DataFrame()

    secids = missing_secids['secid'].dropna().astype(str).tolist()
    metadata_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_safe_fetch_security_metadata, secid): secid for secid in secids}

        for future in as_completed(future_map):
            metadata_rows.append(future.result())

    metadata_frame = pd.DataFrame(metadata_rows)
    merged = missing_secids.merge(metadata_frame, on='secid', how='left')

    result = pd.DataFrame()
    result['secid'] = merged['secid']
    result['asset_code'] = merged['asset_code_meta']
    result['option_type'] = merged['option_type_meta'].combine_first(merged['option_type'])
    result['strike'] = merged['strike_meta'].combine_first(pd.to_numeric(merged['strike'], errors='coerce'))

    series_month_base = pd.to_numeric(merged['series_month'], errors='coerce')
    series_year_base = pd.to_numeric(merged['series_year'], errors='coerce')

    result['series_month'] = merged['series_month_meta'].where(merged['series_month_meta'].notna(), series_month_base)
    result['series_year'] = merged['series_year_meta'].where(merged['series_year_meta'].notna(), series_year_base)

    result['expiry'] = merged['expiry_meta']
    result['underlying_secid'] = merged['underlying_secid_meta']
    result['shortname'] = merged['shortname_meta']
    result['updated_at'] = datetime.utcnow().isoformat(timespec='seconds')

    return result.drop_duplicates(subset=['secid']).reset_index(drop=True)


def build_missing_contract_references(
    connection,
    start_date: str,
    end_date: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    missing_secids = _load_missing_secids(connection, start_date, end_date)
    return _build_reference_rows(missing_secids, max_workers=max_workers)