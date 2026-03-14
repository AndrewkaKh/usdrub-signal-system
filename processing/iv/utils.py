from datetime import datetime
from typing import Any

import pandas as pd

DATE_FORMATS = [
    '%Y-%m-%d',
    '%Y-%m-%d %H:%M:%S',
    '%d.%m.%Y',
    '%d.%m.%Y %H:%M:%S',
]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if pd.notna(number) else None
    text = str(value).strip().replace(' ', '')
    if not text:
        return None
    text = text.replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    text = str(value).strip()
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors='coerce')
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def normalize_option_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        'c': 'c',
        'call': 'c',
        'p': 'p',
        'put': 'p',
        'с': 'c',
        'р': 'p',
    }
    return mapping.get(text)


def compute_time_to_expiry(as_of: datetime, expiry: datetime) -> float:
    delta_seconds = (expiry - as_of).total_seconds()
    if delta_seconds <= 0:
        return 0.0
    return delta_seconds / (365.0 * 24.0 * 60.0 * 60.0)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None