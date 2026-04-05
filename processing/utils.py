from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd


def normalize_date(value: Union[date, datetime, str]) -> date:
    """Канонический перевод date/datetime/str → date.

    Заменяет приватные копии _normalize_date в:
        processing/backfill/futures_loader.py
        processing/backfill/options_loader.py
        processing/backfill/candidates.py
        processing/daily_pipeline.py (_to_date)
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), '%Y-%m-%d').date()


def normalize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Нормализация DataFrame перед записью в SQLite.

    Гарантирует наличие всех ожидаемых колонок (добавляет None для отсутствующих),
    приводит к нужному порядку столбцов и заменяет NaN на None для совместимости
    с sqlite3.

    Заменяет приватные копии _normalize_frame в:
        processing/backfill/storage.py
        processing/dataset/storage.py
    """
    if frame.empty:
        return pd.DataFrame(columns=columns)

    normalized = frame.copy()

    for col in columns:
        if col not in normalized.columns:
            normalized[col] = None

    normalized = normalized[columns]
    normalized = normalized.where(pd.notnull(normalized), None)
    return normalized
