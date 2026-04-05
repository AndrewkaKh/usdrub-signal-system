from __future__ import annotations

from datetime import date, datetime


def parse_date(value: str) -> date:
    """Разбор даты из строки формата YYYY-MM-DD для argparse."""
    return datetime.strptime(value, '%Y-%m-%d').date()
