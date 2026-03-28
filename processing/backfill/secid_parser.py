from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

FULL_OPTION_CODE_PATTERN = re.compile(
    r'^(?P<asset>[A-Za-z]+)-(?P<series_month>\d{1,2})\.(?P<series_year>\d{2})M(?P<expiry>\d{6})(?P<option_letter>[CP])A(?P<strike>\d+(?:\.\d+)?)$'
)

SHORT_OPTION_CODE_PATTERN = re.compile(
    r'^(?P<asset>[A-Za-z]+)(?P<strike>\d+)(?P<settlement_type>[A-Z])(?P<month_code>[A-Z])(?P<year_digit>\d)(?P<suffix>[A-Z0-9]*)$'
)

CALL_MONTH_CODES = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
}

PUT_MONTH_CODES = {
    'M': 1, 'N': 2, 'O': 3, 'P': 4, 'Q': 5, 'R': 6,
    'S': 7, 'T': 8, 'U': 9, 'V': 10, 'W': 11, 'X': 12,
}


def _infer_full_year(year_digit: int, reference_date: date | None) -> int:
    if reference_date is None:
        return 2020 + year_digit

    decade = (reference_date.year // 10) * 10
    candidates = [
        decade - 10 + year_digit,
        decade + year_digit,
        decade + 10 + year_digit,
    ]
    return min(candidates, key=lambda x: abs(x - reference_date.year))


def normalize_option_type(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {'c', 'ca', 'call'} or 'call' in text:
        return 'call'
    if text in {'p', 'pa', 'put'} or 'put' in text:
        return 'put'

    return None


def parse_option_contract_code(contract_code: str, reference_date: date | None = None) -> dict[str, Any]:
    text = str(contract_code).strip()

    full_match = FULL_OPTION_CODE_PATTERN.match(text)
    if full_match:
        option_letter = full_match.group('option_letter')
        expiry = datetime.strptime(full_match.group('expiry'), '%d%m%y').date()

        return {
            'asset_code': full_match.group('asset'),
            'option_type': 'call' if option_letter == 'C' else 'put',
            'strike': float(full_match.group('strike')),
            'expiry': expiry.isoformat(),
            'series_month': int(full_match.group('series_month')),
            'series_year': 2000 + int(full_match.group('series_year')),
            'settlement_type': 'futures_style',
            'is_parsed': True,
            'code_format': 'full',
        }

    short_match = SHORT_OPTION_CODE_PATTERN.match(text)
    if short_match:
        month_code = short_match.group('month_code')
        year_digit = int(short_match.group('year_digit'))

        if month_code in CALL_MONTH_CODES:
            option_type = 'call'
            series_month = CALL_MONTH_CODES[month_code]
        elif month_code in PUT_MONTH_CODES:
            option_type = 'put'
            series_month = PUT_MONTH_CODES[month_code]
        else:
            option_type = None
            series_month = None

        series_year = _infer_full_year(year_digit, reference_date)

        return {
            'asset_code': short_match.group('asset'),
            'option_type': option_type,
            'strike': float(short_match.group('strike')),
            'expiry': None,
            'series_month': series_month,
            'series_year': series_year,
            'settlement_type': short_match.group('settlement_type'),
            'is_parsed': option_type is not None and series_month is not None,
            'code_format': 'short',
        }

    return {
        'asset_code': None,
        'option_type': None,
        'strike': None,
        'expiry': None,
        'series_month': None,
        'series_year': None,
        'settlement_type': None,
        'is_parsed': False,
        'code_format': None,
    }


def extract_option_type_from_code(contract_code: str, reference_date: date | None = None) -> str | None:
    parsed = parse_option_contract_code(contract_code, reference_date=reference_date)
    return parsed.get('option_type')